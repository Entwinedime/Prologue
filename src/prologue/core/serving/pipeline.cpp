#include "prologue/core/serving/pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>

namespace prologue::serving {
namespace {

using nlohmann::json;
using Index = std::size_t;

json identity(const json& action) {
    auto members = json::array();
    for (const auto& member : action.at("members")) {
        members.push_back(json::array({member.at("request"), member.at("kv_committed_tokens"),
                                       member.at("output_tokens")}));
    }
    return json::array({action.at("phase"), std::move(members)});
}

Nanoseconds plus(Nanoseconds start, Nanoseconds duration) {
    if (duration < 0 || duration > std::numeric_limits<Nanoseconds>::max() - start) {
        throw Stopped("timing_gap", "invalid elapsed time for pipeline work");
    }
    return start + duration;
}

}

class Simulation::Pipeline::Program final {
public:
    Program(json action, json graph, ProgramTiming timing, std::size_t width,
            std::map<Queue, Nanoseconds>& lanes, bool record)
        : action_(std::move(action)), graph_(std::move(graph)), timing_(std::move(timing)),
          steps_(graph_.at("steps").size()), ranks_(width), intervals_(graph_.at("work").size()),
          lanes_(lanes), record_(record) {
        for (Index index = 0; index < steps_.size(); ++index) {
            const auto& step = graph_.at("steps").at(index);
            auto& state = steps_.at(index);
            const auto stage = step.at("stage").get<Index>();
            const auto name = step.at("name").get<std::string>();
            names_.emplace(std::pair{stage, name}, index);
            state.ranks = step.at("ranks").get<std::vector<Index>>();
            state.results.resize(state.ranks.size());
            for (const auto rank : state.ranks) {
                ranks_.at(stage).insert(rank);
            }
        }
        for (const auto& call : graph_.at("calls")) {
            const auto& path = call.at("path");
            if (path.size() != 2) {
                continue;
            }
            auto& step = steps_.at(path.front().get<Index>());
            const auto& operation = call.at("operation");
            step.signal = operation == "signal_record" || operation == "signal_wait";
            step.blocking = operation == "signal_wait" && call.at("parameters").at("host") == true;
            step.device_fence = (operation == "signal_wait" && !step.blocking) || operation == "transfer_wait";
        }
        for (const auto& edge : graph_.at("dependencies")) {
            const auto& source = edge.at(0);
            const auto& target = edge.at(1);
            const auto from = source.at("step").get<Index>();
            auto& to = steps_.at(target.at("step").get<Index>());
            to.inputs.push_back({from, position(steps_.at(from), source.at("rank")),
                                 position(to, target.at("rank"))});
        }
        for (const auto& wire : graph_.at("transfers")) {
            const auto source = wire.at("source").at("step").get<Index>();
            const auto target = wire.at("target").at("step").get<Index>();
            auto& route = routes_[source];
            route.target = graph_.at("steps").at(target).at("stage");
            route.waits.insert(wire.at("sender_wait").get<Index>());
        }
        for (const auto& lane : graph_.at("lanes")) {
            queues_.emplace_back(lane.at(0), lane.at(1), lane.at(2));
        }
    }

    void affirm(Index actor, const json& action, const json& resources) {
        if (identity(action_) != identity(action)) {
            throw Stopped("action_mismatch", "pipeline stages selected different microbatch members");
        }
        if (record_) {
            actor_states_[std::to_string(actor)] = {{"members", action.at("members")}, {"resources", resources}};
        }
    }

    Nanoseconds cut(Index actor, const std::string& name, Nanoseconds now, bool open = false) {
        const auto index = locate(actor, name);
        apply(index, now);
        return open ? entry(index) : returned(index);
    }

    void send(Index actor, Index peer, const std::string& kind, Timeline& clock, Nanoseconds& posted) {
        const auto cut = locate(actor, "send:" + kind);
        if (routes_.at(cut).target != peer) {
            throw Stopped("action_mismatch", "native send disagrees with the bound pipeline peer");
        }
        apply(cut, clock.now());
        posted = entry(cut);
        clock.advance_to(returned(cut));
    }

    void wait(Index actor, const std::string& kind, Timeline& clock) {
        for (const auto cut : routes_.at(locate(actor, "send:" + kind)).waits) {
            apply(cut, clock.now());
            clock.advance_to(returned(cut));
        }
    }

    void finish(Index actor, const json& resources, Timeline& clock) {
        const auto index = locate(actor, "process");
        clock.advance_to(std::max(clock.now(), returned(index)));
        finished_.insert(actor);
        if (record_) {
            actor_states_.at(std::to_string(actor))["after_result"] = resources;
        }
    }

    Nanoseconds publication(Index actor, const json& call, Nanoseconds now) const {
        const auto publisher = graph_.at("calls").front().at("parameters").at("publication_rank").get<Index>();
        if (!ranks_.at(actor).contains(publisher)) {
            throw Stopped("action_mismatch", "a non-publishing stage emitted a token");
        }
        return std::max(now, timing_.publications.at(call.at("request").get<std::string>()));
    }

    bool done() const {
        return finished_.size() == ranks_.size() && std::ranges::all_of(steps_, &Step::issued);
    }

    json response() {
        json report{{"phase", action_.at("phase")}, {"members", action_.at("members")},
                    {"publications", std::move(publications)}};
        if (record_) {
            report["actors"] = std::move(actor_states_);
            report["cuts"] = std::move(receipts_);
            report["graph"] = std::move(graph_);
            auto work = json::array();
            for (const auto& interval : intervals_) {
                work.push_back({interval->begin_ns, interval->end_ns});
            }
            report["intervals"] = std::move(work);
        }
        return report;
    }

    json publications = json::array();

private:
    struct Boundary {
        Nanoseconds began_ns;
        Nanoseconds returned_ns;
        Nanoseconds ready_ns;
    };

    struct Edge {
        Index step;
        Index peer;
        Index local;
    };

    struct Step {
        std::vector<Index> ranks;
        std::vector<Edge> inputs;
        std::vector<Boundary> results;
        bool signal = false;
        bool blocking = false;
        bool device_fence = false;
        bool issued = false;
    };

    struct Route {
        Index target;
        std::set<Index> waits;
    };

    static Index position(const Step& step, Index rank) {
        const auto member = std::ranges::find(step.ranks, rank);
        if (member == step.ranks.end()) {
            throw Stopped("action_mismatch", "a pipeline dependency uses an inactive rank");
        }
        return static_cast<Index>(member - step.ranks.begin());
    }

    Index locate(Index actor, const std::string& name) const {
        const auto found = names_.find({actor, name});
        if (found == names_.end()) {
            throw Stopped("action_mismatch", "native stage " + std::to_string(actor) + " issued uncovered cut " + name);
        }
        return found->second;
    }

    Nanoseconds entry(Index index) const {
        return steps_.at(index).results.front().began_ns;
    }

    Nanoseconds returned(Index index) const {
        return steps_.at(index).results.front().returned_ns;
    }

    void apply(Index index, Nanoseconds now) {
        auto& state = steps_.at(index);
        if (state.issued) {
            throw Stopped("action_mismatch", "a pipeline cut was issued more than once");
        }

        std::vector<Nanoseconds> ready(state.ranks.size());
        std::vector<Nanoseconds> host(state.ranks.size(), 0);

        // The leading rank drives the replicated native policy. Its peers keep their own measured gates.
        host.front() = now;
        for (const auto& edge : state.inputs) {
            const auto& producer = steps_.at(edge.step);
            if (!producer.issued) {
                if (!producer.signal) {
                    throw Stopped("action_mismatch", "native pipeline issued a cut before its producer");
                }
                apply(edge.step, 0);
            }
            const auto& boundary = producer.results.at(edge.peer);
            ready.at(edge.local) = std::max(ready.at(edge.local), boundary.ready_ns);
            if (producer.blocking) {
                host.at(edge.local) = std::max(host.at(edge.local), boundary.returned_ns);
            }
        }
        for (Index local = 0; local < state.ranks.size(); ++local) {
            const auto& observation = timing_.steps.at(index).at(local);
            const auto began = std::max(host.at(local), observation.host.begin_ns);
            const auto device_ready = std::max(ready.at(local), observation.ready_ns);
            auto returned = std::max(began, observation.host.end_ns);
            if (state.blocking) {
                returned = std::max(returned, device_ready);
            }
            state.results.at(local) = {began, returned, device_ready};
        }

        // A cut may return to native control while its device streams still have queued work.
        const auto& cut = graph_.at("steps").at(index);
        if (state.device_fence) {
            const auto queue = cut.at("queue").get<std::string>();
            for (Index local = 0; local < state.ranks.size(); ++local) {
                auto& lane = lanes_[{state.ranks[local], queue, 0}];
                lane = std::max(lane, state.results.at(local).ready_ns);
            }
        }
        const auto first = cut.at("first_work").get<Index>();
        const auto end = first + cut.at("work_count").get<Index>();
        for (Index index_work = first; index_work < end; ++index_work) {
            const auto& work = graph_.at("work").at(index_work);
            const auto& sample = timing_.work.at(index_work);
            const auto local = position(state, work.at("rank"));
            auto& lane = lanes_[queues_.at(work.at("lane").get<Index>())];
            auto begin = std::max({sample.begin_ns, ready.at(local), lane, state.results.at(local).began_ns});
            if (work.contains("after")) {
                for (const auto& prior : work.at("after")) {
                    begin = std::max(begin, intervals_.at(prior.get<Index>()).value().end_ns);
                }
            }
            lane = plus(begin, sample.end_ns - sample.begin_ns);
            intervals_.at(index_work) = Interval{begin, lane};
            auto& result = state.results.at(local);
            result.ready_ns = std::max(result.ready_ns, lane);
        }
        state.issued = true;
        if (record_) {
            auto boundaries = json::array();
            for (const auto& result : state.results) {
                boundaries.push_back({result.began_ns, result.returned_ns, result.ready_ns});
            }
            receipts_.push_back({{"step", index}, {"boundaries", std::move(boundaries)}});
        }
    }

    json action_;
    json graph_;
    ProgramTiming timing_;
    std::vector<Step> steps_;
    std::vector<std::set<Index>> ranks_;
    std::vector<std::optional<Interval>> intervals_;
    std::map<Queue, Nanoseconds>& lanes_;
    std::vector<Queue> queues_;
    std::map<std::pair<Index, std::string>, Index> names_;
    std::map<Index, Route> routes_;
    std::set<Index> finished_;
    bool record_;
    json actor_states_ = json::object();
    json receipts_ = json::array();
};

Simulation::Pipeline::Pipeline(Simulation& session) : Group(session) {}
Simulation::Pipeline::~Pipeline() = default;

bool Simulation::Pipeline::done() const {
    return session_.finished_ == session_.workload_.requests().size() && programs_.empty();
}

void Simulation::Pipeline::report(json& output) {
    output["peak_inflight"] = peak_inflight_;
}

Nanoseconds Simulation::Pipeline::drained_time(Nanoseconds actor_end) const {
    for (const auto& lane : lanes_) {
        actor_end = std::max(actor_end, lane.second);
    }
    return actor_end;
}

std::optional<Nanoseconds> Simulation::Pipeline::eligible(Index actor, const json& call) const {
    auto time = now();
    const auto& operation = call.at("operation");
    if (operation == "receive") {
        const auto channel = Channel{call.at("source"), actor, call.at("channel")};
        const auto box = inboxes_.find(channel);
        if (box == inboxes_.end() || box->second.empty()) {
            return std::nullopt;
        }
        time = std::max(time, deliveries_.at(box->second.front()).posted_ns);
    } else if (operation == "wait_send") {
        const auto& message = deliveries_.at(call.at("delivery"));
        if (std::get<2>(message.channel) == "requests" && !message.received) {
            return std::nullopt;
        }
    } else if (operation == "idle" && actor == 0 && programs_.empty()) {
        const auto arrival = session_.next_arrival(actor);
        if (!arrival) {
            return std::nullopt;
        }
        time = *arrival;
    }
    return time;
}

json Simulation::Pipeline::dispatch(Index actor, const json& call) {
    const auto& op = call.at("operation");
    auto& clock = session_.actors_.at(actor).clock;
    if (op == "arrivals") {
        return session_.arrivals(actor, false);
    }
    if (op == "launch") {
        return launch(actor, call);
    }
    if (op == "send") {
        return send(actor, call);
    }
    if (op == "receive") {
        return receive(actor, call);
    }
    if (op == "wait_send") {
        wait_send(actor, call.at("delivery"));
        return nullptr;
    }
    if (op == "cut") {
        const auto name = call.at("cut").get<std::string>();
        clock.advance_to(programs_.at(call.at("program"))->cut(actor, name, clock.now(), name == "process"));
        return nullptr;
    }
    if (op == "publish") {
        auto& program = *programs_.at(call.at("program"));
        clock.advance_to(program.publication(actor, call, clock.now()));
        session_.record_publication(call, clock.now(), program.publications);
        return nullptr;
    }
    if (op == "result_finished") {
        programs_.at(call.at("program"))->finish(actor, session_.capacity(actor), clock);
        return nullptr;
    }
    if (op == "idle") {
        return nullptr;
    }
    return session_.resources(actor, call);
}

json Simulation::Pipeline::launch(Index actor, const json& call) {
    session_.require_residence(actor, call);
    auto program_id = next_program_;
    if (call.at("program").is_null()) {
        if (actor != 0) {
            throw Stopped("action_mismatch", "a downstream stage launched without its input");
        }
        auto graph = session_.templates_.expand(structure::batch_geometry(call.at("phase"), call.at("members")));
        auto timing = session_.timing_source_.for_program(call, graph);
        programs_.emplace(next_program_++, std::make_unique<Program>(call, std::move(graph), std::move(timing),
            session_.actors_.size(), lanes_, session_.record_structure_));
        peak_inflight_ = std::max(peak_inflight_, programs_.size());
    } else {
        program_id = call.at("program");
    }
    auto& program = *programs_.at(program_id);
    program.affirm(actor, call, session_.capacity(actor));
    auto& clock = session_.actors_.at(actor).clock;
    clock.advance_to(program.cut(actor, call.at("cut"), clock.now()));
    return program_id;
}

json Simulation::Pipeline::send(Index actor, const json& call) {
    const auto target = call.at("target").get<Index>();
    const auto channel = call.at("channel").get<std::string>();
    const auto& payload = call.at("payload");
    auto& clock = session_.actors_.at(actor).clock;
    auto posted = clock.now();
    if (channel == "tensors") {
        programs_.at(payload.at("program"))->send(actor, target, payload.at("kind"), clock, posted);
    }
    const auto identity = next_delivery_++;
    auto key = Channel{actor, target, channel};
    inboxes_[key].push_back(identity);
    deliveries_.emplace(identity, Delivery{std::move(key), payload, posted});
    return identity;
}

json Simulation::Pipeline::receive(Index actor, const json& call) {
    const auto key = Channel{call.at("source"), actor, call.at("channel")};
    auto& queue = inboxes_.at(key);
    const auto identity = queue.front();
    queue.pop_front();
    auto& delivery = deliveries_.at(identity);
    auto payload = std::move(delivery.payload);
    if (std::get<2>(key) == "tensors") {
        auto& clock = session_.actors_.at(actor).clock;
        clock.advance_to(programs_.at(payload.at("program"))->cut(actor,
            "receive:" + payload.at("kind").get<std::string>(), clock.now()));
    }
    delivery.received = true;
    if (delivery.waited) {
        deliveries_.erase(identity);
    } else if (std::get<2>(key) == "tensors") {
        delivery.payload = {{"program", payload.at("program")}, {"kind", payload.at("kind")}};
    }
    return payload;
}

void Simulation::Pipeline::wait_send(Index actor, Index identity) {
    auto& delivery = deliveries_.at(identity);
    if (actor != std::get<0>(delivery.channel) || delivery.waited) {
        throw Stopped("action_mismatch", "native send wait has no outstanding delivery");
    }
    if (std::get<2>(delivery.channel) == "tensors") {
        const auto& payload = delivery.payload;
        programs_.at(payload.at("program"))->wait(actor, payload.at("kind"), session_.actors_.at(actor).clock);
    }
    delivery.waited = true;
    if (delivery.received) {
        deliveries_.erase(identity);
    }
}

void Simulation::Pipeline::settle() {
    for (auto position = programs_.begin(); position != programs_.end();) {
        if (position->second->done()) {
            session_.actions_.push_back(position->second->response());
            position = programs_.erase(position);
        } else {
            ++position;
        }
    }
}

}
