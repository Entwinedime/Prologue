#include "prologue/core/serving/simulation.hpp"
#include "prologue/core/serving/disaggregation.hpp"
#include "prologue/core/serving/pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace prologue::serving {
namespace {

using nlohmann::json;

}

Stopped::Stopped(std::string status, const std::string& reason) : std::runtime_error(reason), status_(std::move(status)) {}

const std::string& Stopped::status() const {
    return status_;
}

Simulation::Simulation(const json& job, Workload workload, TimingSource& timing_source)
    : job_(job), workload_(std::move(workload)),
      templates_(job.at("templates"), job.at("domain"), job.at("rules").at("templates")),
      timing_source_(timing_source), record_structure_(job.value("record_structure", false)),
      prefix_cache_(job.at("domain").at("configuration").value("prefix_cache", false)) {
    for (const auto& request : workload_.requests()) {
        output_limits_.emplace(request.id, request.output_tokens);
        output_counts_.emplace(request.id, 0);
    }

    const auto& config = job.at("domain").at("configuration");
    const auto page_size = config.at("page_size").get<std::int64_t>();
    const bool disaggregated = config.contains("disaggregation");
    if (disaggregated) {
        for (const auto& pool : job.at("actor_resources")) {
            actors_.emplace_back(workload_, ResourceLimits{pool.at("slots"), pool.at("kv_tokens"), page_size,
                pool.at("sort_released_pages")});
        }
    } else {
        const auto width = config.value("pp", Index{1});
        const ResourceLimits limits{config.at("max_running_requests"), config.at("max_total_tokens"), page_size};
        actors_.reserve(width);
        for (Index stage = 0; stage < width; ++stage) {
            actors_.emplace_back(workload_, limits);
        }
    }

    if (const auto ingress = job.find("measured_ingress"); ingress != job.end()) {
        measure_arrivals(ingress->at("arrivals"));
    }
    if (disaggregated) {
        group_ = std::make_unique<Disaggregation>(*this, page_size);
    } else if (actors_.size() > 1) {
        group_ = std::make_unique<Pipeline>(*this);
    }
}

Simulation::~Simulation() = default;

Nanoseconds Simulation::now() const {
    return group_ ? group_->now() : actors_.front().clock.now();
}

bool Simulation::done() const {
    return group_ ? group_->done() : finished_ == workload_.requests().size() && !actors_.front().action;
}

json Simulation::capacity(Index actor) const {
    const auto& pool = actors_.at(actor).resources;
    json capacity{{"slots", pool.available_slots()}, {"kv_tokens", pool.available_tokens()}};
    if (prefix_cache_) {
        capacity["evictable_tokens"] = pool.evictable_tokens();
        capacity["protected_tokens"] = pool.protected_tokens();
    }
    return capacity;
}

void Simulation::record_cache_effect(Index actor, const char* operation, const std::vector<std::int64_t>& pages) {
    if (record_structure_ && !pages.empty()) {
        const auto started = actions_.size() + static_cast<std::size_t>(actors_.at(actor).action.has_value());
        auto ordered = pages;
        std::ranges::sort(ordered);
        cache_effects_.push_back({{"after_action", started}, {"operation", operation}, {"pages", std::move(ordered)}});
    }
}

void Simulation::measure_arrivals(const json& sources) {
    if (sources.size() != actors_.size()) {
        throw Stopped("timing_gap", "measured ingress must cover every actor");
    }
    for (Index actor = 0; actor < actors_.size(); ++actor) {
        auto& arrivals = actors_.at(actor).measured_arrivals;
        for (const auto& request : workload_.requests()) {
            const auto instant = sources.at(actor).at(request.id).get<Nanoseconds>();
            if (instant < 0 || (!arrivals.empty() && instant < arrivals.back())) {
                throw Stopped("timing_gap", "measured ingress disagrees with workload order");
            }
            arrivals.push_back(instant);
        }
    }
}

std::optional<Nanoseconds> Simulation::next_arrival(Index actor) const {
    const auto& state = actors_.at(actor);
    const auto next = state.next_arrival;
    const auto& requests = workload_.requests();
    if (next == requests.size()) {
        return std::nullopt;
    }
    return state.measured_arrivals.empty() ? requests.at(next).arrival_ns : state.measured_arrivals.at(next);
}

void Simulation::require_drained() const {
    if (!std::ranges::all_of(actors_, [](const Actor& actor) { return actor.resources.drained(); })) {
        throw Stopped("resource_error", "terminal resources are not fully returned");
    }
}

json Simulation::arrivals(Index index, bool finished) {
    auto arrivals = json::array();
    auto& actor = actors_.at(index);
    if (actor.read) {
        actor.clock.advance_to(std::max(actor.clock.now(), actor.read->earliest_ns));
        actions_.at(actor.read->preceding_action)["next_read_ns"] = actor.clock.now();
        actor.read.reset();
    }

    const auto& requests = workload_.requests();
    while (const auto arrival = next_arrival(index)) {
        if (*arrival > actor.clock.now()) {
            break;
        }
        const auto& request = requests.at(actor.next_arrival++);
        arrivals.push_back({{"id", request.id}, {"prompt_tokens", request.prompt_tokens},
                            {"output_tokens", request.output_tokens}});
    }
    if (finished) {
        require_drained();
    }
    return {{"requests", std::move(arrivals)}, {"done", finished}};
}

json Simulation::resources(Index actor, const json& message) {
    const auto operation = message.at("operation").get<std::string>();
    auto& pool = actors_.at(actor).resources;
    try {
        if (operation == "capacity") {
            return capacity(actor);
        }
        if (operation == "free_slots") {
            return pool.free_slots();
        }
        if (operation == "bind_slots") {
            return pool.bind_slots(message.at("requests"));
        }
        if (operation == "grow") {
            std::vector<KvGrowth> members;
            for (const auto& member : message.at("members")) {
                members.push_back({member.at("request"), member.at("begin"), member.at("end"),
                    member.value("prefix_pages", std::vector<std::int64_t>{})});
            }
            return pool.grow(members);
        }
        if (operation == "release_slot") {
            pool.release_slot(message.at("request"));
            return nullptr;
        }
        if (operation == "release_pages") {
            record_cache_effect(actor, "evict", pool.release_pages(message.at("pages")));
            return nullptr;
        }
        if (operation == "retain_prefix") {
            const auto pages = message.at("pages").get<std::vector<std::int64_t>>();
            pool.retain_prefix(pages);
            record_cache_effect(actor, "store", pages);
            return nullptr;
        }
        if (operation == "pin_prefix") {
            return pool.pin_prefix(message.at("pages"), message.at("acquire"));
        }
        if (operation == "rebind_prefix") {
            pool.rebind_prefix(message.at("request"), message.at("pages"));
            return nullptr;
        }
    } catch (const std::exception& error) {
        throw Stopped("resource_error", error.what());
    }
    throw Stopped("controller_error", "unknown operation " + operation);
}

void Simulation::require_residence(Index actor, const json& message) const {
    for (const auto& member : message.at("members")) {
        const auto request = member.at("request").get<std::string>();
        const auto& residence = actors_.at(actor).resources.residence(request);
        if (!residence.slot || *residence.slot != member.at("slot") || residence.tokens != member.at("kv_committed_tokens") ||
            residence.tokens != member.at("kv_reserved_tokens")) {
            throw Stopped("resource_error", "native batch does not agree with its resource binding for " + request);
        }
    }
}

json Simulation::cache(Index actor, const json& message) const {
    auto rows = json::array();
    auto written = json::array();
    auto prefix = json::array();
    const auto& pool = actors_.at(actor).resources;
    for (const auto& member : message.at("members")) {
        const auto request = member.at("request").get<std::string>();
        auto locations = pool.locations(request);
        const auto begin = structure::batch_read_tokens(message.at("phase"), member);
        if (begin < 0 || static_cast<Index>(begin) >= locations.size()) {
            throw Stopped("resource_error", "batch KV read prefix leaves no write extent");
        }
        for (auto token = static_cast<Index>(begin); token < locations.size(); ++token) {
            written.push_back(locations.at(token));
        }
        rows.push_back(std::move(locations));
        if (prefix_cache_) {
            prefix.push_back(pool.pinned_prefix(request));
        }
    }
    json cache{{"rows", std::move(rows)}, {"written", std::move(written)}};
    if (prefix_cache_) {
        cache["prefix_tokens"] = std::move(prefix);
    }
    return cache;
}

void Simulation::execute(Index index, const json& message) {
    auto& actor = actors_.at(index);
    if (actor.action) {
        throw Stopped("controller_error", "native action began before the preceding result finished");
    }
    if (actor.read) {
        throw Stopped("action_mismatch", "native action began before a measured ingress boundary");
    }
    require_residence(index, message);

    const auto query = message.contains("query")
        ? message.at("query")
        : structure::batch_geometry(message.at("phase"), message.at("members"), message.value("expert_routes", json{}));
    auto graph = templates_.expand(query);

    // Only a fully selected action can query time. The provider cannot choose the graph.
    auto action = message;
    action["clock_ns"] = actor.clock.now();
    auto timing = timing_source_.for_action(action, graph);
    actor.clock.advance_by(timing.prepare_ns);
    const auto started = actor.clock.now();
    std::optional<std::int64_t> publisher;
    if (graph.contains("collectives")) {
        publisher = graph.at("calls").front().at("parameters").at("publication_rank").get<std::int64_t>();
    }
    const auto completed = actor.clock.schedule_work(timing.work, record_structure_, publisher);
    actor.clock.advance_by(timing.model_return_ns);

    // A measured host return is independent of device work relaying tokens to the next iteration.
    if (timing.await_output_lane) {
        actor.clock.advance_to(std::max(actor.clock.now(), completed.output_lane_ready_ns));
    }

    json record{
        {"phase", message.at("phase")}, {"members", message.at("members")}, {"began_ns", started},
        {"model_ready_ns", actor.clock.now()}, {"publications", json::array()},
    };
    if (message.contains("expert_routes")) {
        record["expert_routes"] = message.at("expert_routes");
    }
    if (record_structure_) {
        auto observed = json::array();
        for (const auto& interval : completed.intervals) {
            observed.push_back({interval.begin_ns, interval.end_ns});
        }
        record["resources"] = capacity(index);
        record["graph"] = std::move(graph);
        record["intervals"] = std::move(observed);
        if (job_.value("record_cache", false)) {
            record["cache"] = cache(index, message);
        }
    }
    actor.action.emplace(ActiveAction{std::move(record), std::move(timing), completed.all_work_ready_ns});
}

void Simulation::publish(Index index, const json& message) {
    auto& actor = actors_.at(index);
    if (!actor.action) {
        throw Stopped("controller_error", "publication has no executing action");
    }
    auto& action = *actor.action;
    const auto request = message.at("request").get<std::string>();
    const auto offset = action.timing.publications.find(request);
    if (offset == action.timing.publications.end()) {
        throw Stopped("timing_gap", "missing publication readiness for " + request);
    }

    auto& record = action.record;
    actor.clock.satisfy(record.at("began_ns"), record.at("model_ready_ns"), offset->second);
    record_publication(message, actor.clock.now(), record.at("publications"));
}

void Simulation::record_publication(const json& message, Nanoseconds time_ns, json& target) {
    const auto request = message.at("request").get<std::string>();
    const auto count = message.at("count").get<std::int64_t>();
    auto& previous = output_counts_.at(request);
    if (count <= previous || count > output_limits_.at(request) ||
        message.at("finished").get<bool>() != (count == output_limits_.at(request))) {
        throw Stopped("action_mismatch", "native publication is outside the expected token extent");
    }
    target.push_back({{"request", request}, {"count", count}, {"at_ns", time_ns}});
    previous = count;
    if (message.at("finished").get<bool>()) {
        ++finished_;
    }
}

void Simulation::finish_action(Index index) {
    auto& actor = actors_.at(index);
    if (!actor.action) {
        throw Stopped("controller_error", "native result has no executing action");
    }
    auto& action = *actor.action;
    auto& record = action.record;
    auto& clock = actor.clock;
    clock.satisfy(record.at("began_ns"), record.at("model_ready_ns"), action.timing.result_finished);

    // Other ranks may still be finishing when the publication owner emits a token.
    clock.advance_to(std::max(clock.now(), action.work_ready_ns));
    record["finished_ns"] = clock.now();
    if (record_structure_) {
        record["after_result"] = capacity(index);
    }
    if (action.timing.next_read) {

        // Other native callbacks can run before the scheduler reads its input queue.
        auto input = clock;
        input.satisfy(record.at("began_ns"), record.at("model_ready_ns"), *action.timing.next_read);
        actor.read = PendingRead{input.now(), actions_.size()};
    }
    actions_.push_back(std::move(record));
    actor.action.reset();
}

json Simulation::direct(Index actor, const json& message) {
    const auto& operation = message.at("operation");
    if (operation == "execute") {
        execute(actor, message);
    } else if (operation == "publish") {
        publish(actor, message);
    } else if (operation == "result_finished") {
        finish_action(actor);
    } else {
        return resources(actor, message);
    }
    return nullptr;
}

json Simulation::request(const json& message) {
    const auto operation = message.at("operation").get<std::string>();
    if (operation == "initialize") {
        json setup{{"domain", job_.at("domain")}, {"model_config", job_.at("model_config")},
                   {"workload", job_.at("workload")}};
        if (job_.contains("expert_routes")) {
            setup["expert_routes"] = job_.at("expert_routes");
        }
        return setup;
    }
    if (group_) {
        return group_->request(message);
    }
    if (operation == "arrivals") {
        return arrivals(0, done());
    }
    if (operation == "idle") {
        if (const auto arrival = next_arrival(0)) {
            actors_.front().clock.advance_to(*arrival);
        } else if (!done()) {
            throw Stopped("controller_stall", "native control is idle with incomplete requests and no arrivals");
        }
        return nullptr;
    }
    return direct(0, message);
}

json Simulation::result(std::string status, std::string reason) {
    json report{{"status", std::move(status)}, {"reason", std::move(reason)}, {"now_ns", now()},
                {"finished_requests", finished_}, {"actions", std::move(actions_)}};
    if (prefix_cache_ && record_structure_) {
        report["cache_effects"] = std::move(cache_effects_);
    }
    if (group_) {
        group_->report(report);
    }
    return report;
}

}
