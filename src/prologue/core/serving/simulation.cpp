#include "prologue/core/serving/simulation.hpp"
#include "prologue/core/serving/pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace prologue::serving {
namespace {

using nlohmann::json;

ResourceLimits limits(const json& job) {
    const auto& config = job.at("domain").at("configuration");
    return {config.at("max_running_requests"), config.at("max_total_tokens"), config.at("page_size")};
}

}

Stopped::Stopped(std::string status, const std::string& reason) : std::runtime_error(reason), status_(std::move(status)) {}

const std::string& Stopped::status() const {
    return status_;
}

Simulation::Simulation(const json& job, Workload workload, TimingSource& timing_source)
    : job_(job), workload_(std::move(workload)),
      templates_(job.at("templates"), job.at("domain"), job.at("rules").at("templates")),
      timing_source_(timing_source), record_structure_(job.value("record_structure", false)) {
    for (const auto& request : workload_.requests()) {
        output_limits_.emplace(request.id, request.output_tokens);
        output_counts_.emplace(request.id, 0);
    }

    const auto width = job.at("domain").at("configuration").value("pp", std::size_t{1});
    pools_.reserve(width);
    for (std::size_t stage = 0; stage < width; ++stage) {
        pools_.emplace_back(workload_, limits(job));
    }
    if (width > 1) {
        pipeline_ = std::make_unique<Pipeline>(*this, width);
    }
}

Simulation::~Simulation() = default;

Nanoseconds Simulation::now() const {
    return pipeline_ ? pipeline_->now() : timeline_.now();
}

bool Simulation::done() const {
    return pipeline_ ? pipeline_->done() : finished_ == workload_.requests().size() && !timing_;
}

json Simulation::capacity(std::size_t actor) const {
    const auto& pool = pools_.at(actor);
    return {{"slots", pool.available_slots()}, {"kv_tokens", pool.available_tokens()}};
}

json Simulation::arrivals(Nanoseconds time_ns, bool finished) {
    auto arrivals = json::array();
    const auto& requests = workload_.requests();
    while (next_arrival_ < requests.size() && requests.at(next_arrival_).arrival_ns <= time_ns) {
        const auto& request = requests.at(next_arrival_++);
        arrivals.push_back({{"id", request.id}, {"prompt_tokens", request.prompt_tokens},
                            {"output_tokens", request.output_tokens}});
    }
    if (finished && !std::ranges::all_of(pools_, &Resources::drained)) {
        throw Stopped("resource_error", "terminal resources are not fully returned");
    }
    return {{"requests", std::move(arrivals)}, {"done", finished}};
}

json Simulation::resources(std::size_t actor, const json& message) {
    const auto operation = message.at("operation").get<std::string>();
    auto& pool = pools_.at(actor);
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
                members.push_back({member.at("request"), member.at("begin"), member.at("end")});
            }
            return pool.grow(members);
        }
        if (operation == "release_slot") {
            pool.release_slot(message.at("request"));
            return nullptr;
        }
        if (operation == "release_pages") {
            pool.release_pages(message.at("pages"));
            return nullptr;
        }
    } catch (const std::exception& error) {
        throw Stopped("resource_error", error.what());
    }
    throw Stopped("controller_error", "unknown operation " + operation);
}

void Simulation::require_residence(std::size_t actor, const json& message) const {
    for (const auto& member : message.at("members")) {
        const auto request = member.at("request").get<std::string>();
        const auto& residence = pools_.at(actor).residence(request);
        if (!residence.slot || *residence.slot != member.at("slot") || residence.tokens != member.at("kv_committed_tokens") ||
            residence.tokens != member.at("kv_reserved_tokens")) {
            throw Stopped("resource_error", "native batch does not agree with its resource binding for " + request);
        }
    }
}

void Simulation::execute(const json& message) {
    if (timing_) {
        throw Stopped("controller_error", "native action began before the preceding result finished");
    }
    require_residence(0, message);

    const auto query = structure::batch_geometry(message.at("phase"), message.at("members"),
                                                 message.value("expert_routes", json{}));
    auto graph = templates_.expand(query);

    // Only a fully selected action can query time. The provider cannot choose the graph.
    timing_ = timing_source_.for_action(message, graph);
    timeline_.advance_by(timing_->prepare_ns);
    const auto started = now();
    std::optional<std::int64_t> publisher;
    if (graph.contains("collectives")) {
        publisher = graph.at("calls").front().at("parameters").at("publication_rank").get<std::int64_t>();
    }
    const auto completed = timeline_.complete_action(timing_->work, timing_->model_return_ns, record_structure_, publisher);
    work_ready_ns_ = completed.all_work_ready_ns;
    current_ = {
        {"phase", message.at("phase")}, {"members", message.at("members")}, {"began_ns", started},
        {"model_ready_ns", now()}, {"publications", json::array()},
    };
    if (message.contains("expert_routes")) {
        current_["expert_routes"] = message.at("expert_routes");
    }
    if (record_structure_) {
        auto observed = json::array();
        for (const auto& interval : completed.intervals) {
            observed.push_back({interval.begin_ns, interval.end_ns});
        }
        current_["resources"] = capacity(0);
        current_["graph"] = std::move(graph);
        current_["intervals"] = std::move(observed);
    }
}

void Simulation::wait_for(BoundaryTime boundary) {
    timeline_.satisfy(current_.at("began_ns"), current_.at("model_ready_ns"), boundary);
}

void Simulation::publish(const json& message) {
    if (!timing_) {
        throw Stopped("controller_error", "publication has no executing action");
    }
    const auto request = message.at("request").get<std::string>();
    const auto offset = timing_->publications.find(request);
    if (offset == timing_->publications.end()) {
        throw Stopped("timing_gap", "missing publication readiness for " + request);
    }

    wait_for(offset->second);
    record_publication(message, now(), current_.at("publications"));
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

void Simulation::finish_action() {
    if (!timing_) {
        throw Stopped("controller_error", "native result has no executing action");
    }
    wait_for(timing_->result_finished);

    // Other ranks may still be finishing when the publication owner emits a token.
    timeline_.advance_to(std::max(now(), work_ready_ns_));
    current_["finished_ns"] = now();
    if (record_structure_) {
        current_["after_result"] = capacity(0);
    }
    actions_.push_back(std::move(current_));
    timing_.reset();
}

json Simulation::request(const json& message) {
    const auto operation = message.at("operation").get<std::string>();
    if (operation == "initialize") {
        json setup{{"domain", job_.at("domain")}, {"model_config", job_.at("model_config")}};
        if (job_.contains("expert_routes")) {
            setup["expert_routes"] = job_.at("expert_routes");
        }
        return setup;
    }
    if (pipeline_) {
        return pipeline_->request(message);
    }
    if (operation == "arrivals") {
        return arrivals(now(), done());
    }
    if (operation == "idle") {
        if (next_arrival_ < workload_.requests().size()) {
            timeline_.advance_to(workload_.requests().at(next_arrival_).arrival_ns);
        } else if (!done()) {
            throw Stopped("controller_stall", "native control is idle with incomplete requests and no arrivals");
        }
        return nullptr;
    }
    if (operation == "execute") {
        execute(message);
        return nullptr;
    }
    if (operation == "publish") {
        publish(message);
        return nullptr;
    }
    if (operation == "result_finished") {
        finish_action();
        return nullptr;
    }
    return resources(0, message);
}

json Simulation::result(std::string status, std::string reason) {
    json report{{"status", std::move(status)}, {"reason", std::move(reason)}, {"now_ns", now()},
                {"finished_requests", finished_}, {"actions", std::move(actions_)}};
    if (pipeline_) {
        report["peak_inflight"] = pipeline_->peak_inflight();
    }
    return report;
}

}
