#include "prologue/core/serving/simulation.hpp"

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
    : job_(job), workload_(std::move(workload)), resources_(workload_, limits(job)), templates_(job.at("libraries")),
      timing_source_(timing_source) {
    for (const auto& request : workload_.requests()) {
        output_limits_.emplace(request.id, request.output_tokens);
        output_counts_.emplace(request.id, 0);
    }
}

Nanoseconds Simulation::now() const {
    return timeline_.now();
}

bool Simulation::done() const {
    return finished_ == workload_.requests().size() && !timing_;
}

json Simulation::capacity() const {
    return {{"slots", resources_.available_slots()}, {"kv_tokens", resources_.available_tokens()}};
}

json Simulation::arrivals() {
    auto arrivals = json::array();
    const auto& requests = workload_.requests();
    while (next_arrival_ < requests.size() && requests.at(next_arrival_).arrival_ns <= now()) {
        const auto& request = requests.at(next_arrival_++);
        arrivals.push_back({{"id", request.id}, {"prompt_tokens", request.prompt_tokens},
                            {"output_tokens", request.output_tokens}});
    }
    if (done() && !resources_.drained()) {
        throw Stopped("resource_error", "terminal resources are not fully returned");
    }
    return {{"requests", std::move(arrivals)}, {"done", done()}};
}

json Simulation::resources(const std::string& operation, const json& message) {
    try {
        if (operation == "capacity") {
            return capacity();
        }
        if (operation == "free_slots") {
            return resources_.free_slots();
        }
        if (operation == "bind_slots") {
            return resources_.bind_slots(message.at("requests"));
        }
        if (operation == "grow") {
            std::vector<KvGrowth> members;
            for (const auto& member : message.at("members")) {
                members.push_back({member.at("request"), member.at("begin"), member.at("end")});
            }
            return resources_.grow(members);
        }
        if (operation == "release_slot") {
            resources_.release_slot(message.at("request"));
            return nullptr;
        }
        if (operation == "release_pages") {
            resources_.release_pages(message.at("pages"));
            return nullptr;
        }
    } catch (const std::exception& error) {
        throw Stopped("resource_error", error.what());
    }
    throw Stopped("controller_error", "unknown operation " + operation);
}

void Simulation::execute(const json& message) {
    if (timing_) {
        throw Stopped("controller_error", "native action began before the preceding result finished");
    }
    for (const auto& member : message.at("members")) {
        const auto request = member.at("request").get<std::string>();
        const auto& residence = resources_.residence(request);
        if (!residence.slot || *residence.slot != member.at("slot") || residence.tokens != member.at("kv_committed_tokens") ||
            residence.tokens != member.at("kv_reserved_tokens")) {
            throw Stopped("resource_error", "native batch does not agree with its resource binding for " + request);
        }
    }

    auto graph = templates_.expand(message);
    if (!graph) {
        throw Stopped("template_gap", "no source template for native action " + message.dump());
    }

    // Only a fully selected action can query time. The provider cannot choose the graph.
    timing_ = timing_source_.for_action(message, graph->at("work"));
    const auto started = now();
    const auto intervals = timeline_.complete_action(timing_->work, timing_->model_return_ns);
    auto intervals_json = json::array();
    for (const auto& interval : intervals) {
        intervals_json.push_back({interval.begin_ns, interval.end_ns});
    }
    current_ = {
        {"phase", message.at("phase")}, {"members", message.at("members")}, {"resources", capacity()},
        {"began_ns", started}, {"model_ready_ns", now()}, {"graph", std::move(*graph)},
        {"intervals", std::move(intervals_json)}, {"publications", json::array()},
    };
}

void Simulation::wait_for_action_offset(Nanoseconds offset) {
    const auto elapsed = now() - current_.at("began_ns").get<Nanoseconds>();
    if (offset > elapsed) {
        timeline_.advance_by(offset - elapsed);
    }
}

void Simulation::publish(const json& message) {
    if (!timing_) {
        throw Stopped("controller_error", "publication has no executing action");
    }
    const auto request = message.at("request").get<std::string>();
    const auto count = message.at("count").get<std::int64_t>();
    auto& previous = output_counts_.at(request);
    if (count <= previous || count > output_limits_.at(request) ||
        message.at("finished").get<bool>() != (count == output_limits_.at(request))) {
        throw Stopped("action_mismatch", "native publication is outside the expected token extent");
    }
    const auto offset = timing_->publications.find(request);
    if (offset == timing_->publications.end()) {
        throw Stopped("timing_gap", "missing publication readiness for " + request);
    }

    wait_for_action_offset(offset->second);
    current_.at("publications").push_back({{"request", request}, {"count", count}, {"at_ns", now()}});
    previous = count;
    if (message.at("finished").get<bool>()) {
        ++finished_;
    }
}

void Simulation::finish_action() {
    if (!timing_) {
        throw Stopped("controller_error", "native result has no executing action");
    }
    wait_for_action_offset(timing_->result_finished_ns);
    current_["finished_ns"] = now();
    current_["after_result"] = capacity();
    actions_.push_back(std::move(current_));
    timing_.reset();
}

json Simulation::request(const json& message) {
    const auto operation = message.at("operation").get<std::string>();
    if (operation == "initialize") {
        return {{"domain", job_.at("domain")}, {"model_config", job_.at("model_config")}};
    }
    if (operation == "arrivals") {
        return arrivals();
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
    return resources(operation, message);
}

json Simulation::result(std::string status, std::string reason) {
    return {{"status", std::move(status)}, {"reason", std::move(reason)}, {"now_ns", now()},
            {"finished_requests", finished_}, {"actions", std::move(actions_)}};
}

}
