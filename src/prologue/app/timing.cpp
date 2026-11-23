#include "prologue/app/timing.hpp"

#include "prologue/core/performance/costs.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace prologue::app {
namespace {

using nlohmann::json;
using serving::BoundaryTime;
using serving::Nanoseconds;
using serving::Stopped;

Nanoseconds elapsed(const json& row, const char* key) {
    const auto it = row.find(key);
    if (it == row.end() || !it->is_number_integer() || *it < 0 ||
        (it->is_number_unsigned() && it->get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<Nanoseconds>::max()))) {
        throw Stopped("timing_gap", std::string("missing or invalid measured ") + key);
    }
    return it->get<Nanoseconds>();
}

BoundaryTime boundary(const json& row) {
    return {elapsed(row, "after_start_ns"), elapsed(row, "after_model_ns")};
}

}

serving::ActionTiming read_timing(const json& record, const json& work) {
    const auto& samples = record.at("work");
    if (!samples.is_array() || samples.size() != work.size()) {
        throw Stopped("timing_gap", "one measured interval is required per selected work");
    }

    serving::ActionTiming timing{{}, elapsed(record, "prepare_ns"), elapsed(record, "model_return_ns"),
                                  boundary(record.at("result_finished")), {}};
    for (std::size_t index = 0; index < work.size(); ++index) {
        const auto& sample = samples.at(index);
        timing.work.push_back({work.at(index).at("lane"), elapsed(sample, "submitted_after_ns"),
                               elapsed(sample, "dispatch_ns"), elapsed(sample, "duration_ns")});
    }
    for (const auto& entry : record.at("publications").items()) {
        timing.publications.emplace(entry.key(), boundary(entry.value()));
    }
    return timing;
}

json replay(const json& job) {
    const auto& domain = job.at("domain");
    const auto restore = job.at("restore_kind");
    std::optional<performance::CostCatalog> costs;
    if (job.contains("costs")) {
        costs.emplace(job.at("costs"), job.at("rules").at("performance"));
    }

    serving::Timeline timeline;
    auto actions = json::array();
    for (const auto& row : job.at("batches")) {
        const auto& action = row.at("action");
        const auto& graph = row.at("graph");
        auto timing = read_timing(row.at("timing"), graph.at("work"));
        if (costs) {
            const performance::CostQueries queries(action, graph);
            for (std::size_t index = 0; index < queries.device.size(); ++index) {
                const auto& coordinate = queries.device.at(index);
                if (coordinate.at("kind") != restore) {
                    const auto& estimate = costs->require(domain, coordinate);
                    timing.work.at(index).duration_ns = estimate.at("duration_ns").get<Nanoseconds>();
                }
            }
        }

        timeline.advance_to(std::max(timeline.now(), row.at("input_ready_ns").get<Nanoseconds>()));
        timeline.advance_by(timing.prepare_ns);
        const auto started = timeline.now();
        const auto intervals = timeline.complete_action(timing.work, timing.model_return_ns, !costs);
        const auto device_ready = timeline.now();
        timeline.advance_by(row.at("model_tail_ns"));
        const auto model_ready = timeline.now();

        auto publications = json::array();
        for (const auto& member : action.at("members")) {
            const auto request = member.at("request").get<std::string>();
            timeline.satisfy(started, model_ready, timing.publications.at(request));
            publications.push_back({{"request", request}, {"count", member.at("output_tokens").get<std::int64_t>() + 1},
                                    {"at_ns", timeline.now()}});
        }
        timeline.satisfy(started, model_ready, timing.result_finished);
        auto result = json{{"began_ns", started}, {"device_ready_ns", device_ready},
                           {"publications", std::move(publications)}, {"finished_ns", timeline.now()}};
        if (!costs) {
            auto observed = json::array();
            for (const auto& interval : intervals) {
                observed.push_back({interval.begin_ns, interval.end_ns});
            }
            result["intervals"] = std::move(observed);
        }
        actions.push_back(std::move(result));
    }
    return {{"status", "complete"}, {"actions", std::move(actions)}};
}

}
