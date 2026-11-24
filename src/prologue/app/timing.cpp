#include "prologue/app/timing.hpp"

#include "prologue/core/performance/costs.hpp"
#include "prologue/core/serving/cost_timing.hpp"
#include "prologue/core/structure/parallel.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace prologue::app {
namespace {

using nlohmann::json;
using serving::ActionTiming;
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

void estimate_device_service(ActionTiming& timing, const json& action, const json& graph,
                             const json& domain, const performance::CostCatalog& costs) {
    const performance::CostQueries queries(action, graph);
    std::vector<const json*> coordinates;
    coordinates.reserve(queries.device.size());
    for (const auto& coordinate : queries.device) {
        coordinates.push_back(&coordinate);
    }
    const auto estimates = costs.require(domain, coordinates);
    for (std::size_t index = 0; index < estimates.size(); ++index) {
        timing.work.at(index).duration_ns = estimates.at(index).at("duration_ns").get<Nanoseconds>();
    }
}

void restore_device_kind(ActionTiming& timing, const ActionTiming& measured, const json& work, const json& kind) {
    for (std::size_t index = 0; index < work.size(); ++index) {
        if (work.at(index).at("kind") == kind) {
            timing.work.at(index).dispatch_ns = measured.work.at(index).dispatch_ns;
            timing.work.at(index).duration_ns = measured.work.at(index).duration_ns;
        }
    }
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
        const auto& entry = work.at(index);
        timing.work.push_back({entry.at("lane"), elapsed(sample, "submitted_after_ns"),
                               elapsed(sample, "dispatch_ns"), elapsed(sample, "duration_ns"),
                               entry.value("after", std::vector<std::size_t>{})});
    }
    for (const auto& entry : record.at("publications").items()) {
        timing.publications.emplace(entry.key(), boundary(entry.value()));
    }
    if (const auto next = record.find("next_read"); next != record.end()) {
        timing.next_read = boundary(*next);
    }
    return timing;
}

serving::ProgramTiming read_program_timing(const json& record, const json& graph) {
    auto interval = [](const json& item) {
        const auto begin = elapsed(item, "begin_ns");
        const auto end = elapsed(item, "end_ns");
        if (end < begin) {
            throw Stopped("timing_gap", "a measured program interval ends before it begins");
        }
        return serving::Interval{begin, end};
    };
    const auto& work = record.at("work");
    const auto& steps = record.at("steps");
    if (work.size() != graph.at("work").size() || steps.size() != graph.at("steps").size()) {
        throw Stopped("timing_gap", "measured program timing must cover every selected cut and work item");
    }
    serving::ProgramTiming timing;
    for (const auto& item : work) {
        timing.work.push_back(interval(item));
    }
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const auto& ranks = steps.at(index);
        if (ranks.size() != graph.at("steps").at(index).at("ranks").size()) {
            throw Stopped("timing_gap", "a pipeline cut is missing measured rank timing");
        }
        auto& samples = timing.steps.emplace_back();
        for (const auto& rank : ranks) {
            samples.push_back({interval(rank), elapsed(rank, "ready_ns")});
        }
    }
    for (const auto& item : record.at("publications").items()) {
        timing.publications.emplace(item.key(), elapsed(record.at("publications"), item.key().c_str()));
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

    std::optional<serving::CostTiming> modeled;
    if (costs && !job.value("device_only", false)) {
        modeled.emplace(domain, *costs);
    }
    serving::Timeline timeline;
    auto actions = json::array();
    for (const auto& row : job.at("batches")) {
        const auto& action = row.at("action");
        const auto& graph = row.at("graph");
        const auto& work = graph.at("work");
        auto timing = modeled ? modeled->for_action(action, graph) : read_timing(row.at("timing"), work);
        if (modeled) {
            if (!restore.is_null()) {
                restore_device_kind(timing, read_timing(row.at("timing"), work), work, restore);
            }
        } else {

            // Preserve the observed host schedule to isolate device service in this replay.
            timing.await_output_lane = true;
            if (costs) {
                estimate_device_service(timing, action, graph, domain, *costs);
            }
        }

        timeline.advance_to(std::max(timeline.now(), row.at("input_ready_ns").get<Nanoseconds>()));
        const auto publisher = structure::publication_lane(graph);
        const auto execution = timeline.begin_action(timing, !costs, publisher);
        const auto started = execution.began_ns;
        if (!modeled) {
            timeline.advance_by(row.at("model_tail_ns"));
        }
        const auto model_ready = timeline.now();

        auto publications = json::array();
        for (const auto& member : action.at("members")) {
            const auto request = member.at("request").get<std::string>();
            if (!timing.publications.contains(request)) {
                continue;
            }
            timeline.satisfy(started, model_ready, timing.publications.at(request));
            publications.push_back({{"request", request}, {"count", member.at("output_tokens").get<std::int64_t>() + 1},
                                    {"at_ns", timeline.now()}});
        }
        timeline.satisfy(started, model_ready, timing.result_finished);
        timeline.advance_to(std::max(timeline.now(), execution.work.all_work_ready_ns));
        auto result = json{{"began_ns", started}, {"device_ready_ns", execution.work.all_work_ready_ns},
                           {"model_ready_ns", model_ready},
                           {"publications", std::move(publications)}, {"finished_ns", timeline.now()}};
        if (!costs) {
            auto observed = json::array();
            for (const auto& interval : execution.work.intervals) {
                observed.push_back({interval.begin_ns, interval.end_ns});
            }
            result["intervals"] = std::move(observed);
        }
        actions.push_back(std::move(result));
    }
    return {{"status", "complete"}, {"actions", std::move(actions)}};
}

}
