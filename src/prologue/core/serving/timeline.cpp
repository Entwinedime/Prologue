#include "prologue/core/serving/timeline.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace prologue::serving {
namespace {

Nanoseconds after(Nanoseconds begin, Nanoseconds elapsed) {
    if (elapsed < 0 || elapsed > std::numeric_limits<Nanoseconds>::max() - begin) {
        throw std::invalid_argument("invalid elapsed time");
    }
    return begin + elapsed;
}

}

Nanoseconds Timeline::now() const {
    return now_;
}

void Timeline::advance_to(Nanoseconds instant) {
    if (instant < now_) {
        throw std::invalid_argument("virtual time cannot move backwards");
    }
    now_ = instant;
}

void Timeline::advance_by(Nanoseconds elapsed) {
    now_ = after(now_, elapsed);
}

void Timeline::satisfy(Nanoseconds started, Nanoseconds model_ready, BoundaryTime boundary) {
    now_ = std::max({now_, after(started, boundary.after_start_ns), after(model_ready, boundary.after_model_ns)});
}

RunningAction Timeline::begin_action(const ActionTiming& timing, bool record_intervals,
                                    std::optional<std::int64_t> publication_lane) {
    advance_by(timing.prepare_ns);
    const auto started = now_;
    auto completed = schedule_work(timing.work, record_intervals, publication_lane);
    advance_by(timing.model_return_ns);

    // An observed host return can precede the device completion used by the cost model.
    if (timing.await_output_lane) {
        advance_to(std::max(now_, completed.output_lane_ready_ns));
    }
    return {started, now_, std::move(completed)};
}

ActionCompletion Timeline::schedule_work(const std::vector<TimedWork>& work, bool record_intervals,
                                         std::optional<std::int64_t> publication_lane) const {
    std::unordered_map<std::int64_t, Nanoseconds> lanes;
    std::vector<Nanoseconds> completed;
    completed.reserve(work.size());
    std::vector<Interval> intervals;
    auto output_ready = now_;
    auto all_work_ready = now_;

    for (const auto& entry : work) {
        auto& available = lanes[entry.lane];
        auto ready = std::max(available, after(now_, entry.submitted_after_ns));
        for (const auto predecessor : entry.after) {
            ready = std::max(ready, completed.at(predecessor));
        }
        const auto begin = after(ready, entry.dispatch_ns);
        const auto end = after(begin, entry.duration_ns);
        if (record_intervals) {
            intervals.push_back({begin, end});
        }
        completed.push_back(end);
        available = end;
        all_work_ready = std::max(all_work_ready, end);
        if (!publication_lane || entry.lane == *publication_lane) {
            output_ready = std::max(output_ready, end);
        }
    }
    return {std::move(intervals), all_work_ready, output_ready};
}

}
