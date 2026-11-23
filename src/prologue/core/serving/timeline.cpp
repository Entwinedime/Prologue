#include "prologue/core/serving/timeline.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>

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

std::vector<Interval> Timeline::complete_action(const std::vector<TimedWork>& work, Nanoseconds host_boundary_ns) {
    std::unordered_map<std::int64_t, Nanoseconds> lanes;
    std::vector<Interval> intervals;
    auto completion = after(now_, host_boundary_ns);

    for (const auto& entry : work) {
        auto& available = lanes[entry.lane];
        const auto begin = std::max(available, after(now_, entry.submitted_after_ns));
        const auto end = after(begin, entry.duration_ns);
        intervals.push_back({begin, end});
        available = end;
        completion = std::max(completion, end);
    }
    now_ = completion;
    return intervals;
}

}
