#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace prologue::serving {

using Nanoseconds = std::int64_t;

struct TimedWork final {
    std::int64_t lane;
    Nanoseconds submitted_after_ns;
    Nanoseconds dispatch_ns;
    Nanoseconds duration_ns;
    std::vector<std::size_t> after{};
};

struct BoundaryTime final {
    Nanoseconds after_start_ns;
    Nanoseconds after_model_ns;
};

struct Interval final {
    Nanoseconds begin_ns;
    Nanoseconds end_ns;
};

struct ActionCompletion final {
    std::vector<Interval> intervals;
    Nanoseconds all_work_ready_ns;
    Nanoseconds output_lane_ready_ns;
};

class Timeline final {
public:
    Nanoseconds now() const;
    void advance_to(Nanoseconds instant);
    void advance_by(Nanoseconds elapsed);
    void satisfy(Nanoseconds started, Nanoseconds model_ready, BoundaryTime boundary);

    ActionCompletion schedule_work(const std::vector<TimedWork>& work, bool record_intervals,
                                   std::optional<std::int64_t> publication_lane = {}) const;

private:
    Nanoseconds now_ = 0;
};

}
