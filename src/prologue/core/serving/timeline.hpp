#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
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

struct ActionTiming final {
    std::vector<TimedWork> work;
    Nanoseconds prepare_ns;
    Nanoseconds model_return_ns;
    BoundaryTime result_finished;
    std::unordered_map<std::string, BoundaryTime> publications;
    std::optional<BoundaryTime> next_read{};
    bool await_output_lane = false;
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

struct RunningAction final {
    Nanoseconds began_ns;
    Nanoseconds model_ready_ns;
    ActionCompletion work;
};

class Timeline final {
public:
    Nanoseconds now() const;
    void advance_to(Nanoseconds instant);
    void advance_by(Nanoseconds elapsed);
    void satisfy(Nanoseconds started, Nanoseconds model_ready, BoundaryTime boundary);

    RunningAction begin_action(const ActionTiming& timing, bool record_intervals,
                               std::optional<std::int64_t> publication_lane = {});
    ActionCompletion schedule_work(const std::vector<TimedWork>& work, bool record_intervals,
                                   std::optional<std::int64_t> publication_lane = {}) const;

private:
    Nanoseconds now_ = 0;
};

}
