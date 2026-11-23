#pragma once

#include <cstdint>
#include <vector>

namespace prologue::serving {

using Nanoseconds = std::int64_t;

struct TimedWork final {
    std::int64_t lane;
    Nanoseconds submitted_after_ns;
    Nanoseconds duration_ns;
};

struct Interval final {
    Nanoseconds begin_ns;
    Nanoseconds end_ns;
};

class Timeline final {
public:
    Nanoseconds now() const;
    void advance_to(Nanoseconds instant);
    void advance_by(Nanoseconds elapsed);

    // A synchronous action returns after its host boundary and every submitted device interval.
    std::vector<Interval> complete_action(const std::vector<TimedWork>& work, Nanoseconds host_boundary_ns);

private:
    Nanoseconds now_ = 0;
};

}
