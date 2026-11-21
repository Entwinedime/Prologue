#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "prologue/serving/execution/timing.hpp"
#include "prologue/graph/types.hpp"

namespace prologue {

class CoarseGraph;

// A consumer may record actual execution without retaining historical events in the live timeline.
class ExecutionObserver {
public:
    virtual ~ExecutionObserver() = default;
    virtual void inserted(const CoarseGraph & graph, std::span<const std::size_t> events, const GraphTiming & timing, std::int64_t time_ns) = 0;
    virtual void rejected_layout(const CoarseGraph & graph, std::string_view group, std::size_t step) = 0;
    virtual void connected(std::size_t source, std::size_t target, Relation relation, std::int64_t delay_ns,
                           std::optional<std::size_t> activation) = 0;
    virtual void activated(std::size_t event, std::int64_t time_ns) = 0;
    virtual void completed(std::size_t event, std::int64_t time_ns) = 0;
    virtual void conditioned(const BoundaryCondition & condition) = 0;
};

} // namespace prologue
