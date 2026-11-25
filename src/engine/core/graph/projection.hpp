#pragma once

#include <optional>

#include <nlohmann/json.hpp>

#include "core/graph/graph.hpp"
#include "core/graph/observation.hpp"

namespace prologue {

struct ProjectedGraph {
    CoarseGraph graph;
    std::optional<nlohmann::json> evidence;
};

// Source and independent target runs use this projection, without evaluating template rules.
class CoarseProjection {
public:

    static ProjectedGraph project(NativeObservation observation, bool include_evidence);
};

} // namespace prologue
