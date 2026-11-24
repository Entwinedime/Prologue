#pragma once

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <optional>

namespace prologue::structure {

void bind_collectives(nlohmann::json& graph);
std::optional<std::int64_t> publication_lane(const nlohmann::json& graph);

}
