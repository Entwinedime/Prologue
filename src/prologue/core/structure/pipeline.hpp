#pragma once

#include <nlohmann/json_fwd.hpp>

namespace prologue::structure {

void bind_pipeline(nlohmann::json& graph);

}
