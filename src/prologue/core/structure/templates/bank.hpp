#pragma once

#include <nlohmann/json.hpp>

namespace prologue::structure {

nlohmann::json assemble_bank(const nlohmann::json& input);

}
