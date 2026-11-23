#pragma once

#include "prologue/core/serving/simulation.hpp"

#include <nlohmann/json_fwd.hpp>

namespace prologue::app {

serving::ActionTiming read_timing(const nlohmann::json& record, const nlohmann::json& work);
nlohmann::json replay(const nlohmann::json& job);

}
