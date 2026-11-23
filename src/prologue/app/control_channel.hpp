#pragma once

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace prologue::serving {
class Simulation;
}

namespace prologue::app {

nlohmann::json serve_control(const std::string& socket_path, serving::Simulation& simulation);

}
