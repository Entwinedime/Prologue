#pragma once

#include <nlohmann/json_fwd.hpp>

namespace prologue::structure {

class Compiler final {
public:
    nlohmann::json compile(const nlohmann::json& observation) const;
};

}
