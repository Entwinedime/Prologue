#pragma once

#include "prologue/core/performance/costs.hpp"
#include "prologue/core/serving/simulation.hpp"

namespace prologue::serving {

class CostTiming final : public TimingSource {
public:
    CostTiming(const nlohmann::json& domain, const performance::CostCatalog& costs);

    ActionTiming for_action(const nlohmann::json& action, const nlohmann::json& graph) override;

private:
    const nlohmann::json& domain_;
    const performance::CostCatalog& costs_;
};

}
