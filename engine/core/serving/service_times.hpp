#pragma once

#include <chrono>
#include <cstddef>
#include <map>
#include <optional>
#include <span>
#include <string>

#include <nlohmann/json.hpp>

#include "core/serving/execution/timing.hpp"
#include "core/performance/work_costs.hpp"
#include "core/serving/execution_inputs.hpp"

namespace prologue {

class ServiceTimes {
public:
    ServiceTimes(ServiceDuration given, ExecutionInputs * conditions, const nlohmann::json & assets, ExecutionObserver * observer);

    void bind(const std::string & group, nlohmann::json domain, std::span<const TemplateDefinition> definitions);
    bool modeled() const { return !assets_.is_null(); }
    ExecutionInputs * conditions() const { return conditions_; }
    std::optional<InvocationInput> action_input(const ServingAction & action);
    GraphTiming action(const ServingAction & action, const CoarseGraph & graph, std::span<const TemplateInterval> origins,
                       const std::optional<InvocationInput> & observed);
    ServiceDuration external(const std::string & group, const nlohmann::json & identity, WorkKind kind, const WorkFeatures & features);
    nlohmann::json finish() const;

private:
    ServiceDuration given_;
    ExecutionInputs * conditions_;
    const nlohmann::json & assets_;
    ExecutionObserver * observer_;
    std::map<std::string, WorkCosts> participants_;
    std::size_t queries_ = 0;
    std::chrono::nanoseconds query_time_{};
};

} // namespace prologue
