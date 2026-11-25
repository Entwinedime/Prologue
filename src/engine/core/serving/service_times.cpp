#include "core/serving/service_times.hpp"

#include <utility>
#include <vector>

#include "core/graph/graph.hpp"
#include "core/templates/library.hpp"

namespace prologue {

ServiceTimes::ServiceTimes(ServiceDuration given, ExecutionInputs * conditions, const nlohmann::json & assets, ExecutionObserver * observer)
    : given_(given), conditions_(conditions), assets_(assets), observer_(observer) {}

void ServiceTimes::bind(const std::string & group, nlohmann::json domain, std::span<const TemplateDefinition> definitions) {
    participants_.try_emplace(group, std::move(domain), definitions, assets_);
}

std::optional<InvocationInput> ServiceTimes::action_input(const ServingAction & action) {
    if (!conditions_) return std::nullopt;
    return conditions_->invocation(action.task.group, "action", ExecutionInputs::action_identity(action));
}

GraphTiming ServiceTimes::action(const ServingAction & action, const CoarseGraph & graph, std::span<const TemplateInterval> origins,
                                 const std::optional<InvocationInput> & observed) {
    if (observed) return observed->timing(graph, observer_);
    if (given_) return { std::vector<ServiceDuration>(graph.work.size(), given_), {} };

    const auto began = std::chrono::steady_clock::now();
    GraphTiming result;
    try {
        result = participants_.at(action.task.group).timing(graph, origins, nullptr, observer_ != nullptr);
    } catch (const CostGap & gap) {
        auto details = gap.details();

        // Keep the measured work identity separate from the live invocation an adapter can use to reproduce it.
        auto & call = details["invocation"] = {{ "action", action.name }, { "condition", action.condition }, { "parameters", action.parameters }};
        for (const auto key : { "members", "context", "callers" }) {
            if (action.work.contains(key)) call[key] = action.work.at(key);
        }
        throw CostGap(std::move(details));
    }
    query_time_ += std::chrono::steady_clock::now() - began;
    queries_ += graph.work.size();
    return result;
}

ServiceDuration ServiceTimes::external(const std::string & group, const nlohmann::json & identity, WorkKind kind, const WorkFeatures & features) {
    if (given_) return given_;

    const auto began = std::chrono::steady_clock::now();
    const auto duration = participants_.at(group).external(identity, kind, features);
    query_time_ += std::chrono::steady_clock::now() - began;
    ++queries_;
    return duration;
}

nlohmann::json ServiceTimes::finish() const {
    if (conditions_) return conditions_->finish();
    if (given_) return {{ "duration_basis", "uniform_given_work_ns" }, { "given_work_ns", *given_ }};
    return {{ "duration_basis", "independent_service_costs" }, { "cost_queries", queries_ }, { "cost_query_ns", query_time_.count() }};
}

} // namespace prologue
