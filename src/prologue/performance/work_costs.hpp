#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "prologue/performance/service_model.hpp"
#include "prologue/serving/execution/timing.hpp"

namespace prologue {

class CoarseGraph;
class WorkFeatures;
enum class WorkKind;
struct TemplateDefinition;
struct TemplateInterval;

class CostGap final : public std::runtime_error {
public:
    explicit CostGap(nlohmann::json details);
    const nlohmann::json & details() const { return details_; }

private:
    nlohmann::json details_;
};

class WorkCosts {
public:
    WorkCosts(nlohmann::json domain, std::span<const TemplateDefinition> definitions, const nlohmann::json & asset);
    GraphTiming timing(const CoarseGraph & graph, std::span<const TemplateInterval> origins, nlohmann::json * first_gap = nullptr,
                       bool record_bindings = false) const;
    std::int64_t external(const nlohmann::json & identity, WorkKind kind, const WorkFeatures & features) const;
    static nlohmann::json evaluate(const nlohmann::json & groups, const nlohmann::json & asset, bool bindings);

private:
    struct Estimate {
        const ServiceModel * model;
        ServiceDuration service_ns;

        std::string_view status() const;
    };

    using PhaseModels = std::map<std::string, std::unique_ptr<const ServiceModel>, std::less<>>;

    template<class Definitions>
    WorkCosts(nlohmann::json domain, const nlohmann::json & asset, const Definitions & definitions);
    std::optional<std::size_t> external_index(const nlohmann::json & identity, WorkKind kind) const;

    template<class Features>
    Estimate estimate(std::optional<std::size_t> definition, std::size_t interval, std::string_view phase, const Features & features) const;
    template<class Features>
    Estimate work_estimate(std::optional<std::size_t> definition, std::size_t interval, WorkKind kind, const Features & features,
                           const nlohmann::json * explicit_service = nullptr) const;
    CostGap gap(const Estimate & estimate, const WorkFeatures & features, nlohmann::json details) const;

    nlohmann::json domain_;
    std::span<const TemplateDefinition> definitions_;
    std::vector<std::optional<std::size_t>> model_indices_;
    std::map<nlohmann::json, std::pair<std::size_t, WorkKind>> external_;
    std::vector<std::vector<PhaseModels>> models_;
};

} // namespace prologue
