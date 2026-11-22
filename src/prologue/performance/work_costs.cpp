#include "prologue/performance/work_costs.hpp"

#include <ranges>
#include <utility>

#include "prologue/graph/graph.hpp"
#include "prologue/performance/work_features.hpp"
#include "prologue/templates/library.hpp"

namespace prologue {

namespace {

class RecordedFeatures final : public FeatureValues {
public:
    explicit RecordedFeatures(const nlohmann::json & document)
        : document_(document), phase_(document.at("phase").get_ref<const std::string &>()),
          axes_(document.at("axes").get_ref<const nlohmann::json::object_t &>()) {}

    std::string_view phase() const { return phase_; }

    const nlohmann::json * transport() const {
        const auto found = document_.find("transport");
        return found == document_.end() ? nullptr : &*found;
    }

    std::optional<double> value(std::string_view axis) const override {
        const auto found = axes_.find(axis);
        if (found == axes_.end()) return {};
        return found->second.get<double>();
    }

private:
    const nlohmann::json & document_;
    std::string_view phase_;
    const nlohmann::json::object_t & axes_;
};

} // namespace

CostGap::CostGap(nlohmann::json details)
    : std::runtime_error("no qualified service cost at " + details.at("site").get<std::string>()),
      details_(std::move(details)) {}

template<class Definitions>
WorkCosts::WorkCosts(nlohmann::json domain, const nlohmann::json & asset, const Definitions & definitions)
    : domain_(std::move(domain)), model_indices_(definitions.size()) {
    const nlohmann::json * selected = nullptr;
    for (const auto & candidate : asset.at("domains").get_ref<const nlohmann::json::array_t &>()) {
        if (candidate.at("domain") != domain_) continue;
        if (selected) throw std::invalid_argument("cost asset repeats the selected implementation domain");
        selected = &candidate;
    }
    if (!selected) return;

    const auto & measured = selected->at("definitions").get_ref<const nlohmann::json::array_t &>();
    std::map<nlohmann::json, std::size_t> known;
    models_.resize(measured.size());
    for (std::size_t index = 0; index < measured.size(); ++index) {
        const auto & definition = measured[index];
        const auto & work = definition.at("work").get_ref<const nlohmann::json::array_t &>();
        if (definition.contains("external")) {
            if (work.size() != 1) throw std::invalid_argument("external cost definition must declare one work interval");
            const auto kind = work_kind(work.front().at("kind").get<std::string>());
            if (!external_.emplace(definition.at("external"), std::pair{ index, kind }).second)
                throw std::invalid_argument("cost asset repeats an external work definition");
        } else if (!known.emplace(definition, index).second) {
            throw std::invalid_argument("cost asset repeats a template definition");
        }
        models_[index].resize(work.size());
    }
    const auto & bindings = selected->at("models").get_ref<const nlohmann::json::array_t &>();
    for (std::size_t ordinal = 0; ordinal < bindings.size(); ++ordinal) {
        const auto & model = bindings[ordinal];
        const auto definition = model.at("definition").get<std::size_t>();
        const auto interval = model.at("interval").get<std::size_t>();
        auto & phases = models_.at(definition).at(interval);
        auto service = ServiceModel::decode(model, ordinal, measured.at(definition), interval);
        if (!phases.try_emplace(model.at("phase").get<std::string>(), std::move(service)).second)
            throw std::invalid_argument("cost asset repeats a work interval and phase");
    }

    // Work definitions may be shared; the models always come from the selected execution domain.
    std::size_t index = 0;
    for (const nlohmann::json & definition : definitions) {
        auto & model = model_indices_[index++];
        if (definition.contains("external")) {
            const auto kind = work_kind(definition.at("work").at(0).at("kind").get<std::string>());
            model = external_index(definition.at("external"), kind);
        } else if (const auto source = known.find(definition); source != known.end()) {
            model = source->second;
        }
    }
}

WorkCosts::WorkCosts(nlohmann::json domain, std::span<const TemplateDefinition> definitions, const nlohmann::json & asset)
    : WorkCosts(std::move(domain), asset, definitions | std::views::transform([](const auto & definition) {
        return definition.document();
    })) {
    definitions_ = definitions;
}

std::optional<std::size_t> WorkCosts::external_index(const nlohmann::json & identity, WorkKind kind) const {
    const auto found = external_.find(identity);
    if (found == external_.end()) return {};
    if (found->second.second != kind)
        throw std::invalid_argument("external cost definition changes the declared work kind");
    return found->second.first;
}

template<class Features>
WorkCosts::Estimate WorkCosts::estimate(std::optional<std::size_t> definition, std::size_t interval, const Features & features) const {
    if (!definition) return {};
    const auto & phases = models_[*definition].at(interval);
    const auto match = phases.find(features.phase());
    if (match == phases.end()) return {};
    return { match->second.get(), match->second->estimate(features) };
}

template<class Features>
WorkCosts::Estimate WorkCosts::work_estimate(std::optional<std::size_t> definition, std::size_t interval, WorkKind kind,
                                             const Features & features, const nlohmann::json * explicit_service) const {
    const auto * operand = features.transport();
    if (explicit_service) {
        if (operand && *operand != *explicit_service)
            throw std::invalid_argument("cost query operand differs from its declared external service");
        return estimate(definition, interval, features);
    }
    if (kind == WorkKind::Transfer && operand)
        return estimate(external_index(*operand, kind), 0, features);
    const auto measured = estimate(definition, interval, features);
    if (measured.service_ns || kind != WorkKind::Collective || !operand) return measured;

    // The profile point already binds the actual payload and participants. It supplies only the peer service.
    const auto point = external_index(*operand, kind);
    return point ? estimate(point, 0, WorkFeatures(Parameters{})) : measured;
}

std::string_view WorkCosts::Estimate::status() const {
    if (service_ns) return "covered";
    return model ? "outside_sample_ranges" : "unseen_work";
}

CostGap WorkCosts::gap(const Estimate & estimate, const WorkFeatures & features, nlohmann::json details) const {
    auto bounds = nlohmann::json::object();
    if (estimate.model) {
        for (const auto & [name, bound] : estimate.model->bounds) bounds[name] = { bound.lower, bound.upper };
    }
    details["status"] = estimate.status();
    details["domain"] = domain_;
    details["features"] = features.document();
    details["sample_bounds"] = std::move(bounds);
    return CostGap(std::move(details));
}

GraphTiming WorkCosts::timing(const CoarseGraph & graph, std::span<const TemplateInterval> origins, nlohmann::json * first_gap,
                             bool record_bindings) const {
    if (origins.size() != graph.work.size()) throw std::logic_error("cost queries require one origin per instantiated work interval");
    WorkCoordinates coordinates(graph);
    GraphTiming result;
    result.work_ns.reserve(graph.work.size());
    if (record_bindings) result.model_bindings.reserve(graph.work.size());
    for (std::size_t index = 0; index < graph.work.size(); ++index) {
        const auto & work = graph.work[index];
        const auto & origin = origins[index];
        const auto & scope = graph.scopes[work.scope];
        const auto & geometry = coordinates[coordinates.index(work)];
        const auto cost = work_estimate(model_indices_.at(origin.definition), origin.interval, work.kind, geometry);
        if (!cost.service_ns && (!first_gap || first_gap->is_null())) {
            auto missing = gap(cost, geometry, {
                { "action_site", graph.scopes.front().info.site },
                { "site", scope.info.site }, { "implementation", scope.info.implementation },
                { "scope", scope.info.role }, { "work", work_name(work.kind) }, { "resource", graph.resources[work.resource].role },
                { "definition", definitions_[origin.definition].document() }, { "interval", origin.interval },
            });
            if (!first_gap) throw missing;
            *first_gap = missing.details();
            (*first_gap)["work_index"] = index;
        }
        result.work_ns.push_back(cost.service_ns);
        if (record_bindings) result.model_bindings.push_back(cost.model ? std::optional(cost.model->binding) : std::nullopt);
    }
    return result;
}

std::int64_t WorkCosts::external(const nlohmann::json & identity, WorkKind kind, const WorkFeatures & features) const {
    const auto cost = work_estimate(external_index(identity, kind), 0, kind, features, &identity);
    if (cost.service_ns) return *cost.service_ns;

    throw gap(cost, features, {{ "site", identity.at("operation") }, { "work", work_name(kind) },
                                   { "definition", {{ "external", identity }, { "work", {{{ "kind", work_name(kind) }}} }} },
                                   { "interval", 0 }});
}

nlohmann::json WorkCosts::evaluate(const nlohmann::json & groups, const nlohmann::json & asset, bool bindings) {
    auto results = nlohmann::json::array();
    for (const auto & group : groups.get_ref<const nlohmann::json::array_t &>()) {
        const auto & definitions = group.at("definitions").get_ref<const nlohmann::json::array_t &>();
        const WorkCosts costs(group.at("domain"), asset, definitions);
        auto estimates = nlohmann::json::array();
        for (const auto & point : group.at("points").get_ref<const nlohmann::json::array_t &>()) {
            if (!point.is_array() || point.size() != 3)
                throw std::invalid_argument("cost point requires a definition, interval, and features");
            const auto definition = point[0].get<std::size_t>();
            const auto interval = point[1].get<std::size_t>();
            const auto & declared = definitions.at(definition);
            const auto kind = work_kind(declared.at("work").at(interval).at("kind").get<std::string>());
            const auto service = declared.find("external");
            const auto estimate = costs.work_estimate(costs.model_indices_[definition], interval, kind, RecordedFeatures(point[2]),
                                                      service == declared.end() ? nullptr : &*service);
            auto row = nlohmann::json::array({estimate.status(), estimate.service_ns ? nlohmann::json(*estimate.service_ns) : nullptr});
            if (bindings) row.push_back(estimate.model ? nlohmann::json(estimate.model->binding) : nullptr);
            estimates.push_back(std::move(row));
        }
        results.push_back(std::move(estimates));
    }
    return results;
}

} // namespace prologue
