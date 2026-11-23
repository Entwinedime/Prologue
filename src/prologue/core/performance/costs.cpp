#include "prologue/core/performance/costs.hpp"

#include "prologue/core/structure/templates/library.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace prologue::performance {

using nlohmann::json;

class Estimator {
public:
    virtual ~Estimator() = default;
    virtual std::optional<json> estimate(const json& coordinate) const = 0;
};

namespace {

using Samples = std::vector<const json*>;

class ExactMedian final : public Estimator {
public:
    explicit ExactMedian(const Samples& rows) {
        for (const auto* row : rows) {
            const auto key = row->at("coordinate").dump();
            const auto& estimate = row->at("estimate");
            if (!estimate.is_object()) {
                throw std::invalid_argument("a cost estimate must contain named values");
            }
            if (!estimates_.emplace(key, estimate).second) {
                throw std::invalid_argument("multiple cost models for one exact coordinate");
            }
        }
    }

    std::optional<json> estimate(const json& coordinate) const override {
        const auto entry = estimates_.find(coordinate.dump());
        if (entry == estimates_.end()) {
            return std::nullopt;
        }
        return entry->second;
    }

private:
    std::unordered_map<std::string, json> estimates_;
};

std::unique_ptr<Estimator> exact_median(const Samples& rows) {
    return std::make_unique<ExactMedian>(rows);
}

}

CostGap::CostGap(json gap) : std::runtime_error(gap.at("reason").get<std::string>()), detail(std::move(gap)) {}

CostQueries::CostQueries(const json& action, const json& graph)
    : host{{"family", "host_thread"}, {"geometry", structure::batch_geometry(action.at("phase"), action.at("members"))}},
      device(json::array()) {
    std::unordered_map<std::string, const json*> scopes;
    for (const auto& call : graph.at("calls")) {
        scopes.emplace(call.at("path").dump(), &call);
    }

    // Identity is local to a reusable call, independent of request and layer instance IDs.
    for (const auto& work : graph.at("work")) {
        const auto& call = *scopes.at(work.at("path").dump());
        device.push_back({
            {"family", "device_task"}, {"geometry", host.at("geometry")},
            {"operation", call.at("operation")}, {"implementation", call.at("implementation")},
            {"parameters", call.at("parameters")}, {"inputs", call.at("inputs")}, {"outputs", call.at("outputs")},
            {"kind", work.at("kind")}, {"name", work.at("name")}, {"slot", work.at("slot")},
            {"ordinal", work.at("ordinal")},
        });
    }
}

CostCatalog::CostCatalog(const json& asset, const json& rules) : domain_(asset.at("domain")) {
    using Decoder = std::unique_ptr<Estimator> (*)(const Samples&);
    static const std::unordered_map<std::string, Decoder> decoders{{"exact_median", exact_median}};

    std::unordered_map<std::string, Samples> samples;
    for (const auto& entry : asset.at("models")) {
        const auto family = entry.at("coordinate").at("family").get<std::string>();
        if (rules.at(family).at("estimator") != entry.at("estimator")) {
            throw std::invalid_argument("trained estimator does not match its rule for " + family);
        }
        samples[family].push_back(&entry);
    }
    for (const auto& binding : rules.items()) {
        const auto name = binding.value().at("estimator").get<std::string>();
        const auto decoder = decoders.find(name);
        if (decoder == decoders.end()) {
            throw std::invalid_argument("unsupported cost estimator " + name);
        }
        models_.emplace(binding.key(), decoder->second(samples[binding.key()]));
    }
}

CostCatalog::~CostCatalog() = default;

json CostCatalog::require(const json& domain, const json& coordinate) const {
    if (domain != domain_) {
        throw CostGap({{"status", "cost_domain_gap"}, {"reason", "cost asset does not cover the requested domain"},
                       {"coordinate", coordinate}});
    }
    const auto model = models_.find(coordinate.at("family").get<std::string>());
    if (model != models_.end()) {
        if (auto estimate = model->second->estimate(coordinate)) {
            return std::move(*estimate);
        }
    }
    throw CostGap({{"status", "cost_gap"}, {"reason", "no cost model covers the work coordinate"},
                   {"coordinate", coordinate}});
}

json CostCatalog::quote(const json& domain, const json& coordinate) const {
    try {
        return {{"status", "covered"}, {"estimate", require(domain, coordinate)}};
    } catch (const CostGap& gap) {
        return gap.detail;
    }
}

}
