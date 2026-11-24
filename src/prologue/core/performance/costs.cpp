#include "prologue/core/performance/costs.hpp"

#include "prologue/core/structure/templates/library.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <unordered_set>
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

json missing_cost(bool covered_domain) {
    if (covered_domain) {
        return {{"status", "cost_gap"}, {"reason", "no cost model covers the work coordinate"}};
    }
    return {{"status", "cost_domain_gap"}, {"reason", "cost asset does not cover the requested domain"}};
}

}

CostGap::CostGap(json gap) : std::runtime_error(gap.at("reason").get<std::string>()), detail(std::move(gap)) {}

CostQueries::CostQueries(const json& action, const json& graph)
    : host{{"family", "host_thread"}, {"geometry", action.contains("query")
        ? action.at("query")
        : structure::batch_geometry(action.at("phase"), action.at("members"), action.value("expert_routes", json{}))}},
      device(json::array()) {
    auto positions = json::array();
    const auto& members = action.at("members");
    for (std::size_t index = 0; index < members.size(); ++index) {
        if (members.at(index).value("publishes", true)) {
            positions.push_back(index);
        }
    }
    if (positions.size() != members.size()) {
        host["publication_positions"] = std::move(positions);
    }

    std::unordered_map<std::string, const json*> scopes;
    for (const auto& call : graph.at("calls")) {
        scopes.emplace(call.at("path").dump(), &call);
    }

    // Identity is local to a reusable call, independent of request and layer instance IDs.
    for (const auto& work : graph.at("work")) {
        const auto& call = *scopes.at(work.at("path").dump());
        auto coordinate = json{
            {"family", "device_task"}, {"geometry", host.at("geometry")},
            {"operation", call.at("operation")}, {"implementation", call.at("implementation")},
            {"parameters", call.at("parameters")}, {"inputs", call.at("inputs")}, {"outputs", call.at("outputs")},
            {"kind", work.at("kind")}, {"name", work.at("name")}, {"slot", work.at("slot")},
            {"ordinal", work.at("ordinal")},
        };
        if (work.contains("collective")) {
            coordinate["family"] = "collective";
            coordinate["rank"] = work.at("path").front();
            const auto& group = graph.at("collectives").at(work.at("collective").get<std::size_t>());
            if (group.contains("transfers")) {
                coordinate["transfers"] = group.at("transfers");
            }
        }
        device.push_back(std::move(coordinate));
    }
}

CostCatalog::CostCatalog(const json& asset, const json& rules) {
    using Decoder = std::unique_ptr<Estimator> (*)(const Samples&);
    static const std::unordered_map<std::string, Decoder> decoders{{"exact_median", exact_median}};

    for (const auto& domain : asset.at("domains")) {
        auto [entry, inserted] = domains_.try_emplace(domain.at("domain").dump());
        if (!inserted) {
            throw std::invalid_argument("multiple cost assets for one domain");
        }

        std::unordered_map<std::string, Samples> samples;
        for (const auto& row : domain.at("models")) {
            const auto family = row.at("coordinate").at("family").get<std::string>();
            if (rules.at(family).at("estimator") != row.at("estimator")) {
                throw std::invalid_argument("trained estimator does not match its rule for " + family);
            }
            samples[family].push_back(&row);
        }
        for (const auto& binding : rules.items()) {
            const auto name = binding.value().at("estimator").get<std::string>();
            const auto decoder = decoders.find(name);
            if (decoder == decoders.end()) {
                throw std::invalid_argument("unsupported cost estimator " + name);
            }
            entry->second.emplace(binding.key(), decoder->second(samples[binding.key()]));
        }
    }
}

CostCatalog::~CostCatalog() = default;

const CostCatalog::Models* CostCatalog::resolve(const json& domain) const {
    const auto entry = domains_.find(domain.dump());
    return entry == domains_.end() ? nullptr : &entry->second;
}

std::optional<json> CostCatalog::estimate(const Models* models, const json& coordinate) {
    if (!models) {
        return std::nullopt;
    }
    const auto model = models->find(coordinate.at("family").get<std::string>());
    return model == models->end() ? std::nullopt : model->second->estimate(coordinate);
}

std::vector<json> CostCatalog::require(const json& domain, std::span<const json* const> coordinates) const {
    const auto* models = resolve(domain);
    std::vector<json> estimates;
    estimates.reserve(coordinates.size());
    auto missing = json::array();
    std::unordered_set<std::string> reported;
    for (const auto* coordinate : coordinates) {
        if (auto value = estimate(models, *coordinate)) {
            estimates.push_back(std::move(*value));
            continue;
        }
        if (reported.insert(coordinate->dump()).second) {
            missing.push_back(*coordinate);
        }
    }

    // A caller can schedule the action only after all of its costs are known.
    if (!missing.empty()) {
        auto gap = missing_cost(models != nullptr);
        gap["domain"] = domain;
        gap["coordinates"] = std::move(missing);
        throw CostGap(std::move(gap));
    }
    return estimates;
}

json CostCatalog::quotes(const json& domain, const json& coordinates) const {
    const auto* models = resolve(domain);
    auto result = json::array();
    for (const auto& coordinate : coordinates) {
        if (auto value = estimate(models, coordinate)) {
            result.push_back({{"status", "covered"}, {"estimate", std::move(*value)}});
        } else {
            auto gap = missing_cost(models != nullptr);
            gap["coordinate"] = coordinate;
            result.push_back(std::move(gap));
        }
    }
    return result;
}

}
