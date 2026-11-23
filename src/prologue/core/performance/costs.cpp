#include "prologue/core/performance/costs.hpp"

#include <utility>

namespace prologue::performance {
namespace {

using nlohmann::json;

json geometry(const json& action) {
    auto lengths = json::array();
    for (const auto& member : action.at("members")) {
        lengths.push_back(member.at("kv_committed_tokens"));
    }
    return {{"phase", action.at("phase")}, {"sequence_lengths", std::move(lengths)}};
}

}

CostGap::CostGap(json gap) : std::runtime_error(gap.at("reason").get<std::string>()), detail(std::move(gap)) {}

CostQueries::CostQueries(const json& action, const json& graph)
    : host{{"family", "host_thread"}, {"geometry", geometry(action)}}, device(json::array()) {
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

CostCatalog::CostCatalog(const json& asset) : domain_(asset.at("domain")) {
    if (asset.at("estimator") != "exact_median") {
        throw std::invalid_argument("unsupported cost estimator");
    }
    for (const auto& entry : asset.at("models")) {
        const auto key = entry.at("coordinate").dump();
        if (!estimates_.emplace(key, entry.at("estimate")).second) {
            throw std::invalid_argument("multiple cost models for one exact coordinate");
        }
    }
}

const json& CostCatalog::require(const json& domain, const json& coordinate) const {
    if (domain != domain_) {
        throw CostGap({{"status", "cost_domain_gap"}, {"reason", "cost asset does not cover the requested domain"},
                       {"coordinate", coordinate}});
    }
    const auto entry = estimates_.find(coordinate.dump());
    if (entry == estimates_.end()) {
        throw CostGap({{"status", "cost_gap"}, {"reason", "no cost model covers the exact work coordinate"},
                       {"coordinate", coordinate}});
    }
    return entry->second;
}

json CostCatalog::quote(const json& domain, const json& coordinate) const {
    try {
        return {{"status", "covered"}, {"estimate", require(domain, coordinate)}};
    } catch (const CostGap& gap) {
        return gap.detail;
    }
}

}
