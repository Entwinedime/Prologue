#include "prologue/core/structure/templates/bank.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace prologue::structure {
namespace {

using nlohmann::json;
using Index = std::size_t;

class DefinitionPool final {
public:
    explicit DefinitionPool(const json& rule)
        : shared_operations_(rule.at("shared_operations").get<std::unordered_set<std::string>>()) {}

    json roots(const json& source, const json& definitions) {
        const auto& domain = source.at("domain");
        const auto domain_key = domain.dump();
        const auto revision_key = domain.at("framework_revision").dump();

        // Each source keeps its observed root claims. Only definition storage is shared.
        auto roots = source.at("roots");
        std::unordered_map<Index, Index> remapped;
        for (auto& root : roots) {
            root.at("definition") = intern(definitions, root.at("definition"), domain_key, revision_key, remapped);
        }
        return roots;
    }

    json take() {
        return std::move(definitions_);
    }

private:
    Index intern(const json& source, Index identity, const std::string& domain, const std::string& revision,
                 std::unordered_map<Index, Index>& remapped) {
        if (const auto known = remapped.find(identity); known != remapped.end()) {
            return known->second;
        }

        auto definition = source.at(identity);
        auto& children = definition.at("children");
        for (auto& child : children) {
            if (child.is_number_integer()) {
                const auto local = child.get<Index>();
                if (local >= identity) {
                    throw std::invalid_argument("template child must precede its parent definition");
                }
                child = intern(source, local, domain, revision, remapped);
            }
        }

        // A leaf needs local work and an exact body to share storage across observed domains.
        const bool shared = children.empty() && !definition.at("device_work").empty() &&
            shared_operations_.contains(definition.at("operation").get<std::string>());
        auto key = (shared ? revision : domain) + '\n' + definition.dump();
        const auto [slot, inserted] = identities_.try_emplace(std::move(key), definitions_.size());
        if (inserted) {
            definitions_.push_back(std::move(definition));
        }
        const auto normalized = slot->second;
        remapped.emplace(identity, normalized);
        return normalized;
    }

    std::unordered_set<std::string> shared_operations_;
    std::unordered_map<std::string, Index> identities_;
    json definitions_ = json::array();
};

void record_model(json& models, const json& model, const json& configuration) {
    const auto existing = std::ranges::find_if(models, [&](const auto& row) { return row.at("model") == model; });
    if (existing == models.end()) {
        models.push_back({{"model", model}, {"model_config", configuration}});
    } else if (existing->at("model_config") != configuration) {
        throw std::invalid_argument("template sources disagree on the model configuration");
    }
}

void record_source(json& sources, json source) {
    if (std::ranges::find(sources, source) == sources.end()) {
        sources.push_back(std::move(source));
    }
}

}

nlohmann::json assemble_bank(const nlohmann::json& input) {
    DefinitionPool pool(input.at("rule"));
    auto models = json::array();
    auto sources = json::array();

    const auto& previous = input.at("previous");
    if (!previous.is_null()) {
        models = previous.at("models");
        for (const auto& source : previous.at("sources")) {
            auto scoped = source;
            scoped.at("roots") = pool.roots(source, previous.at("definitions"));
            sources.push_back(std::move(scoped));
        }
    }
    for (const auto& source : input.at("sources")) {
        const auto& domain = source.at("domain");
        record_model(models, domain.at("model"), source.at("model_config"));
        record_source(sources, {{"domain", domain}, {"source", source.at("source")},
                                {"roots", pool.roots(source, source.at("definitions"))}});
    }
    return {{"definitions", pool.take()}, {"models", std::move(models)}, {"sources", std::move(sources)}};
}

}
