#include "prologue/core/structure/templates/library.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prologue::structure {

using nlohmann::json;
using Index = std::size_t;

TemplateGap::TemplateGap(json gap) : std::runtime_error(gap.at("reason").get<std::string>()), detail(std::move(gap)) {}

class Templates::Expansion final {
public:
    Expansion(const Templates& library, const json& request) : library_(library), request_(request) {}

    json resolve(const json& query) {
        const auto key = query.dump();
        const auto candidates = library_.queries_.find(key);
        if (candidates == library_.queries_.end()) {
            throw TemplateGap({{"status", active_.empty() ? "parent_gap" : "child_gap"},
                               {"reason", "no template covers the required query"},
                               {"query", request_}, {"required", query}});
        }
        if (std::ranges::find(active_, key) != active_.end()) {
            throw std::invalid_argument("template queries form a cycle");
        }

        active_.push_back(key);
        json selected;
        for (const auto& candidate : candidates->second) {
            auto graph = json{{"calls", json::array()}, {"work", json::array()}};
            std::vector<Index> path;
            const auto& definitions = library_.libraries_.at(candidate.library).at("definitions");
            visit(definitions, candidate.definition, path, graph);
            if (!selected.is_null() && selected != graph) {
                throw TemplateGap({{"status", "candidate_conflict"}, {"reason", "template candidates disagree"},
                                   {"query", request_}, {"required", query}});
            }
            selected = std::move(graph);
        }
        active_.pop_back();
        return selected;
    }

private:
    void visit(const json& definitions, Index identity, std::vector<Index>& path, json& graph) {
        const auto& definition = definitions.at(identity);
        auto call = definition;
        call.erase("children");
        call.erase("device_work");
        call["path"] = path;
        graph.at("calls").push_back(std::move(call));

        const auto& work = definition.at("device_work");
        const auto& children = definition.at("children");
        Index next = 0;
        for (Index slot = 0; slot <= children.size(); ++slot) {
            while (next < work.size() && work.at(next).at("slot") == slot) {
                auto entry = work.at(next);
                entry["path"] = path;
                entry["ordinal"] = next++;
                graph.at("work").push_back(std::move(entry));
            }
            if (slot == children.size()) {
                break;
            }

            path.push_back(slot);
            const auto& child = children.at(slot);
            if (child.is_number_integer()) {
                const auto index = child.get<Index>();
                if (index >= identity) {
                    throw std::invalid_argument("template child must precede its parent definition");
                }
                visit(definitions, index, path, graph);
            } else {
                auto resolved = resolve(child.at("query"));
                const auto& ports = resolved.at("calls").front();
                if (ports.at("inputs") != child.at("inputs") || ports.at("outputs") != child.at("outputs")) {
                    throw std::invalid_argument("an imported template does not match its call ports");
                }
                append(std::move(resolved), path, graph);
            }
            path.pop_back();
        }
        if (next != work.size()) {
            throw std::invalid_argument("template work is outside its child slots");
        }
    }

    static void append(json child, const std::vector<Index>& prefix, json& graph) {
        // Keep reusable definitions relative to their own root and bind their paths at the call site.
        for (const auto* kind : {"calls", "work"}) {
            for (auto& row : child.at(kind)) {
                auto path = prefix;
                for (const auto& index : row.at("path")) {
                    path.push_back(index.get<Index>());
                }
                row["path"] = std::move(path);
                graph.at(kind).push_back(std::move(row));
            }
        }
    }

    const Templates& library_;
    const json& request_;
    std::vector<std::string> active_;
};

json batch_geometry(const json& phase, const json& members) {
    auto sequence = json::array();
    for (const auto& member : members) {
        sequence.push_back(member.at("kv_committed_tokens"));
    }
    return {{"phase", phase}, {"sequence_lengths", std::move(sequence)}};
}

Templates::Templates(const json& libraries, const json& rule) : libraries_(libraries) {
    if (rule.at("selector") != "exact_query") {
        throw std::invalid_argument("unsupported template selector");
    }
    for (Index source = 0; source < libraries_.size(); ++source) {
        for (const auto& root : libraries_.at(source).at("roots")) {
            queries_[root.at("query").dump()].push_back({source, root.at("definition").get<Index>()});
        }
    }
}

json Templates::expand(const json& query) const {
    return Expansion(*this, query).resolve(query);
}

void Templates::qualify() const {
    if (queries_.empty()) {
        throw std::invalid_argument("template sources have no root candidates");
    }
    for (const auto& [query, candidates] : queries_) {
        expand(json::parse(query));
    }
}

}
