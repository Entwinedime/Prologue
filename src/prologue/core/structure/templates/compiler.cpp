#include "prologue/core/structure/templates/compiler.hpp"
#include "prologue/core/structure/templates/library.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace prologue::structure {
namespace {

using nlohmann::json;
using Index = std::size_t;

struct Root final {
    Index call;
    json query;
    Index definition = 0;
};

class Definitions final {
public:
    Index insert(json definition) {
        const auto [position, inserted] = catalog_.emplace(definition.dump(), entries_.size());
        if (inserted) {
            entries_.push_back(std::move(definition));
        }
        return position->second;
    }

    json take() {
        return std::move(entries_);
    }

private:
    json entries_ = json::array();
    std::unordered_map<std::string, Index> catalog_;
};

json layouts(const json& ports) {
    auto result = json::array();
    for (const auto& port : ports) {
        auto layout = port;
        layout.erase("tensor");
        layout.erase("views");
        result.push_back(std::move(layout));
    }
    return result;
}

class PortBindings final {
public:
    enum class Use { source, target };

    PortBindings() : links_(json::array()) {}

    void bind(const json& ports, const json& child, Use use) {
        for (const auto& port : ports) {
            auto endpoint = json{{"child", child}, {"port", port.at("name")}};
            if (const auto views = port.find("views"); views != port.end()) {
                for (const auto& view : *views) {
                    endpoint["interval"] = view.at("interval");
                    connect(view.at("tensor"), endpoint, use);
                }
            } else {
                connect(port.at("tensor"), std::move(endpoint), use);
            }
        }
    }

    json take() {
        return std::move(links_);
    }

private:
    void connect(Index tensor, json endpoint, Use use) {
        if (use == Use::source) {
            producers_[tensor] = std::move(endpoint);
        } else if (const auto source = producers_.find(tensor); source != producers_.end()) {
            links_.push_back({{"from", source->second}, {"to", std::move(endpoint)}});
        }
    }

    std::unordered_map<Index, json> producers_;
    json links_;
};

class Calls final {
public:
    Calls(const json& calls, const json& observed_work)
        : calls_(calls), children_(calls.size()), work_(calls.size()) {
        if (!calls.is_array() || calls.empty() || !observed_work.is_array()) {
            throw std::invalid_argument("template compilation requires observed calls and device work");
        }

        for (Index index = 0; index < calls_.size(); ++index) {
            const auto& call = calls_.at(index);
            const auto& enclosing = call.at("parent");
            if (call.at("id") != index || (!enclosing.is_null() && enclosing.get<Index>() >= index)) {
                throw std::invalid_argument("calls must declare their parent before their children");
            }
            if (call.contains("template_query") && enclosing.is_null()) {
                throw std::invalid_argument("an imported call requires a parent scope");
            }
            if (!enclosing.is_null()) {
                const auto parent = enclosing.get<Index>();
                if (calls_.at(parent).contains("template_query")) {
                    throw std::invalid_argument("an imported call only observes ports");
                }
                children_.at(parent).push_back(index);
            }
        }

        std::unordered_map<Index, Index> lanes;
        for (const auto& entry : observed_work) {
            const auto owner = entry.at("owner").get<Index>();
            const auto& call = calls_.at(owner);
            const auto slot = entry.at("slot").get<Index>();
            if (call.contains("template_query")) {
                throw std::invalid_argument("an imported call cannot own sampled work");
            }
            if (slot > children_.at(owner).size()) {
                throw std::invalid_argument("device submission must fall between observed child calls");
            }

            // The common collective owns the result. Transport tasks are physical evidence.
            if (call.at("operation") == "collective") {
                continue;
            }
            const auto lane = lanes.try_emplace(entry.at("lane").get<Index>(), lanes.size()).first->second;
            work_.at(owner).push_back({{"slot", slot}, {"lane", lane},
                                       {"kind", entry.at("kind")}, {"name", entry.at("name")}});
        }

        // A template keeps order on each stream without baking in a race between independent lanes.
        for (auto& entries : work_) {
            std::ranges::stable_sort(entries, {}, [](const auto& entry) {
                return std::pair{entry.at("slot").template get<Index>(), entry.at("lane").template get<Index>()};
            });
        }
    }

    json definition(Index index, const std::vector<Index>& selected) const {
        const auto& call = calls_.at(index);
        const auto& direct = children_.at(index);
        PortBindings bindings;
        auto children = json::array();
        bindings.bind(call.at("inputs"), nullptr, PortBindings::Use::source);

        // Port references use a child ordinal and an interface name, independent of observation IDs.
        for (Index ordinal = 0; ordinal < direct.size(); ++ordinal) {
            const auto identity = direct.at(ordinal);
            const auto& child = calls_.at(identity);
            const auto& inputs = child.at("inputs");
            const auto& outputs = child.at("outputs");
            if (const auto query = child.find("template_query"); query != child.end()) {
                children.push_back({{"query", *query}, {"inputs", layouts(inputs)}, {"outputs", layouts(outputs)}});
            } else {
                children.push_back(selected.at(identity));
            }
            bindings.bind(inputs, ordinal, PortBindings::Use::target);
            bindings.bind(outputs, ordinal, PortBindings::Use::source);
        }
        bindings.bind(call.at("outputs"), nullptr, PortBindings::Use::target);

        return {
            {"operation", call.at("operation")}, {"implementation", call.at("implementation")},
            {"parameters", call.at("parameters")}, {"inputs", layouts(call.at("inputs"))},
            {"outputs", layouts(call.at("outputs"))}, {"children", std::move(children)},
            {"device_work", work_.at(index)}, {"tensor_links", bindings.take()},
        };
    }

private:
    const json& calls_;
    std::vector<std::vector<Index>> children_;
    std::vector<std::vector<json>> work_;
};

std::vector<Root> compile_calls(const json& observation, const std::unordered_set<std::string>& wanted,
                                Definitions& definitions) {
    const auto& calls = observation.at("calls");
    const Calls graph(calls, observation.at("work"));

    std::vector<Root> selected_roots;
    std::vector<bool> included(calls.size());
    for (Index index = 0; index < calls.size(); ++index) {
        const auto& call = calls.at(index);
        const auto& parent = call.at("parent");
        if (!parent.is_null()) {
            included.at(index) = included.at(parent.get<Index>());
            continue;
        }

        auto query = call.contains("query") ? call.at("query")
            : batch_geometry(call.at("parameters").at("phase"), call.at("members"));
        if (wanted.empty() || wanted.contains(query.dump())) {
            included.at(index) = true;
            selected_roots.push_back({index, std::move(query)});
        }
    }
    if (selected_roots.empty()) {
        throw std::invalid_argument("source observations did not cover the required root queries");
    }

    std::vector<Index> selected(calls.size());
    for (Index next = calls.size(); next > 0; --next) {
        const auto index = next - 1;
        if (!included.at(index) || calls.at(index).contains("template_query")) {
            continue;
        }
        selected.at(index) = definitions.insert(graph.definition(index, selected));
    }

    for (auto& root : selected_roots) {
        root.definition = selected.at(root.call);
    }
    return selected_roots;
}

std::string observed_action(const json& call) {
    auto members = json::array();
    for (const auto& member : call.at("members")) {
        members.push_back(json::array({member.at("request"), member.at("kv_committed_tokens"),
                                       member.at("output_tokens")}));
    }
    return json::array({call.at("parameters").at("phase"), std::move(members)}).dump();
}

json parallel_roots(const json& observation, const std::unordered_set<std::string>& wanted, Definitions& definitions) {
    using Actions = std::map<std::string, Root>;
    std::unordered_map<Index, Actions> workers;
    for (const auto& worker : observation.at("workers")) {
        auto [entry, inserted] = workers.try_emplace(worker.at("rank").get<Index>());
        if (!inserted) {
            throw std::invalid_argument("parallel source repeats a recorded rank");
        }
        const auto& document = worker.at("document");
        for (auto& root : compile_calls(document, wanted, definitions)) {
            auto key = observed_action(document.at("calls").at(root.call));
            if (!entry->second.emplace(std::move(key), std::move(root)).second) {
                throw std::invalid_argument("parallel source repeats an observed action");
            }
        }
    }

    const auto bindings = observation.at("rank_bindings").get<std::vector<Index>>();
    if (bindings.size() < 2) {
        throw std::invalid_argument("parallel source requires at least two bound ranks");
    }
    const auto publisher = observation.at("publication_rank").get<Index>();
    if (publisher >= bindings.size()) {
        throw std::invalid_argument("publication rank is outside the observed group");
    }
    auto ranks = json::array();
    for (Index rank = 0; rank < bindings.size(); ++rank) {
        ranks.push_back(rank);
    }
    const auto& first = workers.at(bindings.front());
    for (const auto& worker : workers) {
        if (worker.second.size() != first.size()) {
            throw std::invalid_argument("recorded ranks do not cover the same native actions");
        }
    }

    auto roots = json::array();
    for (const auto& [identity, root] : first) {
        auto children = json::array();
        for (const auto rank : bindings) {
            const auto& actions = workers.at(rank);
            const auto action = actions.find(identity);
            if (action == actions.end() || action->second.query != root.query) {
                throw std::invalid_argument("recorded ranks disagree on native action " + identity);
            }
            children.push_back(action->second.definition);
        }
        const auto identity_in_bank = definitions.insert({
            {"operation", "parallel_batch"}, {"implementation", "parallel_batch"},
            {"parameters", {{"ranks", ranks}, {"publication_rank", publisher}}}, {"children", std::move(children)},
            {"inputs", json::array()}, {"outputs", json::array()},
            {"device_work", json::array()}, {"tensor_links", json::array()},
        });
        roots.push_back({{"query", root.query}, {"definition", identity_in_bank}});
    }
    return roots;
}

}

nlohmann::json Compiler::compile(const nlohmann::json& observation) const {
    std::unordered_set<std::string> wanted;
    if (const auto required = observation.find("required_queries"); required != observation.end()) {
        for (const auto& query : *required) {
            wanted.insert(query.dump());
        }
    }
    Definitions definitions;
    auto roots = json::array();
    if (observation.contains("workers")) {
        roots = parallel_roots(observation, wanted, definitions);
    } else {
        for (auto& root : compile_calls(observation, wanted, definitions)) {
            roots.push_back({{"query", std::move(root.query)}, {"definition", root.definition}});
        }
    }
    return {
        {"domain", observation.at("domain")}, {"definitions", definitions.take()},
        {"roots", std::move(roots)}, {"evidence", observation.at("evidence")},
    };
}

}
