#include "prologue/core/structure/templates/compiler.hpp"
#include "prologue/core/structure/templates/library.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
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

        for (const auto& entry : observed_work) {
            const auto owner = entry.at("owner").get<Index>();
            const auto slot = entry.at("slot").get<Index>();
            if (calls_.at(owner).contains("template_query")) {
                throw std::invalid_argument("an imported call cannot own sampled work");
            }
            if (slot > children_.at(owner).size()) {
                throw std::invalid_argument("device submission must fall between observed child calls");
            }
            work_.at(owner).push_back({{"slot", slot}, {"lane", entry.at("lane")},
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

}

nlohmann::json Compiler::compile(const nlohmann::json& observation) const {
    const auto& calls = observation.at("calls");
    const Calls graph(calls, observation.at("work"));

    const auto required = observation.find("required_queries");
    std::unordered_set<std::string> target_keys;
    if (required != observation.end()) {
        for (const auto& query : *required) {
            target_keys.insert(query.dump());
        }
    }

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
        if (required == observation.end() || target_keys.contains(query.dump())) {
            included.at(index) = true;
            selected_roots.push_back({index, std::move(query)});
        }
    }
    if (selected_roots.empty()) {
        throw std::invalid_argument("source observations did not cover the required root queries");
    }

    auto definitions = json::array();
    std::unordered_map<std::string, Index> catalog;
    std::vector<Index> selected(calls.size());
    for (Index next = calls.size(); next > 0; --next) {
        const auto index = next - 1;
        if (!included.at(index) || calls.at(index).contains("template_query")) {
            continue;
        }
        auto definition = graph.definition(index, selected);
        const auto [position, inserted] = catalog.emplace(definition.dump(), definitions.size());
        if (inserted) {
            definitions.push_back(std::move(definition));
        }
        selected.at(index) = position->second;
    }

    auto roots = json::array();
    for (auto& root : selected_roots) {
        roots.push_back({{"query", std::move(root.query)}, {"definition", selected.at(root.call)}});
    }
    return {
        {"domain", observation.at("domain")}, {"definitions", std::move(definitions)},
        {"roots", std::move(roots)}, {"evidence", observation.at("evidence")},
    };
}

}
