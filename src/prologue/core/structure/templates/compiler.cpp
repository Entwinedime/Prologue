#include "prologue/core/structure/templates/compiler.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace prologue::structure {
namespace {

using nlohmann::json;
using Index = std::size_t;

json layouts(const json& ports) {
    auto result = json::array();
    for (const auto& port : ports) {
        auto layout = port;
        layout.erase("tensor");
        result.push_back(std::move(layout));
    }
    return result;
}

class PortBindings final {
public:
    PortBindings() : links_(json::array()) {}

    void publish(const json& ports, const json& child) {
        for (const auto& port : ports) {
            producers_[port.at("tensor").get<Index>()] = endpoint(child, port);
        }
    }

    void consume(const json& ports, const json& child) {
        for (const auto& port : ports) {
            const auto producer = producers_.find(port.at("tensor").get<Index>());
            if (producer != producers_.end()) {
                links_.push_back({{"from", producer->second}, {"to", endpoint(child, port)}});
            }
        }
    }

    json take() {
        return std::move(links_);
    }

private:
    static json endpoint(const json& child, const json& port) {
        return {{"child", child}, {"port", port.at("name")}};
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
            if (!enclosing.is_null()) {
                children_.at(enclosing.get<Index>()).push_back(index);
            }
        }

        for (const auto& entry : observed_work) {
            const auto owner = entry.at("owner").get<Index>();
            const auto slot = entry.at("slot").get<Index>();
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
        bindings.publish(call.at("inputs"), nullptr);

        // Port references use a child ordinal and an interface name, independent of observation IDs.
        for (Index ordinal = 0; ordinal < direct.size(); ++ordinal) {
            const auto child = direct.at(ordinal);
            children.push_back(selected.at(child));
            bindings.consume(calls_.at(child).at("inputs"), ordinal);
            bindings.publish(calls_.at(child).at("outputs"), ordinal);
        }
        bindings.consume(call.at("outputs"), nullptr);

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
    const auto& work = observation.at("work");
    Calls graph(calls, work);

    auto definitions = json::array();
    std::unordered_map<std::string, Index> catalog;
    std::vector<Index> selected(calls.size());
    for (Index next = calls.size(); next > 0; --next) {
        const auto index = next - 1;
        auto definition = graph.definition(index, selected);
        const auto [position, inserted] = catalog.emplace(definition.dump(), definitions.size());
        if (inserted) {
            definitions.push_back(std::move(definition));
        }
        selected.at(index) = position->second;
    }

    // Preserve concrete tensor identities, call sites and timing only on the observed instances.
    auto instances = calls;
    for (Index index = 0; index < instances.size(); ++index) {
        instances.at(index)["definition"] = selected.at(index);
    }
    return {
        {"domain", observation.at("domain")}, {"definitions", std::move(definitions)},
        {"instances", std::move(instances)}, {"evidence", observation.at("evidence")},
        {"work", work}, {"stream_order", observation.at("stream_order")},
    };
}

}
