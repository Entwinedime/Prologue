#include "core/templates/library.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace prologue {
namespace {

auto member_extent(const nlohmann::json & member) {
    const auto fields = member_parameters(member);
    return std::pair{ fields.at("context_tokens"), fields.at("new_tokens") };
}

} // namespace

TemplateCall::Inputs TemplateCall::bind(const Parameters & parent, std::span<const nlohmann::json> members) const {
    Parameters features;
    if (row) {
        if (row->member >= members.size()) throw std::invalid_argument("row call has no admitted member at " + std::to_string(row->member));
        members = members.subspan(row->member, 1);
        features = member_parameters(members.front());
    }
    Inputs result{ {}, members };
    for (const auto & [name, expression] : arguments)
        result.parameters.emplace(name, expression.evaluate(row ? features : parent));
    return result;
}

std::expected<Parameters, nlohmann::json> TemplateDefinition::bind(Parameters arguments, const Parameters & constants) const {
    for (const auto & [name, value] : constants) {
        const auto [bound, inserted] = arguments.try_emplace(name, value);
        if (!inserted && bound->second != value)
            return std::unexpected(nlohmann::json{{"fixed", name}, {"expected", value}, {"actual", bound->second}});
    }

    const auto names = std::views::keys(arguments);
    if (std::ranges::equal(parameters, names)) return arguments;

    std::vector<std::string> missing, unexpected;
    std::ranges::set_difference(parameters, names, std::back_inserter(missing));
    std::ranges::set_difference(names, parameters, std::back_inserter(unexpected));
    return std::unexpected(nlohmann::json{{"missing", missing}, {"unexpected", unexpected}});
}

Parameters TemplateFrame::bind(const TemplateDefinition & body, Parameters arguments) const {
    auto input = body.bind(std::move(arguments), constants);
    if (!input) throw std::invalid_argument("template parameter contract at " + site + ": " + input.error().dump());
    return std::move(input.value());
}

ObservedArguments ObservedArguments::capture(const Parameters & parameters, std::span<const nlohmann::json> members) {
    ObservedArguments point{ parameters, {} };
    for (const auto & member : members) point.member_extents.push_back(member_extent(member));
    return point;
}

bool ObservedArguments::matches(const Parameters & arguments, std::span<const nlohmann::json> members) const {
    return parameters == arguments && (member_extents.empty() || std::ranges::equal(member_extents, members, {}, std::identity{}, member_extent));
}

bool TemplateGuard::evaluate(const Parameters & parameters) const {
    return extents.evaluate(parameters) && std::ranges::all_of(branches, [&](const Expression & rule) { return rule.evaluate(parameters) != 0; });
}

std::set<std::string> TemplateGuard::variables() const {
    auto result = extents.variables();
    for (const auto & rule : branches) result.merge(rule.variables());
    return result;
}

nlohmann::json StateBindings::contract() const {
    std::map<std::string, std::pair<std::int64_t, std::int64_t>> bases;
    for (const auto & value : kv) {
        auto & [layer, byte] = bases.try_emplace(value.pool, value.layer, value.slice.begin).first->second;
        layer = std::min(layer, value.layer);
        byte = std::min(byte, value.slice.begin);
    }
    std::map<std::string, std::size_t> pools;
    const auto pool = [&](const std::string & name) { return pools.try_emplace(name, pools.size()).first->second; };
    nlohmann::json result{{ "kv", nlohmann::json::array() }, { "slots", nlohmann::json::array() }};

    // Whole-pool relocation preserves relative layers and byte aliases, never overlapping independent slices.
    for (const auto & value : kv) {
        const auto & [first_layer, first_byte] = bases.at(value.pool);
        result["kv"].push_back({{ "pool", pool(value.pool) }, { "layer", value.layer - first_layer },
                                { "begin", value.slice.begin - first_byte },
                                { "token_bytes", value.slice.token_bytes }, { "bytes", value.slice.end - value.slice.begin }});
    }
    for (const auto & name : slots) result["slots"].push_back(pool(name));
    return result;
}

bool TemplateFrame::accepts_state(const std::map<std::string, StateBindings> & inputs) const {
    if (!state_input) return true;
    const auto it = inputs.find(*state_input);
    return it != inputs.end() && it->second.contract() == state.contract();
}

std::map<std::string, StateBindings> TemplateFrame::state_inputs() const {
    std::map<std::string, StateBindings> result;
    std::vector<const TemplateFrame *> pending{ this };
    for (std::size_t index = 0; index < pending.size(); ++index) {
        const auto & frame = *pending[index];
        if (frame.state_input) {
            const auto [found, inserted] = result.emplace(*frame.state_input, frame.state);
            if (!inserted && found->second != frame.state)
                throw std::invalid_argument("template state input has conflicting source bindings: " + *frame.state_input);
        }
        for (const auto & child : frame.children) pending.push_back(&child);
    }
    return result;
}

} // namespace prologue
