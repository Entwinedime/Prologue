#include "prologue/templates/selection.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace prologue {
namespace {

bool same_realization(const TemplateFrame & left, const TemplateFrame & right) {
    if (left.definition != right.definition || left.children.size() != right.children.size()) return false;
    auto first = left.local_document(), second = right.local_document();
    first.erase("guard");
    second.erase("guard");
    return first == second && std::ranges::equal(left.children, right.children, same_realization);
}

struct FrameChoice {
    std::optional<TemplateFrame> frame;
    std::optional<std::string> ambiguity;
    std::optional<nlohmann::json> rejected_guard;
    std::optional<nlohmann::json> rejected_parameters;

    void merge(FrameChoice candidate, std::string_view site) {
        if (!rejected_guard) rejected_guard = std::move(candidate.rejected_guard);
        if (!rejected_parameters) rejected_parameters = std::move(candidate.rejected_parameters);
        if (candidate.ambiguity) {
            frame.reset();
            ambiguity = std::move(candidate.ambiguity);
        }
        else if (candidate.frame) {
            if (frame && same_realization(*frame, *candidate.frame)) return;
            if (frame || ambiguity) {
                frame.reset();
                if (!ambiguity) ambiguity = std::string(site);
            }
            else frame = std::move(candidate.frame);
        }
    }
};

} // namespace

class TemplateSelection::FamilyIndex {
    using Callers = std::map<std::optional<std::size_t>, std::size_t>;
    using Body = std::pair<std::size_t, nlohmann::json>;

public:
    explicit FamilyIndex(TemplateSelection & index) : index_(index), library_(index.library_) {}

    void build() {
        for (const auto & implementation : library_.implementations) implementations_[implementation.contract].push_back(&implementation);
        for (const auto & entry : library_.entries) index_.roots_[{entry.action, entry.condition}].insert(add(entry, entry.frame));
        for (const auto & [site, callers] : sites_) {
            if (callers.size() > 1) share_observed_bodies(callers);
        }
    }

private:
    std::size_t add(const TemplateEntry & entry, const TemplateFrame & frame, const std::string & space = "scope/", std::optional<std::size_t> caller = {},
                    std::vector<std::size_t> position = {}) {
        for (std::size_t child = 0; child < frame.children.size(); ++child) {
            position.push_back(child);
            add(entry, frame.children[child], space + std::to_string(child) + "/", frame.definition, position);
            position.pop_back();
        }
        auto local = frame.local_document();
        local.erase("definition");
        local.erase("guard");
        if (frame.state_input) local["state"] = frame.state.contract();
        if (frame.communication && frame.bindings.contains("required_peers")) local["communication"].erase("required");
        local["interface"] = library_.call_interface(frame).document();
        std::set<nlohmann::json> shared;
        std::map<std::string, Expression> arguments;
        for (const auto & parameter : library_.definitions[frame.definition].parameters) arguments.emplace(parameter, Expression::read(parameter));
        for (const auto & [name, value] : frame.constants) arguments.insert_or_assign(name, Expression::read(value));
        foreign_bindings(frame, space, space, arguments, shared);
        local["shared_bindings"] = std::move(shared);
        if (caller) {
            const auto & call = library_.definitions[*caller].children[position.back()];
            for (const auto & [name, expression] : call.arguments) local["arguments"][name] = expression.document();
        }
        auto & callers = sites_[{entry.action, entry.condition, space, std::move(local)}];
        const auto [found, inserted] = callers.try_emplace(caller, index_.families_.size());
        if (inserted) index_.families_.emplace_back();
        auto & family = index_.families_[found->second];
        family.push_back({&frame, frame.definition, &frame.guard});

        const auto contract = library_.local_contract(frame, position);
        if (const auto available = implementations_.find(contract); available != implementations_.end()) {
            for (const auto * implementation : available->second) {

                // Rebind only the observed body and its guard; the recipient retains all caller and rank identities.
                auto replacement = frame;
                replacement.definition = implementation->definition;
                replacement.guard = implementation->guard;
                if (library_.local_contract(replacement, position) != contract)
                    throw std::invalid_argument("local implementation changes the recipient contract");
                family.push_back({&frame, implementation->definition, &implementation->guard});
            }
        }
        index_.membership_.emplace(&frame, found->second);
        return found->second;
    }

    const nlohmann::json & child_contracts(std::size_t definition) {
        auto [found, inserted] = binding_contracts_.try_emplace(definition);
        if (inserted) {
            auto body = library_.definitions[definition].document();

            // Child imports bind to the selected body's own events. Child arguments, interfaces and state effects must agree.
            for (auto & call : body["children"]) call.erase("imports");
            found->second = {std::move(body["children"]), std::move(body["resources"]),
                             std::move(body["kv_accesses"]), std::move(body["slot_accesses"])};
        }
        return found->second;
    }

    void share_observed_bodies(const Callers & callers) {
        using Recipient = std::pair<std::size_t, const TemplateFrame *>;
        std::map<Body, std::vector<Recipient>> witnesses;
        for (const auto & [caller, family] : callers) {
            for (const auto & candidate : index_.families_[family]) {
                const auto * frame = candidate.bindings;
                if (candidate.guard == &frame->guard)
                    witnesses[{frame->definition, frame->guard.document()}].emplace_back(family, frame);
            }
        }
        if (witnesses.size() < 2 || std::ranges::none_of(witnesses, [](const auto & witness) {
                const auto & observed = witness.second;
                return observed.front().first != observed.back().first;
            })) return;

        // A source observation links two callers through a common body. Transfer only changes observed with identical descendants.
        std::map<std::pair<std::size_t, nlohmann::json>, std::map<Body, const TemplateFrame *>> variations;
        for (const auto & [body, recipients] : witnesses) {
            for (const auto & [family, frame] : recipients) {
                auto binding = nlohmann::json::array({child_contracts(frame->definition)});
                for (const auto & child : frame->children) binding.push_back(child.document());
                variations[{family, std::move(binding)}].emplace(body, frame);
            }
        }
        for (const auto & [source, alternatives] : variations) {
            if (alternatives.size() < 2) continue;
            for (const auto & [shared, original] : alternatives) {
                for (const auto & [family, recipient] : witnesses.at(shared)) {
                    if (family == source.first) continue;
                    for (const auto & [body, variant] : alternatives) {
                        if (body != shared) index_.families_[family].push_back({recipient, variant->definition, &variant->guard});
                    }
                }
            }
        }
    }

    void foreign_bindings(const TemplateFrame & frame, std::string_view owner, const std::string & position,
                          const std::map<std::string, Expression> & arguments, std::set<nlohmann::json> & shared) const {
        const auto & definition = library_.definitions[frame.definition];
        for (std::size_t port = 0; port < frame.tensors.size(); ++port) {
            const auto & tensor = frame.tensors[port];
            if (!tensor || (tensor->value.starts_with(owner) && tensor->storage.starts_with(owner))) continue;
            const auto & boundary = definition.ports[port];
            auto meaning = boundary.document();
            meaning.erase("event");
            meaning["tensor"]["shape"] = nlohmann::json::array();
            for (const auto & dimension : boundary.tensor->shape) meaning["tensor"]["shape"].push_back(dimension.substitute(arguments).document());

            // A replacement cannot change another scope's aliases, versions, writers or data access contract.
            shared.insert(nlohmann::json{
                { "position",                                                                   position },
                {     "port",                                                         std::move(meaning) },
                {    "value",                                                              tensor->value },
                {  "storage",                                                            tensor->storage },
                {  "version",                                                            tensor->version },
                { "external",                                                           tensor->external },
                {   "buffer",                                                             tensor->buffer },
                {   "writer", tensor->writer ? nlohmann::json(*tensor->writer) : nlohmann::json(nullptr) }
            });
        }
        for (std::size_t child = 0; child < frame.children.size(); ++child) {
            std::map<std::string, Expression> parameters;
            for (const auto & [name, value] : frame.children[child].constants) parameters.emplace(name, Expression::read(value));
            for (const auto & [name, expression] : definition.children[child].arguments) parameters.emplace(name, expression.substitute(arguments));
            foreign_bindings(frame.children[child], owner, position + std::to_string(child) + "/", parameters, shared);
        }
    }

    TemplateSelection & index_;
    const TemplateLibrary & library_;
    std::map<nlohmann::json, std::vector<const LocalImplementation *>> implementations_;
    std::map<std::tuple<std::string, nlohmann::json, std::string, nlohmann::json>, Callers> sites_;
    std::map<std::size_t, nlohmann::json> binding_contracts_;
};

class TemplateSelection::FrameSelection {
public:
    FrameSelection(const TemplateSelection & index, const std::map<std::string, StateBindings> & state_inputs)
        : index_(index), state_inputs_(state_inputs) {}

    FrameChoice resolve(std::span<const Candidate> candidates, const Parameters & arguments) const {
        FrameChoice result;
        const auto & first = *candidates.front().bindings;
        Parameters parameters = first.constants;
        for (const auto & [name, value] : arguments) {
            const auto [previous, inserted] = parameters.emplace(name, value);
            if (!inserted && previous->second != value) {
                result.rejected_parameters = nlohmann::json{
                    { "site", first.site }, { "fixed", name }, { "expected", previous->second }, { "actual", value }
                };
                return result;
            }
        }

        // Family construction fixes the interface and constants; only guards and child choices vary at runtime.
        const auto & expected = index_.library_.definitions[first.definition].parameters;
        if (expected.size() != parameters.size() || !std::ranges::all_of(expected, [&](const auto & name) { return parameters.contains(name); })) {
            std::vector<std::string> missing, unexpected;
            for (const auto & name : expected) {
                if (!parameters.contains(name)) missing.push_back(name);
            }
            for (const auto & [name, value] : parameters) {
                if (!expected.contains(name)) unexpected.push_back(name);
            }

            // Report the contract failure separately from guards on other, incompatible candidates.
            result.rejected_parameters = nlohmann::json{
                { "site", first.site }, { "missing", missing }, { "unexpected", unexpected }
            };
            return result;
        }

        const Candidate * rejected = nullptr;
        std::map<std::pair<std::size_t, std::vector<std::size_t>>, const Candidate *> implementations;
        for (const auto & candidate : candidates) {
            if (candidate.guard->evaluate(parameters) == 0) {
                if (!rejected) rejected = &candidate;
                continue;
            }
            const auto & bindings = *candidate.bindings;
            if (bindings.state_input) {
                const auto input = state_inputs_.find(*bindings.state_input);
                if (input == state_inputs_.end() || input->second.contract() != bindings.state.contract()) {
                    result.rejected_parameters = nlohmann::json{{ "site", bindings.site }, { "state_input", *bindings.state_input },
                                                               { "reason", "missing or incompatible state binding" }};
                    continue;
                }
            }
            std::vector<std::size_t> children;
            for (const auto & child : bindings.children) children.push_back(index_.membership_.at(&child));
            implementations.try_emplace(std::pair{ candidate.definition, std::move(children) }, &candidate);
        }

        if (implementations.empty() && rejected) {
            Parameters relevant;
            for (const auto & name : rejected->guard->variables()) relevant.emplace(name, parameters.at(name));

            // Keep one actual rejection, not an invented explanation for every candidate in the family.
            result.rejected_guard = nlohmann::json{
                { "site", rejected->bindings->site }, { "definition", rejected->definition },
                { "guard", rejected->guard->document() }, { "parameters", relevant }
            };
        }
        for (const auto & [choice, candidate] : implementations) {
            const auto & [identity, child_families] = choice;
            const auto & source = *candidate->bindings;
            const auto & body = index_.library_.definitions[identity];

            // Keep this observed caller body; only compatible child implementations are independently selected.
            FrameChoice chosen{
                TemplateFrame{ identity,
                              *candidate->guard,
                              source.site,
                              source.bindings,
                              source.constants,
                              source.resources,
                              source.tensors,
                              {},
                              source.state,
                              source.state_input,
                              source.communication },
                {},
                {},
                {}
            };
            for (std::size_t child = 0; child < body.children.size(); ++child) {
                const auto & call = body.children[child];
                Parameters child_arguments;
                for (const auto & [name, expression] : call.arguments) child_arguments.emplace(name, expression.evaluate(parameters));
                auto selected = resolve(index_.families_[child_families[child]], child_arguments);
                if (!selected.frame && !selected.ambiguity) {
                    chosen = std::move(selected);
                    break;
                }
                if (selected.ambiguity) chosen.ambiguity = std::move(selected.ambiguity);
                else chosen.frame->children.push_back(std::move(*selected.frame));
            }
            if (chosen.ambiguity) chosen.frame.reset();
            result.merge(std::move(chosen), source.site);
        }
        return result;
    }

private:
    const TemplateSelection & index_;
    const std::map<std::string, StateBindings> & state_inputs_;
};

TemplateGap::TemplateGap(nlohmann::json details)
    : std::invalid_argument("template selection failed: " + details.dump()), details_(std::move(details)) {}

TemplateSelection::TemplateSelection(const TemplateLibrary & library) : library_(library) {
    FamilyIndex(*this).build();
}

TemplateEntry TemplateSelection::select(std::string_view action, const nlohmann::json & condition, const Parameters & parameters,
                                       const std::map<std::string, StateBindings> & state_inputs) const {
    const auto roots = roots_.find({ std::string(action), condition });
    FrameChoice chosen;
    if (roots != roots_.end()) {
        FrameSelection selection(*this, state_inputs);
        for (const auto family : roots->second) chosen.merge(selection.resolve(families_[family], parameters), action);
    }
    if (!chosen.frame) {
        const auto status = roots == roots_.end() ? "no_declared_candidates"
            : chosen.ambiguity ? "ambiguous_implementation" : "unmatched_conditions";
        nlohmann::json details{
            { "status", status }, { "action", action }, { "condition", condition }, { "parameters", parameters }, { "domain", library_.domain }
        };
        if (chosen.ambiguity) details["site"] = *chosen.ambiguity;
        if (chosen.rejected_parameters) details["rejected_parameter_contract"] = std::move(*chosen.rejected_parameters);
        if (chosen.rejected_guard) details["rejected_local_candidate"] = std::move(*chosen.rejected_guard);
        throw TemplateGap(std::move(details));
    }

    // A composed target is not an observed whole call; source observations remain on the stored candidates.
    return { std::string(action), condition, std::move(*chosen.frame), {} };
}

} // namespace prologue
