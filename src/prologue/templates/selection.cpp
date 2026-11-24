#include "prologue/templates/selection.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
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

enum class Rejection { Parameters, Guard, State };

struct QualifiedCaller {
    std::size_t definition;
    const TemplateFrame * frame;
    Parameters parameters;
};

struct RejectedCall {
    std::vector<std::size_t> path;
    const TemplateFrame * frame;
    const TemplateDefinition * body;
    const TemplateCall * caller;
    Rejection cause;
    Parameters arguments;
    nlohmann::json detail;
    bool competing = false;
    std::vector<QualifiedCaller> callers{};

    void merge(RejectedCall other) {
        if (path == other.path) {
            if (competing || other.competing) competing = true;
            else if (cause != other.cause) {
                if (other.cause > cause) *this = std::move(other);
            }
            else competing = frame->site != other.frame->site || body->implementation != other.body->implementation || arguments != other.arguments
                          || (caller && other.caller && caller->contract != other.caller->contract);
            return;
        }
        const auto common = std::ranges::mismatch(path, other.path);
        if (common.in1 == path.end()) {
            if (!competing) *this = std::move(other);
            return;
        }
        if (common.in2 == other.path.end()) {
            if (other.competing) *this = std::move(other);
            return;
        }

        // Different children of qualified callers need different evidence.
        path.erase(common.in1, path.end());
        competing = true;
    }

    nlohmann::json document() const {
        if (competing) return {{"path", path}, {"cause", "competing_calls"}};
        auto cut = detail;
        cut["path"] = path;
        cut["site"] = frame->site;
        cut["role"] = body->role;
        cut["implementation"] = body->implementation;
        cut["parameters"] = arguments;
        cut["cause"] = cause == Rejection::Guard ? "guard" : cause == Rejection::State ? "state" : "parameters";
        if (caller) cut["interface"] = caller->contract.document();
        return cut;
    }
};

struct FrameChoice {
    std::optional<TemplateFrame> frame;
    std::optional<std::string> ambiguity;
    std::optional<RejectedCall> rejection;

    void merge(FrameChoice candidate, std::string_view site) {
        if (candidate.rejection) {
            if (rejection) rejection->merge(std::move(*candidate.rejection));
            else rejection = std::move(candidate.rejection);
        }
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

bool locate(const TemplateFrame & frame, const TemplateFrame * wanted, std::string & address) {
    if (&frame == wanted) return true;
    const auto base = address.size();
    for (std::size_t child = 0; child < frame.children.size(); ++child) {
        address += "/children/" + std::to_string(child);
        if (locate(frame.children[child], wanted, address)) return true;
        address.resize(base);
    }
    for (std::size_t variant = 0; variant < frame.alternatives.size(); ++variant) {
        const auto & branch = frame.alternatives[variant].frame;
        if (&branch == wanted) return true;
        address += "/alternatives/" + std::to_string(variant) + "/frame";
        if (locate(branch, wanted, address)) return true;
        address.resize(base);
    }
    return false;
}

std::string binding_address(const TemplateLibrary & library, const TemplateFrame * frame) {
    for (std::size_t entry = 0; entry < library.entries.size(); ++entry) {
        auto address = "/entries/" + std::to_string(entry) + "/frame";
        if (locate(library.entries[entry].frame, frame, address)) return address;
    }
    throw std::logic_error("selected binding has no source call");
}

} // namespace

class TemplateSelection::FrameSelection {
public:
    FrameSelection(const TemplateSelection & index, const std::map<std::string, StateBindings> & state_inputs)
        : index_(index), state_inputs_(state_inputs) {}

    FrameChoice resolve(std::span<const Candidate> candidates, const Parameters & arguments, std::span<const nlohmann::json> members,
                        const TemplateCall * caller = nullptr) {
        FrameChoice result;
        const auto & first = *candidates.front().bindings;
        const auto & declaration = index_.library_.definitions[first.definition];
        auto input = declaration.bind(arguments, first.constants);
        if (!input) {
            result.rejection = RejectedCall{path_, &first, &declaration, caller, Rejection::Parameters, arguments, std::move(input.error())};
            return result;
        }
        const auto & parameters = *input;

        const Candidate * rejected = nullptr;
        std::map<std::pair<std::size_t, std::vector<std::size_t>>, const Candidate *> implementations;
        for (const auto & candidate : candidates) {
            if (candidate.guard->evaluate(parameters) == 0) {
                if (!rejected) rejected = &candidate;
                continue;
            }
            const auto & bindings = *candidate.bindings;
            if (!bindings.accepts_state(state_inputs_)) {
                result.rejection = RejectedCall{path_, &bindings, &index_.library_.definitions[candidate.definition], caller,
                                                Rejection::State, parameters, {{"state_input", *bindings.state_input}}};
                continue;
            }
            std::vector<std::size_t> children;
            for (const auto & child : bindings.children) children.push_back(index_.membership_.at(&child));
            implementations.try_emplace(std::pair{ candidate.definition, std::move(children) }, &candidate);
        }

        if (implementations.empty() && !result.rejection && rejected) {
            result.rejection = RejectedCall{path_, rejected->bindings, &index_.library_.definitions[rejected->definition], caller,
                                            Rejection::Guard, parameters, {{"guard", rejected->guard->document()}}};
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
                {}
            };
            for (std::size_t child = 0; child < body.children.size(); ++child) {
                const auto & call = body.children[child];
                if (call.row && call.row->member >= members.size()) {
                    chosen = {};
                    chosen.rejection = RejectedCall{path_, &source, &body, caller, Rejection::Parameters, parameters,
                                                    {{"row_call", call.row->document()}, {"admitted", members.size()}}};
                    break;
                }
                const auto bound = call.bind(parameters, members);
                path_.push_back(child);
                auto selected = resolve(index_.families_[child_families[child]], bound.parameters, bound.members, &call);
                path_.pop_back();
                if (!selected.frame && !selected.ambiguity) {
                    chosen = std::move(selected);
                    if (chosen.rejection) chosen.rejection->callers.push_back({identity, &source, parameters});
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
    std::vector<std::size_t> path_;
};

TemplateGap::TemplateGap(nlohmann::json details)
    : std::invalid_argument("template selection failed: " + details.dump()), details_(std::move(details)) {}

void TemplateSelection::validate_branches(const TemplateFrame & frame) const {
    for (const auto & branch : frame.alternatives) {
        const auto states = branch.frame.state_inputs();
        const auto work = branch.input.work();
        const auto chosen = FrameSelection(*this, states).resolve(families_[membership_.at(&frame)], branch.input.parameters, admitted_members(work));
        if (!chosen.frame || !same_realization(*chosen.frame, branch.frame))
            throw std::invalid_argument("measured local input must select its observed branch uniquely at " + frame.site);
        validate_branches(branch.frame);
    }
    for (const auto & child : frame.children) validate_branches(child);
}

TemplateEntry TemplateSelection::select(std::string_view action, const nlohmann::json & condition, const Parameters & parameters,
                                       const std::map<std::string, StateBindings> & state_inputs, const nlohmann::json & work) const {
    const auto roots = roots_.find({ std::string(action), condition });
    FrameChoice chosen;
    if (roots != roots_.end()) {
        FrameSelection selection(*this, state_inputs);
        const auto members = admitted_members(work);
        for (const auto family : roots->second.observed) chosen.merge(selection.resolve(families_[family], parameters, members), action);

        // Closed imports extend coverage without changing the interpretation of an applicable source observation.
        if (!chosen.frame && !chosen.ambiguity) {
            for (const auto family : roots->second.imported) chosen.merge(selection.resolve(families_[family], parameters, members), action);
        }
    }
    if (!chosen.frame) {
        const auto status = roots == roots_.end() ? "no_declared_candidates"
            : chosen.ambiguity ? "ambiguous_implementation" : "unmatched_conditions";
        nlohmann::json details{
            { "status", status }, { "action", action }, { "condition", condition }, { "parameters", parameters }, { "domain", library_.domain }
        };

        // A local observation needs each admitted member's extent; aggregate dimensions cannot recover their order.
        if (work.contains("members")) details["members"] = work.at("members");
        for (const auto & [name, state] : state_inputs) details["state_inputs"][name] = state.document();
        if (chosen.ambiguity) details["site"] = *chosen.ambiguity;
        else if (chosen.rejection) {
            const auto & rejection = *chosen.rejection;
            auto & cut = details["cut"] = rejection.document();
            if (!rejection.competing) {
                cut["binding"] = binding_address(library_, rejection.frame);
                auto & callers = cut["callers"] = nlohmann::json::array();
                for (auto i = rejection.callers.rbegin(); i != rejection.callers.rend(); ++i)
                    callers.push_back({{"definition", i->definition}, {"parameters", i->parameters}, {"binding", binding_address(library_, i->frame)}});
            }
        }
        throw TemplateGap(std::move(details));
    }

    // A composed target is not an observed whole call; source observations remain on the stored candidates.
    return { std::string(action), condition, std::move(*chosen.frame), {} };
}

nlohmann::json TemplateSelection::query(const nlohmann::json & demand) const {
    if (demand.at("domain") != library_.domain) throw std::invalid_argument("query belongs to a different template domain");
    const auto action = demand.at("action").get<std::string>();
    const auto & condition = demand.at("condition");
    const auto parameters = read_parameters(demand.at("parameters"));
    std::map<std::string, StateBindings> states;
    if (demand.contains("state_inputs")) {
        for (const auto & [name, state] : demand.at("state_inputs").items()) states.emplace(name, StateBindings::read(state));
    }
    try {
        const auto selected = select(action, condition, parameters, states, demand);
        const auto bound = selected.frame.bind(library_.definitions.at(selected.frame.definition), parameters);
        const auto members = admitted_members(demand);
        for (std::size_t id = 0; id < library_.entries.size(); ++id) {
            const auto & entry = library_.entries[id];
            if (entry.action != action || entry.condition != condition) continue;
            const auto point = std::ranges::find_if(entry.observations, [&](const auto & sample) { return sample.matches(bound, members); });
            if (point != entry.observations.end() && same_realization(entry.frame, selected.frame))
                return {{"status", "covered"}, {"observation", id}, {"input", point->document()}};
        }

        // A predictable point may have no direct native observation to donate.
        return {{"status", "covered"}};
    }
    catch (const TemplateGap & gap) {
        return gap.details();
    }
}

} // namespace prologue
