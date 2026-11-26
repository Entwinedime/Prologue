#include "core/templates/selection.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openssl/sha.h>

namespace prologue {
namespace {

using Json = nlohmann::json;

std::optional<std::size_t> matching_resource(const Resource & lane, std::span<const Resource> observed,
                                             std::span<const Resource> known) {
    const auto equivalent = [&](const Resource & other) {
        return other.kind == lane.kind && other.rank == lane.rank && other.role == lane.role && other.serial == lane.serial
            && other.binding == lane.binding;
    };

    // A declared logical lane survives process-local stream numbers. Unnamed lanes still need an unambiguous match.
    if (!lane.binding) {
        auto exact = std::ranges::find(known, lane);
        if (exact != known.end()) return static_cast<std::size_t>(exact - known.begin());
    }
    auto found = std::ranges::find_if(known, equivalent);
    if (found == known.end()) return std::nullopt;
    if (std::ranges::count_if(known, equivalent) != 1 || std::ranges::count_if(observed, equivalent) != 1)
        throw TemplateGap({{"status", "unbound_evidence_resource"}, {"resource", lane.document()}});
    return static_cast<std::size_t>(found - known.begin());
}

void relocate(TemplateFrame & frame, const std::vector<std::size_t> & definitions, const std::vector<std::size_t> & lanes) {
    frame.definition = definitions.at(frame.definition);
    for (auto & lane : frame.resources) lane = lanes.at(lane);
    for (auto & child : frame.children) relocate(child, definitions, lanes);
    for (auto & branch : frame.alternatives) relocate(branch.frame, definitions, lanes);
}

std::string body_key(const Json & body) {
    const auto bytes = body.dump();
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
    if (!SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), digest.data()))
        throw std::runtime_error("cannot identify normalized template definition");

    constexpr std::string_view hex = "0123456789abcdef";
    std::string key;
    key.reserve(2 * digest.size());
    for (const auto byte : digest) {
        key += hex[byte >> 4];
        key += hex[byte & 15];
    }
    return key;
}

struct SourceCall {
    const TemplateFrame * frame;
    Parameters parameters;
    std::span<const Json> members;

    SourceCall child(const TemplateLibrary & library, std::size_t position) const {
        auto bound = library.definitions.at(frame->definition).children.at(position).bind(parameters, members);
        const auto & callee = frame->children.at(position);
        return {&callee, callee.bind(library.definitions.at(callee.definition), std::move(bound.parameters)), bound.members};
    }
};

bool same_program(const TemplateFrame & first, const TemplateFrame & second) {
    return first.definition == second.definition && first.guard.document() == second.guard.document()
        && std::ranges::equal(first.children, second.children, same_program);
}

class MeasuredCalls {
public:
    MeasuredCalls(const TemplateLibrary & source, const TemplateLibrary & target) : source_(source) {
        if (source.domain != target.domain || source.local_domain != target.local_domain)
            throw std::invalid_argument("measured call belongs to another template domain");
        for (const auto & body : target.definitions) known_.insert(body_key(body.document()));
        result_ = {{"domain", source.domain}, {"local_domain", source.local_domain}, {"resources", Json::array()},
                   {"definitions", Json::object()}, {"calls", Json::array()}};
        for (const auto & resource : source.resources) result_["resources"].push_back(resource.document());
    }

    Json extract(const Json & measured) {
        const auto answer = TemplateSelection(source_).query(measured);
        if (!answer.contains("observation")) throw TemplateGap({{"status", "no_direct_observation"}, {"action", measured.at("action")},
                                                                 {"parameters", measured.at("parameters")}});
        const auto & entry = source_.entries.at(answer.at("observation").get<std::size_t>());
        result_["context"] = {{"action", entry.action}, {"condition", entry.condition}};
        SourceCall parent{&entry.frame, ObservedArguments::read(answer.at("input")).parameters, admitted_members(measured)};
        const auto path = measured.value("path", std::vector<std::size_t>{});
        result_["root"] = path.empty();
        if (path.empty()) {
            result_["interface"] = source_.call_interface(*parent.frame).document();
            append(parent, 0);
        }
        else {
            for (std::size_t depth = 0; depth + 1 < path.size(); ++depth) parent = parent.child(source_, path[depth]);

            const auto position = path.back();
            const auto call = parent.child(source_, position);
            const auto interface = source_.call_interface(*call.frame);
            result_["interface"] = interface.document();
            append(call, position);
            for (std::size_t i = 0; i < parent.frame->children.size(); ++i) {
                const auto & sibling = parent.frame->children[i];
                if (i != position && same_program(*call.frame, sibling) && source_.call_interface(sibling) == interface)
                    append(parent.child(source_, i), i);
            }
        }
        return std::move(result_);
    }

private:
    Json binding(const TemplateFrame & frame, bool & uses_members) {
        auto local = frame.local_document();
        const auto [it, added] = keys_.try_emplace(frame.definition);
        const auto & body = source_.definitions.at(frame.definition);
        if (added) {
            auto document = body.document();
            it->second = body_key(document);

            // Known children travel as content references. Their internal work is absent from the parent's evidence.
            if (!known_.contains(it->second)) result_["definitions"][it->second] = std::move(document);
        }
        local["definition"] = it->second;
        if (std::ranges::any_of(body.children, [](const auto & child) { return bool(child.row); })) uses_members = true;
        auto & children = local["children"] = Json::array();
        for (const auto & child : frame.children) children.push_back(binding(child, uses_members));
        return local;
    }

    void append(const SourceCall & call, std::size_t position) {
        bool uses_members = false;
        auto frame = binding(*call.frame, uses_members);
        auto input = ObservedArguments::capture(call.parameters, uses_members ? call.members : std::span<const Json>{});
        result_["calls"].push_back({{"position", position}, {"frame", std::move(frame)}, {"input", input.document()}});
    }

    const TemplateLibrary & source_;
    Json result_;
    std::set<std::string> known_;
    std::map<std::size_t, std::string> keys_;
};

class CandidateBindings {
public:
    CandidateBindings(const TemplateLibrary & target, const Json & evidence)
        : target_(target), evidence_(evidence), candidate_(target.document()) {
        if (target.domain != evidence.at("domain") || target.local_domain != evidence.at("local_domain"))
            throw std::invalid_argument("evidence belongs to a different template domain");
        for (std::size_t i = 0; i < target.definitions.size(); ++i) known_.emplace(body_key(candidate_["definitions"][i]), i);
        for (const auto & lane : evidence.at("resources")) observed_resources_.push_back(Resource::read(lane));
    }

    void apply(const Json & demand, const Json & gap) {
        const auto & context = evidence_.at("context");
        if (context.at("action") != demand.at("action") || context.at("condition") != demand.at("condition"))
            throw std::invalid_argument("measured call has a different action or structural condition");
        const auto & samples = evidence_.at("calls").get_ref<const Json::array_t &>();
        const auto & first = samples.at(0);
        const auto cut = gap.value("cut", Json::object());
        const auto original = cut.value("path", std::vector<std::size_t>{});
        auto path = original;
        if (demand.value("status", "") == "caller_evidence_required") {
            path = demand.at("caller_path").get<std::vector<std::size_t>>();
            if (path.size() >= original.size() || !std::ranges::equal(path, std::span(original).first(path.size())))
                throw std::invalid_argument("requested caller does not enclose the current template gap");
        }

        auto members = admitted_members(demand);
        if (path.empty()) {
            if (!evidence_.at("root").get<bool>() || samples.size() != 1)
                throw std::invalid_argument("the requested action requires one directly measured root");
            candidate_["entries"].push_back({{"action", context.at("action")}, {"condition", context.at("condition")},
                                             {"frame", bind(first.at("frame"))}, {"observations", Json::array({first.at("input")})}});
            return;
        }

        const auto & callers = cut.at("callers");
        for (std::size_t depth = 0; depth + 1 < path.size(); ++depth) {
            const auto & parent = callers.at(depth);
            const auto & call = target_.definitions.at(parent.at("definition").get<std::size_t>()).children.at(path[depth]);
            members = call.bind(read_parameters(parent.at("parameters")), members).members;
        }
        const auto & parent = callers.at(path.size() - 1);
        const auto parameters = read_parameters(parent.at("parameters"));
        const auto & body = target_.definitions.at(parent.at("definition").get<std::size_t>());
        const auto & site = path.size() == original.size() ? cut : callers.at(path.size());
        const Json::json_pointer address(site.at("binding").get<std::string>());
        if (address.back() != std::to_string(path.back()) || address.parent_pointer().back() != "children")
            throw std::invalid_argument("recipient binding differs from the requested child position");
        auto & siblings = candidate_.at(address.parent_pointer());
        const auto original_body = siblings.at(path.back()).at("definition").get<std::size_t>();

        // Only the first cut is required. Additional measured occurrences can use it at their own bound coordinates.
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const auto & sample = samples[i];
            const auto pos = i == 0 ? path.back() : sample.at("position").get<std::size_t>();
            if (pos >= siblings.size() || (i && pos == path.back())) continue;
            auto & slot = siblings.at(pos);
            const auto id = slot.at("definition").get<std::size_t>();
            if (i && id != original_body) continue;
            auto bound = body.children.at(pos).bind(parameters, members);
            const auto arguments = target_.definitions.at(id).bind(std::move(bound.parameters), read_parameters(slot.at("constants")));

            if (i && (!arguments || TemplateGuard::read(slot.at("guard")).evaluate(*arguments))) continue;
            if (!arguments || sample.at("frame").at("site") != slot.at("site")
                || !ObservedArguments::read(sample.at("input")).matches(*arguments, bound.members)) {
                if (i) continue;
                throw std::invalid_argument("measured child differs from the requested site or local inputs");
            }
            if (evidence_.at("interface") != body.children[pos].contract.document()) {
                if (i) continue;
                auto next = gap;
                next["status"] = "caller_evidence_required";
                next["caller_path"] = std::vector<std::size_t>(path.begin(), path.end() - 1);
                next["changed_site"] = slot.at("site");
                throw TemplateGap(std::move(next));
            }
            auto frame = bind(sample.at("frame"));
            auto & alternatives = slot["alternatives"];
            if (!std::ranges::any_of(alternatives, [&](const auto & item) { return item.at("frame") == frame; }))
                alternatives.push_back({{"frame", std::move(frame)}, {"input", sample.at("input")}});
        }
    }

    Json take() { return std::move(candidate_); }

private:
    std::size_t resource(std::size_t slot) {
        if (const auto it = resources_.find(slot); it != resources_.end()) return it->second;
        const auto & observed = observed_resources_.at(slot);
        const auto found = matching_resource(observed, observed_resources_, target_.resources);
        if (!found) throw TemplateGap({{"status", "unbound_evidence_resource"}, {"resource", observed.document()}});
        return resources_.emplace(slot, *found).first->second;
    }

    Json bind(const Json & observed) {
        if (observed.contains("alternatives")) throw std::invalid_argument("evidence must contain only the directly observed call");
        const auto key = observed.at("definition").get<std::string>();
        auto & definitions = candidate_.at("definitions");
        const auto [body, added] = known_.try_emplace(key, definitions.size());
        if (added) {
            const auto & incoming = evidence_.at("definitions");
            if (!incoming.contains(key)) throw TemplateGap({{"status", "missing_reused_definition"}, {"definition", key}});
            auto normalized = TemplateDefinition::read(incoming.at(key)).document();
            if (body_key(normalized) != key) throw std::invalid_argument("evidence definition differs from its content identity");
            definitions.push_back(std::move(normalized));
        }
        auto frame = observed;
        frame["definition"] = body->second;
        for (auto & item : frame.at("resources")) item = resource(item.get<std::size_t>());
        for (auto & child : frame.at("children")) child = bind(child);
        return frame;
    }

    const TemplateLibrary & target_;
    const Json & evidence_;
    Json candidate_;
    std::map<std::string, std::size_t> known_;
    std::map<std::size_t, std::size_t> resources_;
    std::vector<Resource> observed_resources_;
};

} // namespace

void TemplateLibrary::join_observation(const TemplateLibrary & observed) {
    if (domain != observed.domain || local_domain != observed.local_domain || resources.at(0).rank != observed.resources.at(0).rank)
        throw std::invalid_argument("local observations belong to different worker domains");

    // Idle initialization may only have run a control thread. New lanes come from this worker's actual call.
    const auto previous = resources.size();
    std::vector<std::size_t> lanes, bodies;
    for (const auto & lane : observed.resources) {
        const auto found = matching_resource(lane, observed.resources, std::span(resources).first(previous));
        lanes.push_back(found.value_or(resources.size()));
        if (!found) resources.push_back(lane);
    }
    std::map<std::string, std::size_t> known;
    for (std::size_t i = 0; i < definitions.size(); ++i) known.emplace(body_key(definitions[i].document()), i);
    for (const auto & definition : observed.definitions) {
        auto [body, inserted] = known.try_emplace(body_key(definition.document()), definitions.size());
        bodies.push_back(body->second);
        if (inserted) definitions.push_back(definition);
    }

    // A whole observation retains its own arguments; identical programs accumulate the measured coordinates.
    std::map<Json, std::size_t> calls;
    const auto key = [](const TemplateEntry & entry) { return Json::array({entry.action, entry.condition, entry.frame.document()}); };
    for (std::size_t i = 0; i < entries.size(); ++i) calls.emplace(key(entries[i]), i);
    for (auto entry : observed.entries) {
        if (entry.observations.empty()) continue;
        relocate(entry.frame, bodies, lanes);
        auto [call, inserted] = calls.try_emplace(key(entry), entries.size());
        if (inserted) entries.push_back(std::move(entry));
        else entries[call->second].observations.merge(entry.observations);
    }
    collect_local_implementations();
    validate();
}

Json TemplateLibrary::evidence(const Json & measured, const TemplateLibrary & recipient) const {
    return MeasuredCalls(*this, recipient).extract(measured);
}

TemplateLibrary TemplateLibrary::supplement(const Json & demand, const Json & evidence) const {
    const auto gap = TemplateSelection(*this).query(demand);
    const auto status = gap.at("status").get<std::string>();
    if (status == "covered") throw std::invalid_argument("the original query is already covered");
    const auto cut = gap.value("cut", Json::object());
    if (status != "no_declared_candidates" && (status != "unmatched_conditions" || cut.value("cause", "") != "guard"))
        throw TemplateGap(gap);

    CandidateBindings increment(*this, evidence);
    increment.apply(demand, gap);
    auto candidate = decode(increment.take());
    candidate.validate_observations();
    candidate.collect_local_implementations();
    candidate.validate_selection();

    // Another child may still need observation. The supplied cut itself must have become usable.
    const auto next = TemplateSelection(candidate).query(demand);
    if (evidence.at("root").get<bool>()
        && (!next.contains("observation") || next.at("input") != evidence.at("calls").at(0).at("input")))
        throw std::invalid_argument("measured parent differs from the requested action or inputs");
    if (next.at("status") != "covered" && (!next.contains("cut") || next.at("cut").value("cause", "") != "guard"
            || (!cut.empty() && next.at("cut").at("path") == cut.at("path"))))
        throw TemplateGap(next);
    return candidate;
}

} // namespace prologue
