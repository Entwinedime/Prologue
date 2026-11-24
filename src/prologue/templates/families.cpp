#include "prologue/templates/selection.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace prologue {

class TemplateSelection::FamilyIndex {
    using Callers = std::map<std::optional<std::size_t>, std::size_t>;
    using Body = std::pair<std::size_t, nlohmann::json>;

public:
    explicit FamilyIndex(TemplateSelection & index) : index_(index), library_(index.library_) {}

    void build() {
        for (const auto & implementation : library_.implementations) implementations_[implementation.contract].push_back(&implementation);
        for (const auto & entry : library_.entries) {
            auto & roots = index_.roots_[{ entry.action, entry.condition }];
            auto & families = entry.observations.empty() ? roots.imported : roots.observed;
            families.insert(add(entry, entry.frame));
        }
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
        foreign_bindings(frame, space, space, arguments, {}, shared);
        local["shared_bindings"] = std::move(shared);
        if (caller) {
            const auto & call = library_.definitions[*caller].children[position.back()];
            for (const auto & [name, expression] : call.arguments) local["arguments"][name] = expression.document();
            if (call.row) local["row_call"] = call.row->document();
        }

        // An imported root has no observations in this library and cannot supply bodies to observed callers.
        auto & callers = sites_[{ entry.action, entry.condition, space, std::move(local), entry.observations.empty() }];
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
        for (const auto & branch : frame.alternatives) {
            if (!caller || add(entry, branch.frame, space, caller, position) != found->second)
                throw std::invalid_argument("measured branch changes its caller-visible bindings at " + frame.site);
        }
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
                          const std::map<std::string, Expression> & arguments, const std::vector<nlohmann::json> & rows,
                          std::set<nlohmann::json> & shared) const {
        const auto & definition = library_.definitions[frame.definition];
        for (const auto & storage : frame.allocations) {
            if (!storage.starts_with(owner)) shared.insert({{"position", position}, {"allocates", storage}});
        }
        for (std::size_t port = 0; port < frame.tensors.size(); ++port) {
            const auto & tensor = frame.tensors[port];
            if (!tensor || (tensor->value.starts_with(owner) && tensor->storage.starts_with(owner))) continue;
            const auto & boundary = definition.ports[port];
            auto meaning = boundary.operand(arguments);
            if (!rows.empty()) meaning["row_path"] = rows;

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
            const auto & call = definition.children[child];
            for (const auto & [name, expression] : call.arguments)
                parameters.emplace(name, call.row ? expression : expression.substitute(arguments));
            auto path = rows;
            if (call.row) path.push_back(call.row->document());
            foreign_bindings(frame.children[child], owner, position + std::to_string(child) + "/", parameters, path, shared);
        }
    }

    TemplateSelection & index_;
    const TemplateLibrary & library_;
    std::map<nlohmann::json, std::vector<const LocalImplementation *>> implementations_;
    std::map<std::tuple<std::string, nlohmann::json, std::string, nlohmann::json, bool>, Callers> sites_;
    std::map<std::size_t, nlohmann::json> binding_contracts_;
};

TemplateSelection::TemplateSelection(const TemplateLibrary & library) : library_(library) {
    FamilyIndex(*this).build();
}

} // namespace prologue
