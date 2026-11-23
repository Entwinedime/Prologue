#include "prologue/core/structure/templates/library.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace prologue::structure {
namespace {

using nlohmann::json;
using Index = std::size_t;

class Expansion final {
public:
    explicit Expansion(const json& definitions) : definitions_(definitions) {}

    json build(Index root) {
        visit(root);
        return {{"calls", std::move(calls_)}, {"work", std::move(work_)}};
    }

private:
    void visit(Index identity) {
        const auto& definition = definitions_.at(identity);
        auto call = definition;
        call.erase("children");
        call.erase("device_work");
        call["path"] = path_;
        calls_.push_back(std::move(call));

        const auto& work = definition.at("device_work");
        const auto& children = definition.at("children");
        Index next_work = 0;
        for (Index slot = 0; slot <= children.size(); ++slot) {
            while (next_work < work.size() && work.at(next_work).at("slot") == slot) {
                auto entry = work.at(next_work);
                entry["path"] = path_;
                entry["ordinal"] = next_work++;
                work_.push_back(std::move(entry));
            }
            if (slot < children.size()) {
                const auto child = children.at(slot).get<Index>();
                if (child >= identity) {
                    throw std::invalid_argument("template child must precede its parent definition");
                }
                path_.push_back(slot);
                visit(child);
                path_.pop_back();
            }
        }
        if (next_work != work.size()) {
            throw std::invalid_argument("template work is outside its child slots");
        }
    }

    const json& definitions_;
    std::vector<Index> path_;
    json calls_ = json::array();
    json work_ = json::array();
};

}

json batch_geometry(const json& phase, const json& members) {
    auto sequence = json::array();
    for (const auto& member : members) {
        sequence.push_back(member.at("kv_committed_tokens"));
    }
    return {{"phase", phase}, {"sequence_lengths", std::move(sequence)}};
}

Templates::Templates(const json& libraries, const json& rule) : libraries_(libraries) {
    if (rule.at("selector") != "exact_batch_geometry") {
        throw std::invalid_argument("unsupported template selector");
    }
    for (Index source = 0; source < libraries_.size(); ++source) {
        const auto& library = libraries_.at(source);
        for (const auto& instance : library.at("instances")) {
            if (!instance.at("parent").is_null()) {
                continue;
            }
            const auto key = batch_geometry(instance.at("parameters").at("phase"), instance.at("members")).dump();
            batches_[key].push_back({source, instance.at("definition").get<Index>()});
        }
    }
}

std::optional<json> Templates::expand(const json& action) const {
    const auto key = batch_geometry(action.at("phase"), action.at("members")).dump();
    const auto candidate = batches_.find(key);
    if (candidate == batches_.end()) {
        return std::nullopt;
    }
    return select(key, candidate->second);
}

json Templates::select(const std::string& geometry, const std::vector<Candidate>& candidates) const {
    json graph;
    for (const auto& match : candidates) {
        auto expanded = Expansion(libraries_.at(match.library).at("definitions")).build(match.definition);
        if (!graph.is_null() && graph != expanded) {
            throw std::invalid_argument("source templates disagree at batch geometry " + geometry);
        }
        graph = std::move(expanded);
    }
    return graph;
}

void Templates::qualify() const {
    if (batches_.empty()) {
        throw std::invalid_argument("template sources have no root candidates");
    }
    for (const auto& [geometry, candidates] : batches_) {
        select(geometry, candidates);
    }
}

}
