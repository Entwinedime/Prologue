#pragma once

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "prologue/templates/library.hpp"

namespace prologue {

class TemplateGap final : public std::invalid_argument {
public:
    explicit TemplateGap(nlohmann::json details);
    const nlohmann::json & details() const { return details_; }

private:
    nlohmann::json details_;
};

// The immutable library outlives this index; dynamic target choices never modify its source candidates.
class TemplateSelection {
public:
    explicit TemplateSelection(const TemplateLibrary & library);
    TemplateSelection(const TemplateSelection &) = delete;
    TemplateSelection & operator=(const TemplateSelection &) = delete;
    TemplateEntry select(std::string_view action, const nlohmann::json & condition, const Parameters & parameters,
                         const std::map<std::string, StateBindings> & state_inputs = {}, const nlohmann::json & work = nullptr) const;
    nlohmann::json query(const nlohmann::json & demand) const;
    void validate_branches(const TemplateFrame & frame) const;

private:
    struct Candidate {
        const TemplateFrame * bindings;
        std::size_t definition;
        const TemplateGuard * guard;
    };
    using Family = std::vector<Candidate>;

    struct ActionFamilies {
        std::set<std::size_t> observed;
        std::set<std::size_t> imported;
    };

    class FamilyIndex;
    class FrameSelection;

    const TemplateLibrary & library_;
    std::vector<Family> families_;
    std::map<const TemplateFrame *, std::size_t> membership_;
    std::map<std::pair<std::string, nlohmann::json>, ActionFamilies> roots_;
};

} // namespace prologue
