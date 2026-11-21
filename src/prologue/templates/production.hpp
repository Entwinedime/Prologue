#pragma once

#include <span>
#include <vector>

#include "prologue/graph/graph.hpp"
#include "prologue/templates/library.hpp"

namespace prologue {

struct EntrySpecification {
    std::size_t scope;
    std::string action;
    nlohmann::json condition;
};

// When requested, return one template origin per work interval in observed graph order.
class TemplateCompiler {
public:
    static TemplateLibrary compile(const CoarseGraph & source, std::span<const EntrySpecification> entries, nlohmann::json domain,
                                   std::vector<TemplateInterval> * origins = nullptr);
};

} // namespace prologue
