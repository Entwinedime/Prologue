#pragma once

#include <span>
#include <vector>

#include "core/graph/graph.hpp"
#include "core/templates/library.hpp"

namespace prologue {

struct EntrySpecification {
    std::size_t scope;
    std::string action;
    nlohmann::json condition;
};

struct TemplateDeclarations {
    CoarseGraph graph;
    std::vector<EntrySpecification> entries;
};

struct ObservedTemplateWork {
    nlohmann::json domain;
    std::vector<TemplateDefinition> definitions;
    std::vector<TemplateInterval> origins;
};

class TemplateCompiler {
public:
    static TemplateLibrary extract(const CoarseGraph & source, std::span<const EntrySpecification> entries, nlohmann::json domain,
                                   std::vector<TemplateInterval> * origins = nullptr);
};

} // namespace prologue
