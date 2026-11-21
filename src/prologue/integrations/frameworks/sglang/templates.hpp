#pragma once

#include <cstddef>
#include <map>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "prologue/graph/graph.hpp"
#include "prologue/templates/production.hpp"
#include "prologue/templates/library.hpp"

namespace prologue::sglang {

struct TemplateDeclarations {
    CoarseGraph graph;
    std::vector<EntrySpecification> entries;
};

// Native facts select execution classes; capture and control share the same template contract.
class NativeTemplates {
public:
    static TemplateDeclarations describe(CoarseGraph graph);
    static TemplateLibrary compile(CoarseGraph graph, std::vector<TemplateInterval> * origins = nullptr);
    static nlohmann::json condition(std::string_view action, nlohmann::json facts);
    static nlohmann::json domain(const nlohmann::json & environment);
    static nlohmann::json share_closed_actions(std::map<std::size_t, TemplateLibrary> & libraries);
};

} // namespace prologue::sglang
