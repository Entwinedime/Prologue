#pragma once

#include <cstddef>
#include <map>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/graph/graph.hpp"
#include "core/templates/production.hpp"
#include "core/templates/library.hpp"

namespace prologue::sglang {

// Native facts select execution classes; capture and control share the same template contract.
class NativeTemplates {
public:
    static TemplateDeclarations describe(CoarseGraph graph);
    static TemplateLibrary compile(const TemplateDeclarations & source);
    static ObservedTemplateWork observed_work(const TemplateDeclarations & source);
    static nlohmann::json condition(std::string_view action, nlohmann::json facts);
    static nlohmann::json domain(const nlohmann::json & environment);
};

nlohmann::json link_serving_libraries(std::map<std::size_t, TemplateLibrary> & libraries);

} // namespace prologue::sglang
