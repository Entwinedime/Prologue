#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>

#include "core/templates/production.hpp"

namespace prologue {

class ControlPeer;
class ExecutionObserver;
class ResourceObserver;
class ControlObserver;

// Worker-specific state and placement bind a selected local frame to its target.
class LocalWorkerBindings {
public:
    virtual ~LocalWorkerBindings() = default;
    virtual void bind(TemplateFrame & frame, const std::string & instance, nlohmann::json & work) const = 0;
    virtual std::map<std::string, StateBindings> state_inputs(const nlohmann::json & bindings, const nlohmann::json & work) const = 0;
};

struct Framework {
    TemplateDeclarations (*describe)(CoarseGraph);
    TemplateLibrary (*compile)(const TemplateDeclarations &);
    ObservedTemplateWork (*observed_work)(const TemplateDeclarations &);
    nlohmann::json (*domain)(const nlohmann::json &);
    nlohmann::json (*link)(std::map<std::size_t, TemplateLibrary> &);
    std::unique_ptr<LocalWorkerBindings> (*local_bindings)(const TemplateLibrary &, const nlohmann::json &);
    nlohmann::json (*serve)(const std::map<std::size_t, TemplateLibrary> &, const nlohmann::json &, const nlohmann::json &, const nlohmann::json &,
                  ControlPeer &, ExecutionObserver *, ResourceObserver *, ControlObserver *);
};

const Framework & framework(const nlohmann::json & environment);
const Framework & framework(const std::map<std::size_t, TemplateLibrary> & libraries);

} // namespace prologue
