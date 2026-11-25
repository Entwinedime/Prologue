#include "adapters/registry.hpp"

#include <stdexcept>
#include <string_view>

#include "adapters/sglang/serving.hpp"
#include "adapters/sglang/templates.hpp"
#include "adapters/sglang/workers.hpp"

namespace prologue {

const Framework & framework(const nlohmann::json & environment) {
    static const std::map<std::string_view, Framework> adapters{
        {"sglang", {sglang::NativeTemplates::describe, sglang::NativeTemplates::compile, sglang::NativeTemplates::observed_work,
                    sglang::NativeTemplates::domain, sglang::link_serving_libraries,
                    [](const TemplateLibrary & library, const nlohmann::json & target) -> std::unique_ptr<LocalWorkerBindings> {
                        return std::make_unique<sglang::WorkerBindings>(library, target);
                    }, sglang::serve}}
    };
    const auto & name = environment.at("framework").get_ref<const std::string &>();
    const auto found = adapters.find(name);
    if (found == adapters.end()) throw std::invalid_argument("unsupported serving framework: " + name);
    return found->second;
}

const Framework & framework(const std::map<std::size_t, TemplateLibrary> & libraries) {
    const auto & adapter = framework(libraries.at(0).domain);
    for (const auto & [_, library] : libraries) {
        if (&framework(library.domain) != &adapter) throw std::invalid_argument("serving libraries must use the same framework");
    }
    return adapter;
}

} // namespace prologue
