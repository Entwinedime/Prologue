#pragma once

#include <cstddef>
#include <map>

#include <nlohmann/json.hpp>

#include "core/templates/library.hpp"

namespace prologue {

class ControlPeer;
class ExecutionObserver;
class ResourceObserver;
class ControlObserver;

} // namespace prologue

namespace prologue::sglang {

nlohmann::json serve(const std::map<std::size_t, TemplateLibrary> & libraries, const nlohmann::json & task,
                     const nlohmann::json & timing, const nlohmann::json & costs, ControlPeer & peer,
                     ExecutionObserver * execution = nullptr,
                     ResourceObserver * resources = nullptr, ControlObserver * control = nullptr);

} // namespace prologue::sglang
