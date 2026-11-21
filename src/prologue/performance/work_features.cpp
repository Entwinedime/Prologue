#include "prologue/performance/work_features.hpp"

#include <algorithm>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>

namespace prologue {

namespace {

nlohmann::json transport_operand(const GraphScope & scope) {
    if (!scope.communication) return nullptr;
    auto ports = nlohmann::json::object();
    for (const auto & port : scope.ports) {
        if (!port.tensor) continue;
        if (!port.tensor->shape_rule) return nullptr;
        ports[std::string(direction_name(port.direction))][port.name] = {
            { "dtype", port.tensor->dtype }, { "shape", port.tensor->shape },
        };
    }
    if (ports.empty()) return nullptr;

    // A target binds its own ranks and tensor geometry. Stream IDs and posting ordinals belong to the observation.
    const auto & group = *scope.communication;
    nlohmann::json identity{
        { "operation", group.operation }, { "members", group.members }, { "required", group.required },
        { "rank", group.rank }, { "implementation", scope.info.implementation }, { "ports", std::move(ports) },
    };
    const auto local = scope.info.bindings.find("local_implementation");
    if (local != scope.info.bindings.end()) identity["local_implementation"] = *local;
    return identity;
}

} // namespace

WorkFeatures::WorkFeatures(Parameters parameters)
    : parameters_(std::make_move_iterator(parameters.begin()), std::make_move_iterator(parameters.end())) {}

WorkFeatures::WorkFeatures(const GraphScope & scope) : transport_(transport_operand(scope)) {
    const auto & info = scope.info;
    if (!info.parameters.contains("members")) {

        // Local calls declare their own inputs. The enclosing batch can identify the call without driving its cost.
        for (const auto & [name, value] : info.parameters) {
            if (!info.constants.contains(name)) parameters_.emplace(name, value);
        }
        return;
    }

    std::set<std::string> geometry;

    // Metadata describes backing storage. Data ports and local guards describe the call's actual shape.
    for (const auto & port : scope.ports) {
        if (port.use != PortUse::Data || !port.tensor || !port.tensor->shape_rule) continue;
        for (const auto & dimension : *port.tensor->shape_rule) geometry.merge(dimension.variables());
    }
    for (const auto & guard : info.guards) geometry.merge(guard.variables());

    // Inherited values enter the cost coordinate only when the work's own ports or guards use them.
    for (const auto & name : geometry) parameters_.emplace(name, info.parameters.at(name));
    const auto batch = info.work.find("members");
    if (batch == info.work.end()) return;
    if (const auto phase = info.work.find("phase"); phase != info.work.end()) phase_ = phase->get_ref<const std::string &>();

    // A local KV read gives the work an independent context-length and attention coordinate.
    const bool reads_kv = std::ranges::any_of(scope.kv_accesses, [](const auto & access) {
        return access.value.access == StateAccess::Read;
    });
    batch_axis_count_ = reads_kv ? BatchAxisCount : ContextTokens;
    for (const auto & member : batch->get_ref<const nlohmann::json::array_t &>()) {
        const auto query = member.at("new_tokens").get<double>();
        const auto context = member.at("context_tokens").get<double>();
        if (query < 0 || context < 0) throw std::invalid_argument("cost geometry has negative token lengths");

        ++batch_[Members];
        batch_[QueryTokens] += query;
        if (!reads_kv) continue;
        batch_[ContextTokens] += context;
        batch_[MaxContextTokens] = std::max(batch_[MaxContextTokens], context);
        batch_[MaxQueryTokens] = std::max(batch_[MaxQueryTokens], query);
        batch_[AttentionPairs] += query * (context + (query + 1) / 2);
    }
}

std::optional<double> WorkFeatures::value(std::string_view axis) const {
    if (axis.starts_with(parameter_prefix_)) {
        const auto parameter = parameters_.find(axis.substr(parameter_prefix_.size()));
        if (parameter == parameters_.end()) return {};
        return static_cast<double>(parameter->second);
    }
    const auto axes = batch_axes();
    const auto batch = std::ranges::find(axes, axis);
    if (batch == axes.end()) return {};
    return batch_[static_cast<std::size_t>(batch - axes.begin())];
}

nlohmann::json WorkFeatures::document(bool include_transport) const {
    auto axes = nlohmann::json::object();
    const auto names = batch_axes();
    for (std::size_t index = 0; index < names.size(); ++index) axes[names[index]] = batch_[index];
    for (const auto & [name, value] : parameters_) axes[std::string(parameter_prefix_) + name] = value;
    nlohmann::json result{{ "phase", phase_ }, { "axes", std::move(axes) }};
    if (include_transport && transport()) result["transport"] = transport_;
    return result;
}

} // namespace prologue
