#include "core/performance/work_features.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace prologue {

nlohmann::json WorkFeatures::transport_operand(const GraphScope & scope) {
    if (!scope.communication || std::ranges::any_of(scope.ports, [](const auto & port) {
        return port.tensor && !port.tensor->shape_rule;
    })) return nullptr;
    auto ports = nlohmann::json::object();
    for (const auto & port : scope.ports) {
        if (!port.tensor) continue;

        const auto direction = std::string(direction_name(port.direction));
        const auto axis = "collective." + direction + "." + port.name + ".elements";
        std::int64_t elements = 1;
        for (const auto extent : port.tensor->shape) {
            if (__builtin_mul_overflow(elements, extent, &elements))
                throw std::overflow_error("collective operand element count exceeds the supported range");
        }
        axes_.emplace(axis, elements);
        ports[direction][port.name] = {
            { "dtype", port.tensor->dtype }, { "dimensions", port.tensor->shape.size() }, { "elements_axis", axis },
        };
    }
    if (ports.empty()) return nullptr;

    // Layout and placement identify the service. Concrete payload sizes are model coordinates.
    const auto & group = *scope.communication;
    nlohmann::json identity{
        { "operation", group.operation }, { "members", group.members }, { "required", group.required },
        { "rank", group.rank }, { "implementation", scope.info.implementation }, { "ports", std::move(ports) },
    };
    const auto local = scope.info.bindings.find("local_implementation");
    if (local != scope.info.bindings.end()) identity["local_implementation"] = *local;
    if (group.peer_rows) {
        auto names = nlohmann::json::array();
        for (std::size_t peer = 0; peer < group.members.size(); ++peer) {
            const auto prefix = "collective.peer." + std::to_string(group.members[peer]);
            const auto send = prefix + ".send_rows", receive = prefix + ".receive_rows";
            axes_.emplace(send, group.peer_rows->send[peer]);
            axes_.emplace(receive, group.peer_rows->receive[peer]);
            names.push_back(send);
            names.push_back(receive);
        }
        identity["row_axes"] = std::move(names);
    }
    return identity;
}

WorkFeatures::WorkFeatures(Parameters parameters) {
    for (const auto & [name, value] : parameters) axes_.emplace(std::string(parameter_prefix_) + name, value);
}

WorkFeatures::WorkFeatures(const TensorCopy & copy, const GraphScope & scope, const Resource & resource)
    : transport_({{ "operation", "local_copy" }, { "caller", copy.caller }, { "implementation", copy.implementation },
                  { "direction", copy.direction }, { "resource", { resource_name(resource.kind), resource.role }} }),
      copy_bytes_(copy.bytes(scope.info.parameters)) {}

WorkFeatures::WorkFeatures(const GraphScope & scope, WorkKind kind) {
    if (kind == WorkKind::Collective) transport_ = transport_operand(scope);
    const auto & info = scope.info;
    if (!info.parameters.contains("members")) {

        // Local calls declare their own inputs. The enclosing batch can identify the call without driving its cost.
        for (const auto & [name, value] : info.parameters) {
            if (!info.constants.contains(name)) axes_.emplace(std::string(parameter_prefix_) + name, value);
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
    for (const auto & name : geometry) axes_.emplace(std::string(parameter_prefix_) + name, info.parameters.at(name));
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
    if (axis == "transfer.bytes") return copy_bytes_;
    if (const auto point = axes_.find(axis); point != axes_.end()) return static_cast<double>(point->second);
    const auto axes = batch_axes();
    const auto batch = std::ranges::find(axes, axis);
    if (batch == axes.end()) return {};
    return batch_[static_cast<std::size_t>(batch - axes.begin())];
}

nlohmann::json WorkFeatures::document() const {
    auto axes = nlohmann::json::object();
    const auto names = batch_axes();
    for (std::size_t index = 0; index < names.size(); ++index) axes[names[index]] = batch_[index];
    for (const auto & [name, value] : axes_) axes[name] = value;
    if (copy_bytes_) axes["transfer.bytes"] = *copy_bytes_;
    nlohmann::json result{{ "phase", phase_ }, { "axes", std::move(axes) }};
    if (transport()) result["transport"] = transport_;
    return result;
}

std::size_t WorkCoordinates::index(const GraphWork & work) {
    const auto & scope = graph_.scopes[work.scope];
    if (work.copy) {
        points_.emplace_back(*work.copy, scope, graph_.resources[work.resource]);
        return points_.size() - 1;
    }

    // Only a peer service inherits the rendezvous operand. All other ordinary intervals share local geometry.
    auto & index = by_scope_[work.scope][work.kind == WorkKind::Collective];
    if (!index) {
        index = points_.size();
        points_.emplace_back(scope, work.kind);
    }
    return *index;
}

} // namespace prologue
