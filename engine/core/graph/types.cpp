#include "core/graph/types.hpp"

#include "core/contracts/serving.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>
#include <utility>

namespace prologue {
namespace {

constexpr std::array<std::string_view, 3> resource_names{ "thread", "stream", "service" };
constexpr std::array<std::string_view, 7> work_names{ "host", "device", "transfer", "communication", "collective", "wait", "boundary" };
constexpr std::array<std::string_view, 8> relation_names{ "order", "submission", "data", "signal", "control", "state", "communication", "observation" };
constexpr std::array<std::string_view, 2> direction_names{ "input", "output" };
constexpr std::array<std::string_view, 4> port_use_names{ "control", "data", "metadata", "signal" };

template <class Kind, std::size_t Size> Kind read_kind(std::string_view name, const std::array<std::string_view, Size> & names) {
    const auto found = std::ranges::find(names, name);
    if (found == names.end()) throw std::invalid_argument("unknown graph kind: " + std::string(name));
    return static_cast<Kind>(found - names.begin());
}

std::vector<std::int64_t> communicating_peers(const CommunicationGroup & group, const PeerRows & rows) {
    if (rows.send.size() != group.members.size() || rows.receive.size() != group.members.size())
        throw std::invalid_argument("peer rows must cover the declared communication group");

    std::vector<std::int64_t> result;
    for (std::size_t peer = 0; peer < group.members.size(); ++peer) {
        const auto sent = rows.send[peer], received = rows.receive[peer];
        if (sent < 0 || received < 0) throw std::invalid_argument("peer rows must be nonnegative");
        if (group.members[peer] == group.rank || sent || received) result.push_back(group.members[peer]);
    }
    return result;
}

} // namespace

ResourceKind resource_kind(std::string_view name) { return read_kind<ResourceKind>(name, resource_names); }
std::string_view resource_name(ResourceKind kind) { return resource_names.at(static_cast<std::size_t>(kind)); }
WorkKind work_kind(std::string_view name) { return read_kind<WorkKind>(name, work_names); }
std::string_view work_name(WorkKind kind) { return work_names.at(static_cast<std::size_t>(kind)); }
Relation relation_kind(std::string_view name) { return read_kind<Relation>(name, relation_names); }
std::string_view relation_name(Relation kind) { return relation_names.at(static_cast<std::size_t>(kind)); }
Direction direction_kind(std::string_view name) { return read_kind<Direction>(name, direction_names); }
std::string_view direction_name(Direction direction) { return direction_names.at(static_cast<std::size_t>(direction)); }
PortUse port_use(std::string_view name) { return read_kind<PortUse>(name, port_use_names); }
std::string_view port_use_name(PortUse use) { return port_use_names.at(static_cast<std::size_t>(use)); }

ResourceClass ResourceClass::read(const nlohmann::json & source) {
    return {resource_kind(source.at("kind").get<std::string>()), source.at("role").get<std::string>(), source.at("serial").get<bool>()};
}

nlohmann::json ResourceClass::document() const {
    return {{"kind", resource_name(kind)}, {"role", role}, {"serial", serial}};
}

Resource Resource::read(const nlohmann::json & source) {
    Resource result{ ResourceClass::read(source), read_integer(source.at("rank"), "rank"), source.at("identity").get<std::string>() };
    if (result.rank < 0 || result.role.empty() || result.identity.empty()) throw std::invalid_argument("invalid graph resource identity");
    if (result.kind != ResourceKind::Service && !result.serial) throw std::invalid_argument("threads and streams require serial order");
    if (source.contains("binding")) {
        result.binding = source.at("binding").get<std::string>();
        if (result.binding->empty()) throw std::invalid_argument("empty logical resource binding");
    }
    return result;
}

nlohmann::json Resource::document() const {
    auto result = ResourceClass::document();
    result["rank"] = rank;
    result["identity"] = identity;
    if (binding) result["binding"] = *binding;
    return result;
}

TensorRule TensorRule::read(const nlohmann::json & source) {
    TensorRule result{ source.at("dtype").get<std::string>(), {} };
    for (const auto & dimension : source.at("shape").get_ref<const nlohmann::json::array_t &>())
        result.shape.push_back(Expression::read(dimension));
    if (result.dtype.empty()) throw std::invalid_argument("tensor geometry requires a dtype");
    if (source.contains("packing")) result.packing = std::make_shared<const RowPacking>(RowPacking::read(source.at("packing")));
    return result;
}

nlohmann::json TensorRule::document(const std::map<std::string, Expression> & arguments) const {
    auto dimensions = nlohmann::json::array();
    for (const auto & expression : shape) dimensions.push_back(expression.document(arguments));
    nlohmann::json result{{ "dtype", dtype }, { "shape", std::move(dimensions) }};
    if (packing) result["packing"] = packing->document(arguments);
    return result;
}

TensorLayout TensorRule::evaluate(const Parameters & parameters) const {
    TensorLayout result{ dtype, {} };
    for (const auto & expression : shape) {
        const auto dimension = expression.evaluate(parameters);
        if (dimension < 0) throw std::invalid_argument("tensor rule produces a negative dimension");
        result.shape.push_back(dimension);
    }
    return result;
}

std::shared_ptr<const TensorCopy> TensorCopy::read(const nlohmann::json & work) {
    const auto found = work.find("copy");
    if (found == work.end()) return {};
    const auto & copy = *found;
    TensorCopy result{ copy.at("operand").get<std::string>(), TensorRule::read(copy.at("tensor")),
                       read_integer(copy.at("element_bytes"), "element_bytes"), copy.at("caller").get<std::string>(),
                       copy.at("implementation").get<std::string>(), copy.at("direction").get<std::string>() };
    if (work.at("kind") != "transfer" || result.operand.empty() || result.element_bytes <= 0 || result.caller.empty()
        || result.implementation.empty() || result.direction.empty())
        throw std::invalid_argument("a tensor copy requires a transfer interval, operand, element width and native service");
    return std::make_shared<const TensorCopy>(std::move(result));
}

nlohmann::json TensorCopy::document() const {
    return {{ "operand", operand }, { "tensor", tensor.document() }, { "element_bytes", element_bytes },
            { "caller", caller }, { "implementation", implementation }, { "direction", direction }};
}

std::int64_t TensorCopy::bytes(const Parameters & parameters) const {
    auto size = element_bytes;
    for (const auto extent : tensor.evaluate(parameters).shape) {
        if (__builtin_mul_overflow(size, extent, &size)) throw std::overflow_error("copy operand byte extent exceeds the supported range");
    }
    return size;
}

TensorObservation TensorObservation::read(const nlohmann::json & source, const ScopeInfo & scope) {
    TensorObservation result{ source.at("value").get<std::string>(),
                              source.at("storage").get<std::string>(),
                              read_integer(source.at("version"), "version"),
                              source.at("dtype").get<std::string>(),
                              {},
                              std::nullopt };
    for (const auto & dimension : source.at("shape").get_ref<const nlohmann::json::array_t &>()) result.shape.push_back(read_integer(dimension, "shape dimension"));
    if (result.value.empty() || result.storage.empty() || result.dtype.empty() || result.version < 0
        || std::ranges::any_of(result.shape, [](auto size) { return size < 0; }))
        throw std::invalid_argument("invalid observed tensor boundary");

    if (source.contains("shape_rule") && !source.at("shape_rule").is_null()) {
        const auto & rules = source.at("shape_rule");
        if (!rules.is_array() || rules.size() != result.shape.size()) throw std::invalid_argument("shape rule rank differs from observation");
        result.shape_rule.emplace();
        for (const auto & rule : rules) result.shape_rule->push_back(Expression::read(rule));
    }
    if (!source.at("writer").is_null()) {
        const auto writer = read_integer(source.at("writer"), "writer");
        if (writer < 0) throw std::invalid_argument("invalid tensor writer scope");
        result.writer = static_cast<std::size_t>(writer);
    }
    if (!source.at("allocation").is_null()) {
        const auto allocation = read_integer(source.at("allocation"), "allocation");
        if (allocation < 0) throw std::invalid_argument("invalid tensor allocation scope");
        result.allocation = static_cast<std::size_t>(allocation);
    }
    result.buffer = source.at("buffer").get<std::string>();
    if (const auto found = source.find("packing"); found != source.end()) {
        result.bind_rows(std::make_shared<const RowPacking>(RowPacking::read(*found)), scope);
        if (result.packing->slices != found->at("slices")) throw std::invalid_argument("declared row packing differs from observed writes");
    }
    return result;
}

void TensorObservation::bind_rows(const std::shared_ptr<const RowPacking> & rule, const ScopeInfo & scope) {
    packing = std::make_shared<const PackedRows>(rule, rule->bind(scope.work, scope.parameters, shape));
}

nlohmann::json TensorObservation::document() const {
    nlohmann::json result{
        {      "value",                                                              value },
        {    "storage",                                                            storage },
        {    "version",                                                            version },
        {      "dtype",                                                              dtype },
        {      "shape",                                                              shape },
        {     "writer",         writer ? nlohmann::json(*writer) : nlohmann::json(nullptr) },
        { "allocation", allocation ? nlohmann::json(*allocation) : nlohmann::json(nullptr) },
        {     "buffer", buffer }
    };
    if (shape_rule) {
        result["shape_rule"] = nlohmann::json::array();
        for (const auto & rule : *shape_rule) result["shape_rule"].push_back(rule.document());
    }
    if (packing) {
        result["packing"] = packing->rule->document();
        result["packing"]["slices"] = packing->slices;
    }
    return result;
}

ScopeInfo ScopeInfo::read(const nlohmann::json & source) {
    ScopeInfo result;
    const auto & parent = source.at("parent");
    if (!parent.is_null()) {
        const auto value = read_integer(parent, "parent");
        if (value < 0) throw std::invalid_argument("invalid scope parent");
        result.parent = static_cast<std::size_t>(value);
    }
    result.role = source.at("role").get<std::string>();
    result.site = source.at("site").get<std::string>();
    result.implementation = source.at("implementation").get<std::string>();
    result.parameters = read_parameters(source.at("parameters"));
    result.constants = read_parameters(source.value("constants", nlohmann::json::object()));
    result.work = source.at("work");
    result.bindings = source.value("bindings", nlohmann::json::object());
    if (result.role.empty() || !result.work.is_object() || !result.bindings.is_object()) throw std::invalid_argument("invalid scope identity or bindings");

    // Fixed implementation facts must agree with the observation, never override it.
    for (const auto & [name, value] : result.constants) {
        if (result.parameters.at(name) != value) throw std::invalid_argument("scope constant differs from observation: " + name);
    }
    const auto arguments = source.value("arguments", nlohmann::json::object());
    if (!arguments.is_object()) throw std::invalid_argument("scope arguments must be an object");
    for (const auto & [name, value] : arguments.items()) result.arguments.emplace(name, Expression::read(value));
    for (const auto & guard : source.value("guards", nlohmann::json::array())) result.guards.push_back(Expression::read(guard));
    if (source.contains("row_call")) result.row_call = std::make_shared<RowCall>(RowCall::read(source.at("row_call")));
    return result;
}

nlohmann::json ScopeInfo::document() const {
    nlohmann::json arguments_json = nlohmann::json::object();
    for (const auto & [name, expression] : arguments) arguments_json[name] = expression.document();
    auto conditions = nlohmann::json::array();
    for (const auto & guard : guards) conditions.push_back(guard.document());
    nlohmann::json result{
        {         "parent", parent ? nlohmann::json(*parent) : nlohmann::json(nullptr) },
        {           "role",                                                       role },
        {           "site",                                                       site },
        { "implementation",                                             implementation },
        {     "parameters",                                                 parameters },
        {      "constants",                                                  constants },
        {      "arguments",                                  std::move(arguments_json) },
        {           "work",                                                       work },
        {       "bindings",                                                   bindings },
        {         "guards",                                      std::move(conditions) }
    };
    if (row_call) result["row_call"] = row_call->document();
    return result;
}

PeerRows PeerRows::read(const nlohmann::json & source) {
    auto rows = [&](std::string_view name) {
        std::vector<std::int64_t> result;
        for (const auto & value : source.at(name).get_ref<const nlohmann::json::array_t &>()) result.push_back(read_integer(value, name));
        return result;
    };
    return { rows("send"), rows("receive") };
}

nlohmann::json PeerRows::document() const {
    return {{ "send", send }, { "receive", receive }};
}

CommunicationGroup CommunicationGroup::read(const nlohmann::json & source) {
    const auto ordinal = read_integer(source.at("order"), "order");
    if (ordinal < 0) throw std::invalid_argument("communication order must be nonnegative");
    CommunicationGroup result{ source.at("channel").get<std::string>(),
                               source.at("operation").get<std::string>(),
                               source.at("members").get<std::vector<std::int64_t>>(),
                               source.at("required").get<std::vector<std::int64_t>>(),
                               source.at("rank").get<std::int64_t>(),
                               static_cast<std::size_t>(ordinal) };
    if (source.contains("agreements")) {
        for (const auto & field : source.at("agreements").get_ref<const nlohmann::json::array_t &>())
            result.agreements.push_back({ field.at("port").get<std::string>(), direction_kind(field.at("direction").get<std::string>()),
                                          field.at("shape").get<bool>() });
    }
    result.ordered_service = source.value("ordered_service", false);
    if (const auto rows = source.find("peer_rows"); rows != source.end()) result.peer_rows = PeerRows::read(*rows);
    result.validate();
    return result;
}

nlohmann::json CommunicationGroup::document() const {
    nlohmann::json result{
        {   "channel",   channel },
        { "operation", operation },
        {   "members",   members },
        {  "required",  required },
        {      "rank",      rank },
        {     "order",     order }
    };
    if (!agreements.empty()) {
        result["agreements"] = nlohmann::json::array();
        for (const auto & field : agreements)
            result["agreements"].push_back({ { "port", field.port }, { "direction", direction_name(field.direction) }, { "shape", field.shape } });
    }
    if (ordered_service) result["ordered_service"] = true;
    if (peer_rows) result["peer_rows"] = peer_rows->document();
    return result;
}

void CommunicationGroup::bind_rows(PeerRows rows) {
    required = communicating_peers(*this, rows);
    peer_rows = std::move(rows);
}

void CommunicationGroup::validate() const {
    const std::set<std::int64_t> participants(members.begin(), members.end());
    const std::set<std::int64_t> dependencies(required.begin(), required.end());
    if (channel.empty() || operation.empty() || participants.empty() || participants.size() != members.size() || *participants.begin() < 0
        || !participants.contains(rank) || dependencies.size() != required.size() || !std::ranges::includes(participants, dependencies))
        throw std::invalid_argument("invalid communication participant contract");
    if (peer_rows && required != communicating_peers(*this, *peer_rows))
        throw std::invalid_argument("communication prerequisites differ from the declared peer rows");
    if (ordered_service && operation != "transfer")
        throw std::invalid_argument("ordered peer service requires a transfer");
    std::set<std::pair<Direction, std::string>> fields;
    for (const auto & field : agreements) {
        if (field.port.empty() || !fields.emplace(field.direction, field.port).second)
            throw std::invalid_argument("communication payload agreement requires distinct named ports");
    }
}

} // namespace prologue
