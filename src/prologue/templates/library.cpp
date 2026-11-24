#include "prologue/templates/library.hpp"

#include <map>
#include <stdexcept>
#include <utility>

#include "prologue/templates/selection.hpp"

namespace prologue {
namespace {

std::size_t index(const nlohmann::json & value) {
    if (!value.is_number_integer() || value.get<std::int64_t>() < 0) throw std::invalid_argument("template index must be a nonnegative integer");
    return value.get<std::size_t>();
}

std::vector<std::size_t> indices(const nlohmann::json & source) {
    std::vector<std::size_t> result;
    for (const auto & item : source.get_ref<const nlohmann::json::array_t &>()) result.push_back(index(item));
    return result;
}

} // namespace

ObservedArguments ObservedArguments::read(nlohmann::json source) {
    ObservedArguments point;
    if (const auto extents = source.find("member_extents"); extents != source.end()) {
        for (const auto & pair : extents->get_ref<const nlohmann::json::array_t &>()) {
            if (!pair.is_array() || pair.size() != 2) throw std::invalid_argument("an observed member needs its context and new token counts");
            point.member_extents.emplace_back(index(pair[0]), index(pair[1]));
        }
        source.erase(extents);
    }
    point.parameters = read_parameters(source);
    return point;
}

nlohmann::json ObservedArguments::document() const {
    nlohmann::json result = parameters;
    if (result.contains("member_extents")) throw std::invalid_argument("member_extents belongs to the observation, outside its scalar parameters");
    if (!member_extents.empty()) result["member_extents"] = member_extents;
    return result;
}

nlohmann::json ObservedArguments::work() const {
    auto members = nlohmann::json::array();
    for (const auto & [context, added] : member_extents) members.push_back({{ "context_tokens", context }, { "new_tokens", added }});
    return {{ "members", std::move(members) }};
}

nlohmann::json TemplateGuard::document() const {
    auto rules = nlohmann::json::array();
    for (const auto & rule : branches) rules.push_back(rule.document());
    return { { "branches", std::move(rules) }, { "extents", extents.document() } };
}

TemplateGuard TemplateGuard::read(const nlohmann::json & source) {
    TemplateGuard result{ {}, Expression::read(source.at("extents")) };
    for (const auto & rule : source.at("branches").get_ref<const nlohmann::json::array_t &>()) result.branches.push_back(Expression::read(rule));
    return result;
}

StateBindings StateBindings::read(const nlohmann::json & source) {
    StateBindings result;
    for (const auto & value : source.at("kv"))
        result.kv.push_back({ value.at("pool").get<std::string>(), static_cast<std::int64_t>(index(value.at("layer"))), KvSlice::read(value.at("slice")) });
    result.slots = source.at("slots").get<std::vector<std::string>>();
    return result;
}

nlohmann::json StateBindings::document() const {
    nlohmann::json result{{ "kv", nlohmann::json::array() }, { "slots", slots }};
    for (const auto & value : kv)
        result["kv"].push_back({{ "pool", value.pool }, { "layer", value.layer }, { "slice", value.slice.document() }});
    return result;
}

TemplateReference TemplateReference::read(const nlohmann::json & source) {
    const auto & child = source.at("child");
    return { index(source.at("event")), child.is_null() ? std::nullopt : std::optional(index(child)), source.at("imported").get<bool>() };
}

nlohmann::json TemplateReference::document() const {
    return {
        {    "event",                                                    event },
        {    "child", child ? nlohmann::json(*child) : nlohmann::json(nullptr) },
        { "imported",                                                 imported }
    };
}

nlohmann::json TemplatePort::operand(const std::map<std::string, Expression> & arguments) const {
    return {
        {      "name",                      name },
        { "direction", direction_name(direction) },
        {       "use",        port_use_name(use) },
        {  "relation",   relation_name(relation) },
        {    "tensor", tensor ? tensor->document(arguments) : nlohmann::json(nullptr) }
    };
}

nlohmann::json TemplatePort::document() const {
    auto result = operand();
    result["event"] = event.document();
    if (rendezvous) result["rendezvous"] = rendezvous->document();
    if (wait)
        result["wait"] = {
            { "resource",         wait->resource },
            {    "begin", wait->begin.document() }
        };
    return result;
}

TemplateInterface TemplateInterface::read(const nlohmann::json & source) {
    TemplateInterface result{ source.at("role").get<std::string>(),
                              source.at("implementation").get<std::string>(),
                              source.at("parameters").get<std::set<std::string>>(),
                              source.at("fixed").get<std::set<std::string>>(),
                              {},
                              indices(source.at("imports")),
                              indices(source.at("exports")),
                              indices(source.at("boundaries")),
                              source.at("ports"),
                              {} };
    for (const auto & resource : source.at("resources").get_ref<const nlohmann::json::array_t &>())
        result.resources.push_back(
            { resource_kind(resource.at("kind").get<std::string>()), resource.at("role").get<std::string>(), resource.at("serial").get<bool>() });
    for (const auto & relation : source.at("causality")) result.causality.emplace_back(index(relation.at(0)), index(relation.at(1)));
    return result;
}

nlohmann::json TemplateInterface::document() const {
    nlohmann::json result{
        {           "role",                    role },
        { "implementation",          implementation },
        {     "parameters",              parameters },
        {          "fixed",                   fixed },
        {        "imports",                 imports },
        {        "exports",                 exports },
        {     "boundaries",              boundaries },
        {          "ports",                   ports },
        {      "causality",               causality },
        {      "resources", nlohmann::json::array() }
    };
    for (const auto & resource : resources)
        result["resources"].push_back({
            {   "kind", resource_name(resource.kind) },
            {   "role",                resource.role },
            { "serial",              resource.serial }
        });
    return result;
}

TemplateDefinition TemplateDefinition::read(const nlohmann::json & source) {
    TemplateDefinition result;
    result.role = source.at("role").get<std::string>();
    result.implementation = source.at("implementation").get<std::string>();
    result.parameters = source.at("parameters").get<std::set<std::string>>();
    result.fixed = source.at("fixed").get<std::set<std::string>>();
    result.events = indices(source.at("events"));
    result.imports = indices(source.at("imports"));
    for (const auto & endpoint : source.at("exports").get_ref<const nlohmann::json::array_t &>()) result.exports.push_back(TemplateReference::read(endpoint));
    for (const auto & access : source.at("slot_accesses"))
        result.slot_accesses.push_back({ access.at("field").get<std::string>(),
                                         state_access(access.at("access").get<std::string>()),
                                         access.at("members").get<std::string>(),
                                         TemplateReference::read(access.at("start")),
                                         TemplateReference::read(access.at("finish")),
                                         access.value("publication", std::string{}) });
    for (const auto & access : source.at("kv_accesses").get_ref<const nlohmann::json::array_t &>())
        result.kv_accesses.push_back({ state_access(access.at("access").get<std::string>()),
                                       TokenRangeRule::read(access.at("rule")),
                                       TemplateReference::read(access.at("start")),
                                       TemplateReference::read(access.at("finish")) });
    for (const auto & resource : source.at("resources").get_ref<const nlohmann::json::array_t &>())
        result.resources.push_back(
            { resource_kind(resource.at("kind").get<std::string>()), resource.at("role").get<std::string>(), resource.at("serial").get<bool>() });

    for (const auto & piece : source.at("work").get_ref<const nlohmann::json::array_t &>())
        result.work.push_back({ index(piece.at("resource")),
                                work_kind(piece.at("kind").get<std::string>()),
                                TemplateReference::read(piece.at("start")),
                                TemplateReference::read(piece.at("finish")), TensorCopy::read(piece) });
    for (const auto & edge : source.at("edges").get_ref<const nlohmann::json::array_t &>())
        result.edges.push_back(
            { TemplateReference::read(edge.at("source")), TemplateReference::read(edge.at("target")), relation_kind(edge.at("kind").get<std::string>()) });

    for (const auto & port : source.at("ports").get_ref<const nlohmann::json::array_t &>()) {
        TemplatePort boundary{ port.at("name").get<std::string>(),          direction_kind(port.at("direction").get<std::string>()),
                               port_use(port.at("use").get<std::string>()), relation_kind(port.at("relation").get<std::string>()),
                               TemplateReference::read(port.at("event")),   std::nullopt };
        if (port.contains("wait")) {
            const auto & wait = port.at("wait");
            boundary.wait = StreamWait<TemplateReference>{ index(wait.at("resource")), TemplateReference::read(wait.at("begin")) };
        }
        if (!port.at("tensor").is_null()) boundary.tensor = TensorRule::read(port.at("tensor"));
        if (port.contains("rendezvous")) boundary.rendezvous = TemplateReference::read(port.at("rendezvous"));
        result.ports.push_back(std::move(boundary));
    }
    for (const auto & child : source.at("children").get_ref<const nlohmann::json::array_t &>()) {
        TemplateCall call{ TemplateInterface::read(child.at("contract")), {}, {} };
        if (child.contains("row_call")) call.row = std::make_shared<RowCall>(RowCall::read(child.at("row_call")));
        const auto & arguments = child.at("arguments").get_ref<const nlohmann::json::object_t &>();
        for (const auto & [name, expression] : arguments) call.arguments.emplace(name, Expression::read(expression));
        for (const auto & imported : child.at("imports").get_ref<const nlohmann::json::array_t &>()) call.imports.push_back(TemplateReference::read(imported));
        result.children.push_back(std::move(call));
    }
    return result;
}

nlohmann::json TemplateDefinition::document() const {
    nlohmann::json result{
        {           "role",                    role },
        { "implementation",          implementation },
        {     "parameters",              parameters },
        {          "fixed",                   fixed },
        {         "events",                  events },
        {        "imports",                 imports },
        {        "exports", nlohmann::json::array() },
        {      "resources", nlohmann::json::array() },
        {           "work", nlohmann::json::array() },
        {          "edges", nlohmann::json::array() },
        {          "ports", nlohmann::json::array() },
        {       "children", nlohmann::json::array() }
    };
    result["slot_accesses"] = nlohmann::json::array();
    for (const auto & access : slot_accesses) {
        result["slot_accesses"].push_back({
            {   "field",                     access.field },
            {  "access", state_access_name(access.access) },
            { "members",                   access.members },
            {   "start",          access.start.document() },
            {  "finish",         access.finish.document() }
        });
        if (!access.publication.empty()) result["slot_accesses"].back()["publication"] = access.publication;
    }
    result["kv_accesses"] = nlohmann::json::array();
    for (const auto & endpoint : exports) result["exports"].push_back(endpoint.document());
    for (const auto & access : kv_accesses)
        result["kv_accesses"].push_back({
            { "access", state_access_name(access.access) },
            {   "rule",           access.rule.document() },
            {  "start",          access.start.document() },
            { "finish",         access.finish.document() }
        });
    for (const auto & resource : resources)
        result["resources"].push_back({
            {   "kind", resource_name(resource.kind) },
            {   "role",                resource.role },
            { "serial",              resource.serial }
        });
    for (const auto & piece : work) {
        result["work"].push_back({
            { "resource",          piece.resource },
            {     "kind",   work_name(piece.kind) },
            {    "start",  piece.start.document() },
            {   "finish", piece.finish.document() }
        });
        if (piece.copy) result["work"].back()["copy"] = piece.copy->document();
    }
    for (const auto & edge : edges)
        result["edges"].push_back({
            { "source",   edge.source.document() },
            { "target",   edge.target.document() },
            {   "kind", relation_name(edge.kind) }
        });

    for (const auto & port : ports) result["ports"].push_back(port.document());
    for (const auto & child : children) {
        nlohmann::json arguments = nlohmann::json::object();
        for (const auto & [name, expression] : child.arguments) arguments[name] = expression.document();
        auto imports_json = nlohmann::json::array();
        for (const auto & imported : child.imports) imports_json.push_back(imported.document());
        result["children"].push_back({
            {  "contract", child.contract.document() },
            { "arguments",      std::move(arguments) },
            {   "imports",   std::move(imports_json) }
        });
        if (child.row) result["children"].back()["row_call"] = child.row->document();
    }
    return result;
}

namespace {

TemplateFrame read_frame(const nlohmann::json & source, const std::map<std::size_t, std::size_t> & definitions,
                         const nlohmann::json & entry, const nlohmann::json & domain) {
    TemplateFrame result;
    const auto id = index(source.at("definition"));
    const auto found = definitions.find(id);
    if (found == definitions.end())
        throw TemplateGap({ { "status", "missing_definition" }, { "action", entry.at("action") }, { "call", entry.at("frame").at("site") },
                            { "site", source.at("site") }, { "definition", id }, { "domain", domain } });

    result.definition = found->second;
    result.guard = TemplateGuard::read(source.at("guard"));
    if (source.contains("communication")) result.communication = CommunicationGroup::read(source.at("communication"));
    result.site = source.at("site").get<std::string>();
    result.bindings = source.at("bindings");
    result.constants = read_parameters(source.at("constants"));
    result.resources = indices(source.at("resources"));
    if (const auto owned = source.find("allocations"); owned != source.end())
        result.allocations = owned->get<std::set<std::string>>();
    result.state = StateBindings::read(source.at("state"));
    if (!source.at("state_input").is_null()) result.state_input = source.at("state_input").get<std::string>();
    for (const auto & tensor : source.at("tensors").get_ref<const nlohmann::json::array_t &>()) {
        if (tensor.is_null()) result.tensors.push_back(std::nullopt);
        else {
            const auto version = index(tensor.at("version"));
            result.tensors.push_back(TensorBinding{ tensor.at("value").get<std::string>(),
                                                    tensor.at("storage").get<std::string>(),
                                                    static_cast<std::int64_t>(version),
                                                    tensor.at("external").get<bool>(),
                                                    std::nullopt,
                                                    tensor.at("buffer").get<std::string>() });
            if (!tensor.at("writer").is_null()) result.tensors.back()->writer = indices(tensor.at("writer"));
        }
    }
    for (const auto & child : source.at("children").get_ref<const nlohmann::json::array_t &>())
        result.children.push_back(read_frame(child, definitions, entry, domain));
    if (const auto alternatives = source.find("alternatives"); alternatives != source.end()) {
        for (const auto & branch : alternatives->get_ref<const nlohmann::json::array_t &>())
            result.alternatives.push_back({read_frame(branch.at("frame"), definitions, entry, domain), ObservedArguments::read(branch.at("input"))});
    }
    return result;
}

} // namespace

nlohmann::json TemplateFrame::local_document() const {
    nlohmann::json result{
        { "definition",              definition },
        {      "guard",        guard.document() },
        {       "site",                    site },
        {   "bindings",                bindings },
        {  "constants",               constants },
        {  "resources",               resources },
        {    "tensors", nlohmann::json::array() },
        {      "state",        state.document() },
        { "state_input", state_input ? nlohmann::json(*state_input) : nlohmann::json(nullptr) }
    };
    for (const auto & tensor : tensors) {
        if (tensor)
            result["tensors"].push_back({
                {    "value",                                                              tensor->value },
                {  "storage",                                                            tensor->storage },
                {  "version",                                                            tensor->version },
                { "external",                                                           tensor->external },
                {   "buffer",                                                             tensor->buffer },
                {   "writer", tensor->writer ? nlohmann::json(*tensor->writer) : nlohmann::json(nullptr) }
            });
        else result["tensors"].push_back(nullptr);
    }
    if (!allocations.empty()) result["allocations"] = allocations;
    if (communication) result["communication"] = communication->document();
    return result;
}

nlohmann::json TemplateFrame::document() const {
    auto result = local_document();
    result["children"] = nlohmann::json::array();
    for (const auto & child : children) result["children"].push_back(child.document());
    for (const auto & alternative : alternatives)
        result["alternatives"].push_back({{"frame", alternative.frame.document()}, {"input", alternative.input.document()}});
    return result;
}

TemplateLibrary TemplateLibrary::read(const nlohmann::json & source, const nlohmann::json * published_header) {
    auto result = decode(source, published_header);
    result.validate();
    return result;
}

TemplateLibrary TemplateLibrary::decode(const nlohmann::json & source, const nlohmann::json * published_header) {
    TemplateLibrary result;
    const auto & header = published_header ? *published_header : source;
    result.domain = header.at("domain");
    result.local_domain = header.at("local_domain");
    for (const auto & resource : header.at("resources").get_ref<const nlohmann::json::array_t &>()) result.resources.push_back(Resource::read(resource));

    std::map<std::size_t, std::size_t> definitions;
    for (const auto & definition : source.at("definitions").get_ref<const nlohmann::json::array_t &>()) {
        const auto position = result.definitions.size();
        const auto id = published_header ? index(definition.at("id")) : position;
        if (!definitions.emplace(id, position).second) throw std::invalid_argument("duplicate published template definition id");
        result.definitions.push_back(TemplateDefinition::read(definition));
    }

    for (const auto & entry : source.at("entries").get_ref<const nlohmann::json::array_t &>()) {
        TemplateEntry value{ entry.at("action").get<std::string>(), entry.at("condition"),
                             read_frame(entry.at("frame"), definitions, entry, result.domain), {} };
        for (const auto & observation : entry.at("observations").get_ref<const nlohmann::json::array_t &>())
            value.observations.insert(ObservedArguments::read(observation));
        result.entries.push_back(std::move(value));
    }
    for (const auto & local : source.at("implementations")) {
        const auto id = index(local.at("definition"));
        const auto found = definitions.find(id);
        if (found == definitions.end())
            throw TemplateGap({ { "status", "missing_definition" }, { "contract", local.at("contract") },
                                { "definition", id }, { "domain", result.domain } });
        result.implementations.push_back({ local.at("contract"), found->second, TemplateGuard::read(local.at("guard")) });
    }
    return result;
}

nlohmann::json TemplateLibrary::document() const {
    nlohmann::json result{
        {      "domain",                  domain },
        { "local_domain",           local_domain },
        { "implementations", nlohmann::json::array() },
        {   "resources", nlohmann::json::array() },
        { "definitions", nlohmann::json::array() },
        {     "entries", nlohmann::json::array() }
    };
    for (const auto & resource : resources) result["resources"].push_back(resource.document());
    for (const auto & definition : definitions) result["definitions"].push_back(definition.document());
    for (const auto & local : implementations)
        result["implementations"].push_back({ { "contract", local.contract }, { "definition", local.definition }, { "guard", local.guard.document() } });
    for (const auto & entry : entries) {
        auto observations = nlohmann::json::array();
        for (const auto & point : entry.observations) observations.push_back(point.document());
        result["entries"].push_back({
            {       "action",            entry.action },
            {    "condition",         entry.condition },
            {        "frame",  entry.frame.document() },
            { "observations", std::move(observations) }
        });
    }
    return result;
}


} // namespace prologue
