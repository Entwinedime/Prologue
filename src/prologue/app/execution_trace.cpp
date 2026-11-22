#include "prologue/app/execution_trace.hpp"

#include <utility>
#include <vector>

#include "prologue/contracts/context_values.hpp"
#include "prologue/graph/graph.hpp"
#include "prologue/serving/resources/kv_pool.hpp"

namespace prologue {

ExecutionTrace::ExecutionTrace(const std::filesystem::path & path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    stream_.exceptions(std::ios::badbit | std::ios::failbit);
    stream_.open(path, std::ios::out | std::ios::noreplace);
}

std::size_t ExecutionTrace::context_value(const nlohmann::json & value) {
    const auto [entry, inserted] = context_indices_.try_emplace(value.dump(), context_indices_.size());
    if (inserted) {
        stream_ << "{\"kind\":\"context_value\",\"index\":" << entry->second << ",\"value\":" << entry->first << "}\n";
    }
    return entry->second;
}

nlohmann::json service_estimates(const GraphTiming & timing) {
    auto durations = nlohmann::json::array();
    for (const auto duration : timing.work_ns) durations.push_back(duration ? nlohmann::json(*duration) : nullptr);

    nlohmann::json result{{ "work_ns", std::move(durations) }};
    if (!timing.model_bindings.empty()) {
        auto & bindings = result["model_bindings"] = nlohmann::json::array();
        for (const auto binding : timing.model_bindings) bindings.push_back(binding ? nlohmann::json(*binding) : nullptr);
    }
    return result;
}

void ExecutionTrace::inserted(const CoarseGraph & graph, std::span<const std::size_t> events, const GraphTiming & timing, std::int64_t time_ns) {

    // Input and template assets own the environment. The trace declares each shared context value before its first use.
    auto document = graph.document(false);
    for (auto & scope : document.at("scopes")) {
        ContextValues::encode(scope.at("work"), [this](const auto & value) { return context_value(value); });
    }
    auto record = service_estimates(timing);
    record["kind"] = "graph";
    record["time_ns"] = time_ns;
    record["graph"] = std::move(document);
    record["event_ids"] = events;
    stream_ << record << '\n';
}

void ExecutionTrace::rejected_layout(const CoarseGraph & graph, std::string_view group, std::size_t step) {
    // This action has no timeline events. Preserve its own graph for comparison with the witnessed native call.
    stream_ << nlohmann::json{{ "kind", "rejected_layout" }, { "group", group }, { "step", step }, { "graph", graph.document(false) }} << '\n';
}

void ExecutionTrace::connected(std::size_t source, std::size_t target, Relation relation, std::int64_t delay_ns,
                               std::optional<std::size_t> activation) {
    nlohmann::json record{
        {     "kind",                  "edge" },
        {   "source",                  source },
        {   "target",                  target },
        { "relation", relation_name(relation) },
        { "delay_ns",                delay_ns }
    };
    if (activation) record["activation"] = *activation;
    stream_ << record << '\n';
}

void ExecutionTrace::activated(std::size_t event, std::int64_t time_ns) {
    stream_ << nlohmann::json{
        { "kind", "activation" }, { "event", event }, { "time_ns", time_ns }
    } << '\n';
}

void ExecutionTrace::completed(std::size_t event, std::int64_t time_ns) {
    stream_ << nlohmann::json{
        {    "kind", "completion" },
        {   "event",        event },
        { "time_ns",      time_ns }
    } << '\n';
}

void ExecutionTrace::conditioned(const BoundaryCondition & condition) {
    nlohmann::json record{
        { "kind", condition.window ? "boundary_time" : "logical_boundary" },
        { "event", condition.event }
    };
    if (condition.window) {
        record["begin_ns"] = condition.window->begin_ns;
        record["end_ns"] = condition.window->end_ns;
    }
    if (condition.source_event) record["source_event"] = *condition.source_event;
    stream_ << record << '\n';
}

void ExecutionTrace::initialized(const nlohmann::json & instances, const nlohmann::json & requests, std::int64_t clock_epoch_ns) {

    // Request indices in resource operations resolve through this input order, not through graph event numbers.
    stream_ << nlohmann::json{
        {      "kind", "resources" },
        { "instances",   instances },
        {  "requests",    requests },
        { "clock_epoch_ns", clock_epoch_ns }
    } << '\n';
}

void ExecutionTrace::applied(std::string_view group, const nlohmann::json & operation, const nlohmann::json & result, const ResourceChanges & changes,
                             std::int64_t time_ns) {
    stream_ << nlohmann::json{
        {           "kind",             "resource" },
        {          "group",                  group },
        {        "time_ns",                time_ns },
        {      "operation",              operation },
        {         "result",                 result },
        { "returned_pages", changes.returned_pages },
        { "returned_slots", changes.returned_slots },
        { "returned_fields", changes.returned_fields },
        { "capacity_increased", changes.capacity_increased }
    } << '\n';
}

void ExecutionTrace::condition_used(std::string_view group, std::size_t step, std::string_view operation, const nlohmann::json & source) {

    // The observed graph or input read already owns the actual work identity.
    stream_ << nlohmann::json{
        { "kind", "condition_used" }, { "group", group }, { "step", step }, { "operation", operation },
        { "source", source }
    } << '\n';
}

void ExecutionTrace::observed_values(std::string_view group, std::size_t boundary, const nlohmann::json & values, std::int64_t time_ns) {
    stream_ << nlohmann::json{
        { "kind", "action_values" }, { "group", group }, { "boundary", boundary }, { "values", values }, { "time_ns", time_ns }
    } << '\n';
}

void ExecutionTrace::native_protocol(std::string_view group, const nlohmann::json & record, std::int64_t time_ns) {
    stream_ << nlohmann::json{
        { "kind", "native_protocol" }, { "group", group }, { "record", record }, { "time_ns", time_ns }
    } << '\n';
}

void ExecutionTrace::queried(std::string_view group, std::size_t boundary, const nlohmann::json & result,
                                    const nlohmann::json & clocks, std::int64_t time_ns) {
    stream_ << nlohmann::json{
        { "kind", "state_query" }, { "group", group }, { "boundary", boundary }, { "result", result }, { "time_ns", time_ns },
        { "completion_reads", completion_reads_[boundary] }, { "clocks", clocks }
    } << '\n';
    completion_reads_.erase(boundary);
}

void ExecutionTrace::completion_read(std::string_view group, std::size_t boundary, std::string_view signal, std::size_t event,
                                    bool result, std::int64_t time_ns) {

    // Emit at the read itself: a later query result cannot reconstruct completion order at equal timestamps.
    stream_ << nlohmann::json{
        { "kind", "completion_read" }, { "group", group }, { "boundary", boundary }, { "signal", signal },
        { "event", event }, { "result", result }, { "time_ns", time_ns }
    } << '\n';
    ++completion_reads_[boundary];
}

void ExecutionTrace::notification_declared(const nlohmann::json & identity, std::size_t event) {
    stream_ << nlohmann::json{
        { "kind", "notification_declared" }, { "identity", identity }, { "event", event }
    } << '\n';
}

void ExecutionTrace::notification_ready(const nlohmann::json & identity, std::size_t event,
                                       std::span<const std::size_t> prerequisites, std::int64_t time_ns) {
    stream_ << nlohmann::json{
        { "kind", "notification_ready" }, { "identity", identity }, { "event", event }, { "time_ns", time_ns },
        { "prerequisites", std::vector<std::size_t>(prerequisites.begin(), prerequisites.end()) }
    } << '\n';
}

void ExecutionTrace::input_read(const nlohmann::json & identity, std::span<const std::size_t> requests, bool in_flow,
                                std::int64_t time_ns, std::optional<std::int64_t> poll_deadline_ns) {
    auto observed = identity;
    ContextValues::encode(observed, [this](const auto & value) { return context_value(value); });
    stream_ << nlohmann::json{
        { "kind", "input_read" }, { "identity", std::move(observed) }, { "time_ns", time_ns }, { "in_flow", in_flow },
        { "poll_deadline_ns", poll_deadline_ns ? nlohmann::json(*poll_deadline_ns) : nullptr },
        { "request_indices", std::vector<std::size_t>(requests.begin(), requests.end()) }
    } << '\n';
}

void ExecutionTrace::state_bound(std::size_t invocation, std::string_view pool, std::string_view field, std::string_view request, std::size_t slot) {
    stream_ << nlohmann::json{
        { "kind", "state_binding" }, { "invocation", invocation }, { "pool", pool }, { "field", field }, { "request", request }, { "slot", slot }
    } << '\n';
}

void ExecutionTrace::bound(std::size_t invocation, std::string_view pool, std::string_view request, std::optional<std::size_t> slot, std::int64_t begin,
                           std::span<const KvPageRange> ranges) {
    auto pages = nlohmann::json::array();
    for (const auto & range : ranges) pages.push_back({ range.page, range.begin, range.end });
    stream_ << nlohmann::json{
        {       "kind",                                       "binding" },
        { "invocation",                                      invocation },
        {       "pool",                                            pool },
        {    "request",                                         request },
        {       "slot", slot ? nlohmann::json(*slot) : nlohmann::json() },
        {      "begin",                                           begin },
        {     "ranges",                                           pages }
    } << '\n';
}

void ExecutionTrace::finish() {
    stream_ << nlohmann::json{
        { "kind", "end" }
    } << '\n';
    stream_.close();
}

} // namespace prologue
