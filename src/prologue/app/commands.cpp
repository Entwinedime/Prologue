#include "prologue/app/commands.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "prologue/contracts/json_io.hpp"
#include "prologue/contracts/serving.hpp"
#include "prologue/integrations/frameworks/sglang/cpp/serving.hpp"
#include "prologue/integrations/frameworks/sglang/cpp/templates.hpp"
#include "prologue/app/execution_trace.hpp"
#include "prologue/serving/controller/control_peer.hpp"
#include "prologue/serving/execution/timeline.hpp"
#include "prologue/serving/execution/timing.hpp"
#include "prologue/graph/observation.hpp"
#include "prologue/graph/types.hpp"
#include "prologue/performance/work_costs.hpp"
#include "prologue/performance/work_features.hpp"
#include "prologue/templates/entries.hpp"
#include "prologue/graph/projection.hpp"
#include "prologue/templates/production.hpp"
#include "prologue/templates/library.hpp"
#include "prologue/templates/selection.hpp"

namespace {

class StreamPeer final : public prologue::ControlPeer {
public:
    nlohmann::json receive() override {
        std::string line;
        if (!std::getline(std::cin, line)) throw std::runtime_error("native control endpoint closed");

        // Keep unexpected native output diagnosable without accepting it as a control message.
        try {
            return nlohmann::json::parse(line);
        }
        catch (const nlohmann::json::parse_error & error) {
            throw std::runtime_error("native control sent an invalid JSON frame: " + line.substr(0, 160) + " (" + error.what() + ")");
        }
    }

    void send(const nlohmann::json & message) override {
        std::cout << message << '\n' << std::flush;
        if (!std::cout) throw std::runtime_error("native control endpoint cannot receive");
    }
};

using prologue::Publication;
using prologue::read_json;

class PublishedTemplates {
public:
    explicit PublishedTemplates(std::filesystem::path directory) : directory_(std::move(directory)) {}

    prologue::TemplateLibrary read(std::size_t index, const nlohmann::json & header) {
        auto body = read_json(directory_ / (std::to_string(index) + ".json"));
        for (auto & declaration : body.at("definitions")) {
            const auto reference = declaration.find("object");
            if (reference == declaration.end()) continue;
            const auto & key = reference->get_ref<const std::string &>();
            if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != std::string::npos)
                throw std::invalid_argument("invalid published template definition key");

            // A worker keeps its own frame positions even when another worker uses the same body.
            auto [found, inserted] = definitions_.try_emplace(key);
            if (inserted) found->second = read_json(directory_ / "definitions" / (key + ".json"));
            auto id = declaration.at("id");
            declaration = found->second;
            declaration["id"] = std::move(id);
        }
        return prologue::TemplateLibrary::read(body, &header);
    }

private:
    std::filesystem::path directory_;
    std::map<std::string, nlohmann::json> definitions_;
};

using namespace prologue;
using Json = nlohmann::json;

struct Framework {
    TemplateDeclarations (*describe)(CoarseGraph);
    TemplateLibrary (*compile)(const TemplateDeclarations &);
    ObservedTemplateWork (*observed_work)(const TemplateDeclarations &);
    Json (*domain)(const Json &);
    Json (*link)(std::map<std::size_t, TemplateLibrary> &);
    Json (*serve)(const std::map<std::size_t, TemplateLibrary> &, const Json &, const Json &, const Json &,
                  ControlPeer &, ExecutionObserver *, ResourceObserver *, ControlObserver *);
};

const Framework & framework(const Json & environment) {
    static const std::map<std::string_view, Framework> adapters{
        {"sglang", {sglang::NativeTemplates::describe, sglang::NativeTemplates::compile, sglang::NativeTemplates::observed_work,
                    sglang::NativeTemplates::domain, sglang::link_serving_libraries, sglang::serve}}
    };
    const auto & name = environment.at("framework").get_ref<const std::string &>();
    const auto found = adapters.find(name);
    if (found == adapters.end()) throw std::invalid_argument("unsupported serving framework: " + name);
    return found->second;
}

struct ObservedSource {
    explicit ObservedSource(Json input) : adapter(framework(input.at("environment"))), declarations(adapter.describe(CoarseGraph::read(input))) {
        if (input.contains("evidence")) window = input.at("evidence").value("window_ns", Json(nullptr));
    }

    const Framework & adapter;
    TemplateDeclarations declarations;
    Json window;
};

nlohmann::json work_identities(const ObservedSource & source, const nlohmann::json & observation_window) {
    const auto observed = source.adapter.observed_work(source.declarations);
    prologue::WorkCoordinates coordinates(source.declarations.graph);
    auto features = nlohmann::json::array();
    auto intervals = nlohmann::json::array();
    for (std::size_t i = 0; i < observed.origins.size(); ++i) {
        const auto & work = source.declarations.graph.work[i];
        const auto & origin = observed.origins[i];
        const auto point = coordinates.index(work);
        if (point == features.size()) features.push_back(coordinates[point].document());
        intervals.push_back({work.start, work.finish, origin.definition, origin.interval, point});
    }

    auto definitions = nlohmann::json::array();
    for (const auto & definition : observed.definitions) definitions.push_back(definition.document());
    return {{"domain", observed.domain}, {"window_ns", observation_window},
            {"definitions", std::move(definitions)}, {"features", std::move(features)}, {"intervals", std::move(intervals)}};
}

nlohmann::json work_costs(const ObservedSource & source, const nlohmann::json & costs, bool bindings) {
    const auto observed = source.adapter.observed_work(source.declarations);
    nlohmann::json missing;
    const prologue::WorkCosts service(observed.domain, observed.definitions, costs);
    auto result = prologue::service_estimates(service.timing(source.declarations.graph, observed.origins, &missing, bindings));
    result["domain"] = observed.domain;
    result["first_cost_gap"] = std::move(missing);
    return result;
}

nlohmann::json replay(const nlohmann::json & plan, const nlohmann::json & scenarios) {
    std::vector<std::int64_t> releases;
    for (const auto & value : plan.at("releases_ns").get_ref<const nlohmann::json::array_t &>()) releases.push_back(prologue::read_nonnegative_integer(value, "release time"));
    std::vector<prologue::TimedDependency> edges;
    for (const auto name : { "dependencies", "work" }) {
        for (const auto & row : plan.at(name).get_ref<const nlohmann::json::array_t &>()) {
            if (!row.is_array() || row.size() != 3) throw std::invalid_argument("fixed dependency needs two endpoints and an immutable delay");
            edges.push_back({ static_cast<std::size_t>(prologue::read_nonnegative_integer(row[0], "source event")), static_cast<std::size_t>(prologue::read_nonnegative_integer(row[1], "target event")), prologue::read_nonnegative_integer(row[2], "fixed delay") });
        }
    }

    auto times = nlohmann::json::object();
    auto work = std::span(edges).subspan(plan.at("dependencies").size());
    std::vector<std::int64_t> fixed_delays;
    for (const auto & edge : work) fixed_delays.push_back(edge.delay_ns);
    for (const auto & [name, values] : scenarios.get_ref<const nlohmann::json::object_t &>()) {
        if (!values.is_array() || values.size() != work.size()) throw std::invalid_argument("fixed replay needs one service value per declared work interval");
        for (std::size_t index = 0; index < work.size(); ++index) {
            const auto service = prologue::read_nonnegative_integer(values[index], "service duration");
            if (service > std::numeric_limits<std::int64_t>::max() - fixed_delays[index])
                throw std::overflow_error("fixed work delay exceeds int64 nanoseconds");
            work[index].delay_ns = fixed_delays[index] + service;
        }
        times[name] = prologue::Timeline::replay(releases, edges);
    }
    return {{ "event_times_ns", std::move(times) }};
}

nlohmann::json execute(const prologue::CoarseGraph & graph, const nlohmann::json & timing, prologue::ExecutionObserver * observer) {
    std::vector<prologue::ServiceDuration> work_ns;
    for (const auto & value : timing.at("work_ns").get_ref<const nlohmann::json::array_t &>()) {
        if (value.is_null()) work_ns.push_back(std::nullopt);
        else work_ns.push_back(prologue::read_integer(value, "service_ns"));
    }
    std::vector<prologue::BoundaryCondition> boundaries;
    if (timing.contains("boundaries")) {
        for (const auto & value : timing.at("boundaries").get_ref<const nlohmann::json::array_t &>()) {
            const auto fields = prologue::read_parameters(value);
            if (fields.size() != 3 || fields.at("event") < 0)
                throw std::invalid_argument("completion condition requires event, begin_ns and end_ns");
            boundaries.push_back({ static_cast<std::size_t>(fields.at("event")), prologue::TimeWindow{ fields.at("begin_ns"), fields.at("end_ns") } });
        }
    }

    // This command executes supplied dependencies; it does not allocate state or choose serving actions.
    if (timing.at("initial_inputs") != "ready") throw std::invalid_argument("graph execution requires explicitly ready initial inputs");
    auto supplied = prologue::GraphTiming::conditioned(graph, std::move(work_ns), std::move(boundaries));
    prologue::Timeline timeline(graph.resources, observer);
    const auto events = timeline.insert(graph, supplied, {});
    std::size_t completed = 0;
    while (timeline.advance()) ++completed;
    if (!timeline.done()) throw std::runtime_error("graph execution stalled with " + std::to_string(events.size() - completed) + " unfinished boundaries");
    const auto conditioned = std::ranges::count_if(supplied.work_ns, [](const auto & value) { return !value; });
    nlohmann::json result{
        {   "duration_basis", conditioned ? "given_service_and_completion_conditions" : "given_service_durations" },
        { "given_service_durations", supplied.work_ns.size() - static_cast<std::size_t>(conditioned) },
        { "completion_conditioned_work", conditioned },
        {   "initial_inputs",                                                                    "ready" },
        {     "completed_ns",                                                          timeline.now_ns() },
        { "completed_events",                                                                  completed },
        {   "work_intervals",                                                          graph.work.size() }
    };
    if (!supplied.boundaries.empty()) {
        result["boundary_events"] = supplied.boundaries.size();
        result["completion_time_selection"] = "earliest_causal_time_within_supplied_windows";
    }
    return result;
}

prologue::GraphInvocation invocation(const nlohmann::json & source) {
    prologue::GraphInvocation result;
    result.action = source.at("action").get<std::string>();
    result.condition = source.at("condition");
    result.parameters = prologue::read_parameters(source.at("parameters"));
    result.identity = source.at("identity").get<std::string>();
    result.work = source.at("work");
    result.environment = source.at("environment");
    result.domain = framework(result.environment).domain(result.environment);
    result.pools = source.at("pools").get<std::map<std::string, std::string>>();
    if (source.contains("signals")) result.signals = source.at("signals").get<std::map<std::string, std::string>>();
    if (source.contains("state_inputs")) {
        for (const auto & [name, value] : source.at("state_inputs").items()) result.state_inputs.emplace(name, prologue::StateBindings::read(value));
    }
    for (const auto & [name, storage] : source.at("storage").get_ref<const nlohmann::json::object_t &>()) {
        const auto version = prologue::read_integer(storage.at("version"), "version");
        result.storage.emplace(name, prologue::StorageBinding{ storage.at("identity").get<std::string>(), version });
    }
    for (const auto & resource : source.at("resources").get_ref<const nlohmann::json::array_t &>())
        result.resources.push_back(prologue::Resource::read(resource));
    return result;
}

struct Arguments {
    std::span<char * const> files;
    std::map<std::string_view, const char *> options;
};

using Result = std::optional<Json>;

Result project(const Arguments & args) {
    auto projected = prologue::CoarseProjection::project(prologue::NativeObservation::read(read_json(args.files[0])), args.options.contains("--evidence"));
    auto output = projected.graph.document();
    if (projected.evidence) output["evidence"] = std::move(*projected.evidence);
    return output;
}

Result templates_command(const Arguments & args) {
    ObservedSource source(read_json(args.files[0]));
    if (const auto root = args.options.find("--root-site"); root != args.options.end()) {
        auto & [graph, entries] = source.declarations;
        std::erase_if(entries, [&](const auto & entry) { return graph.scopes.at(entry.scope).info.site != root->second; });
        if (entries.empty()) throw std::invalid_argument("requested root site has no observed template entry");
    }
    return source.adapter.compile(source.declarations).document();
}

Result identities_command(const Arguments & args) {
    const ObservedSource source(read_json(args.files[0]));
    if (source.window.is_null()) throw std::invalid_argument("work identities require an observation window");
    return work_identities(source, source.window);
}

Result costs_command(const Arguments & args) {
    return work_costs(ObservedSource(read_json(args.files[0])), read_json(args.files[1]), args.options.contains("--bindings"));
}

Result points_command(const Arguments & args) {
    return WorkCosts::evaluate(read_json(args.files[0]), read_json(args.files[1]), args.options.contains("--bindings"));
}

Result action_layouts(const Arguments & args) {
    const ObservedSource observed(read_json(args.files[0]));
    const auto & source = observed.declarations;
    const prologue::EntryExtraction extraction(source.graph);
    auto output = Json::object();
    for (const auto & entry : source.entries) {
        std::vector<std::size_t> origins;
        const auto action = extraction.extract(entry.scope, &origins);
        auto layout = action.action_layout();
        for (auto & event : layout.events) event = origins[event];
        auto tails = nlohmann::json::array();
        for (auto & tail : action.action_host_tails())
            tails.push_back({{"identity", std::move(tail.identity)}, {"event", origins[tail.event]}});
        output[std::to_string(entry.scope)] = {{"identity", std::move(layout.identity)},
            {"events", std::move(layout.events)}, {"host_tails", std::move(tails)}};
    }
    return output;
}

Result link_templates(const Arguments & args) {
    std::vector<prologue::TemplateLibrary> libraries;
    for (const auto & path : read_json(args.files[0])) libraries.push_back(prologue::TemplateLibrary::read(read_json(path.get<std::string>())));
    if (libraries.empty()) throw std::invalid_argument("template linking requires a source library set");
    std::vector<prologue::TemplateLibrary *> sources;
    for (auto & library : libraries) sources.push_back(&library);
    prologue::TemplateLibrary::link_local_implementations(sources);
    const std::filesystem::path directory(args.files[1]);
    if (!std::filesystem::create_directory(directory)) throw std::invalid_argument("template output directory already exists");
    try {
        for (const auto & library : libraries)
            Publication(directory / (std::to_string(library.resources.at(0).rank) + ".json")).write(library.document());
    }
    catch (...) {

        // This invocation created the directory; a failed set must not leave apparently complete assets behind.
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
        throw;
    }
    return std::nullopt;
}

Result query_template(const Arguments & args) {
    const auto library = TemplateLibrary::read(read_json(args.files[0]));
    return TemplateSelection(library).query(read_json(args.files[1]));
}

Result template_evidence(const Arguments & args) {
    const auto library = TemplateLibrary::read(read_json(args.files[0]));
    const auto observed = TemplateLibrary::read(read_json(args.files[1]));
    return observed.evidence(read_json(args.files[2]), library);
}

Result join_observation(const Arguments & args) {
    auto worker = TemplateLibrary::read(read_json(args.files[0]));
    worker.join_observation(TemplateLibrary::read(read_json(args.files[1])));
    return worker.document();
}

Result supplement_template(const Arguments & args) {
    const auto library = TemplateLibrary::read(read_json(args.files[0]));
    return library.supplement(read_json(args.files[1]), read_json(args.files[2])).document();
}

Result instantiate(const Arguments & args) {
    const auto library = TemplateLibrary::read(read_json(args.files[0]));
    const auto call = invocation(read_json(args.files[1]));
    const TemplateSelection selection(library);
    return library.instantiate(selection.select(call.action, call.condition, call.parameters, call.state_inputs, call.work), call).document();
}

Result replay_command(const Arguments & args) {
    return replay(read_json(args.files[0]), read_json(args.files[1]));
}

Result execute_command(const Arguments & args) {
    std::unique_ptr<ExecutionTrace> trace;
    if (args.options.contains("--trace")) trace = std::make_unique<ExecutionTrace>(args.options.at("--trace"));
    auto output = execute(CoarseGraph::read(read_json(args.files[0])), read_json(args.files[1]), trace.get());
    if (trace) trace->finish();
    return output;
}

Result serve_command(const Arguments & args) {
    const std::filesystem::path destination = args.files.back();
    Json output;
    if (args.options.contains("--execution-inputs") && args.options.contains("--costs"))
        throw std::invalid_argument("select either measured execution inputs or independent costs");
    const auto task = read_json(args.files[1]);
    for (const auto & [field, _] : task.items()) {
        if (field != "template_sources" && field != "instances" && field != "workload"
            && field != "given_work_ns")
            throw std::invalid_argument("unknown serving input field: " + field);
    }

    // Timing assets are immutable and selected at task startup; the core owns their decoding.
    const auto timing = args.options.contains("--execution-inputs") ? read_json(args.options.at("--execution-inputs")) : nlohmann::json();
    const auto costs = args.options.contains("--costs") ? read_json(args.options.at("--costs")) : nlohmann::json();
    if (args.options.contains("--execution-inputs") && !timing.is_object()) throw std::invalid_argument("execution inputs require an object");
    if (args.options.contains("--costs") && !costs.is_object()) throw std::invalid_argument("cost asset requires an object");
    std::unique_ptr<prologue::ExecutionTrace> trace;
    if (args.options.contains("--trace")) trace = std::make_unique<prologue::ExecutionTrace>(args.options.at("--trace"));
    StreamPeer peer;
    try {
        std::map<std::size_t, prologue::TemplateLibrary> libraries;
        PublishedTemplates published(args.files[0]);
        for (const auto & header : task.at("template_sources").get_ref<const nlohmann::json::array_t &>()) {
            const auto index = libraries.size();
            libraries.emplace(index, published.read(index, header));
        }

        // Auxiliary bodies can supply closed actions or local children; participants come from the role binding.
        if (libraries.empty()) throw std::invalid_argument("serving needs a template source");
        const auto & adapter = framework(libraries.begin()->second.domain);
        for (const auto & [_, library] : libraries) {
            if (&framework(library.domain) != &adapter) throw std::invalid_argument("serving libraries must use the same framework");
        }
        const auto imports = adapter.link(libraries);
        output = adapter.serve(libraries, task, timing, costs, peer, trace.get(), trace.get(), trace.get());
        output["template_imports"] = imports;
    }
    catch (const prologue::CostGap & gap) {
        Publication(destination.parent_path() / "cost-gap.json").write(gap.details());
        throw;
    }
    catch (const prologue::TemplateGap & gap) {
        Publication(destination.parent_path() / "template-gap.json").write(gap.details());
        throw;
    }
    if (trace) trace->finish();
    return output;
}

struct Option {
    std::string_view name;
    bool takes_value;
};

constexpr Option evidence[]{{"--evidence", false}}, root_site[]{{"--root-site", true}}, bindings[]{{"--bindings", false}};
constexpr Option trace[]{{"--trace", true}}, serving[]{{"--trace", true}, {"--execution-inputs", true}, {"--costs", true}};

struct Command {
    std::string_view name;
    std::string_view synopsis;
    std::size_t files;
    Result (*run)(const Arguments &);
    std::span<const Option> options{};
};

constexpr Command commands[]{
    {"project", "<native-facts.json> <coarse.json> [--evidence]", 2, project, evidence},
    {"action-layouts", "<observed-coarse.json> <layouts.json>", 2, action_layouts},
    {"templates", "<coarse.json> <templates.json> [--root-site <site>]", 2, templates_command, root_site},
    {"work-identities", "<observed-coarse.json> <work.json>", 2, identities_command},
    {"work-costs", "<observed-coarse.json> <costs.json> <services.json> [--bindings]", 3, costs_command, bindings},
    {"cost-points", "<points.json> <costs.json> <estimates.json> [--bindings]", 3, points_command, bindings},
    {"link-templates", "<source-libraries.json> <output-directory>", 2, link_templates},
    {"query-template", "<templates.json> <query.json> <answer.json>", 3, query_template},
    {"join-observation", "<worker.json> <observed.json> <candidate.json>", 3, join_observation},
    {"template-evidence", "<templates.json> <observed-templates.json> <measured-call.json> <evidence.json>", 4, template_evidence},
    {"supplement-template", "<templates.json> <query.json> <evidence.json> <candidate.json>", 4, supplement_template},
    {"instantiate", "<templates.json> <invocation.json> <coarse.json>", 3, instantiate},
    {"execute", "<coarse.json> <given-timing.json> <execution.json> [--trace <execution.jsonl>]", 3, execute_command, trace},
    {"replay", "<fixed-execution.json> <services.json> <event-times.json>", 3, replay_command},
    {"serve", "<templates-directory> <task.json> <result.json> [--trace <execution.jsonl>] [--execution-inputs <conditions.json> | --costs <costs.json>]", 3, serve_command, serving},
};

std::string help(const Command & command) {
    return "prologue " + std::string(command.name) + " " + std::string(command.synopsis) + "\n";
}

std::string usage() {
    std::string result;
    for (const auto & command : commands) result += help(command);
    return result;
}

} // namespace

namespace prologue {

void run_commands(std::span<char * const> words) {
    if (words.empty()) throw std::invalid_argument(usage());
    const std::string_view name(words.front());
    if (name == "--help" && words.size() == 1) {
        std::cout << usage();
        return;
    }
    const auto command = std::ranges::find(commands, name, &Command::name);
    if (command == std::end(commands)) throw std::invalid_argument(usage());
    words = words.subspan(1);
    if (words.size() == 1 && std::string_view(words.front()) == "--help") {
        std::cout << help(*command);
        return;
    }
    if (words.size() < command->files) throw std::invalid_argument(help(*command));
    Arguments args{words.first(command->files), {}};
    for (std::size_t i = command->files; i < words.size(); ++i) {
        const std::string_view key(words[i]);
        const auto option = std::ranges::find(command->options, key, &Option::name);
        if (option == command->options.end() || (option->takes_value && i + 1 == words.size()))
            throw std::invalid_argument("invalid option " + std::string(key) + "\n" + help(*command));
        const char * value = option->takes_value ? words[++i] : nullptr;
        if (!args.options.emplace(key, value).second) throw std::invalid_argument("repeated option: " + std::string(key));
    }

    const std::filesystem::path destination(args.files.back());
    if (const auto result = command->run(args)) {
        Publication(destination).write(*result);
        std::cerr << "Created " << destination.string() << '\n';
    } else {
        std::cerr << "Created " << destination << '\n';
    }
}

} // namespace prologue
