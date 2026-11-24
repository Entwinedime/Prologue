#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
#include <unistd.h>

#include "prologue/integrations/frameworks/sglang/serving.hpp"
#include "prologue/integrations/frameworks/sglang/templates.hpp"
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

nlohmann::json read_json(const std::filesystem::path & path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot read input: " + path.string());
    stream.exceptions(std::ios::badbit);
    return nlohmann::json::parse(stream);
}

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

// Publishing a new artifact must neither expose a partial JSON nor replace an existing input or output.
class Publication {
public:
    explicit Publication(std::filesystem::path destination)
        : destination_(std::move(destination)),
          temporary_(destination_.string() + ".tmp." + std::to_string(getpid())) {
        if (!destination_.parent_path().empty()) std::filesystem::create_directories(destination_.parent_path());
        stream_.exceptions(std::ios::badbit | std::ios::failbit);
        stream_.open(temporary_, std::ios::out | std::ios::noreplace);
    }

    ~Publication() {
        std::error_code ignored;
        std::filesystem::remove(temporary_, ignored);
    }

    void write(const nlohmann::json & document) {
        stream_ << document << '\n';
        stream_.close();

        // A same-directory hard link is atomic and fails if the destination already exists, including symlinks.
        std::filesystem::create_hard_link(temporary_, destination_);
    }

private:
    std::filesystem::path destination_;
    std::filesystem::path temporary_;
    std::ofstream stream_;
};

constexpr std::string_view usage = "prologue project <native-facts.json> <coarse.json> [--evidence]\n"
                                   "prologue action-layouts <observed-coarse.json> <layouts.json>\n"
                                   "prologue templates <coarse.json> <templates.json> [--root-site <site>]\n"
                                   "prologue work-identities <observed-coarse.json> <work.json>\n"
                                   "prologue work-costs <observed-coarse.json> <costs.json> <services.json> [--bindings]\n"
                                   "prologue cost-points <points.json> <costs.json> <estimates.json> [--bindings]\n"
                                   "prologue link-templates <source-libraries.json> <output-directory>\n"
                                   "prologue query-template <templates.json> <query.json> <answer.json>\n"
                                   "prologue join-observation <worker.json> <observed.json> <candidate.json>\n"
                                   "prologue template-evidence <templates.json> <observed-templates.json> <measured-call.json> <evidence.json>\n"
                                   "prologue supplement-template <templates.json> <query.json> <evidence.json> <candidate.json>\n"
                                   "prologue instantiate <templates.json> <invocation.json> <coarse.json>\n"
                                   "prologue execute <coarse.json> <given-timing.json> <execution.json> [--trace <execution.jsonl>]\n"
                                   "prologue replay <fixed-execution.json> <services.json> <event-times.json>\n"
                                   "prologue serve <templates-directory> <task.json> <result.json> [--trace <execution.jsonl>] "
                                   "[--execution-inputs <conditions.json> | --costs <costs.json>]\n";

nlohmann::json work_identities(const prologue::sglang::TemplateDeclarations & source, const nlohmann::json & observation_window) {
    const auto observed = source.observed_work();
    prologue::WorkCoordinates coordinates(source.graph);
    auto features = nlohmann::json::array();
    auto intervals = nlohmann::json::array();
    for (std::size_t i = 0; i < observed.origins.size(); ++i) {
        const auto & work = source.graph.work[i];
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

nlohmann::json work_costs(const prologue::sglang::TemplateDeclarations & source, const nlohmann::json & costs, bool bindings) {
    const auto observed = source.observed_work();
    nlohmann::json missing;
    const prologue::WorkCosts service(observed.domain, observed.definitions, costs);
    auto result = prologue::service_estimates(service.timing(source.graph, observed.origins, &missing, bindings));
    result["domain"] = observed.domain;
    result["first_cost_gap"] = std::move(missing);
    return result;
}

std::int64_t nonnegative(const nlohmann::json & value) {
    if (!value.is_number_integer() || value.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("fixed execution coordinates must be nonnegative signed 64-bit integers");
    return value.get<std::int64_t>();
}

nlohmann::json replay(const nlohmann::json & plan, const nlohmann::json & scenarios) {
    std::vector<std::int64_t> releases;
    for (const auto & value : plan.at("releases_ns").get_ref<const nlohmann::json::array_t &>()) releases.push_back(nonnegative(value));
    std::vector<prologue::TimedDependency> edges;
    for (const auto name : { "dependencies", "work" }) {
        for (const auto & row : plan.at(name).get_ref<const nlohmann::json::array_t &>()) {
            if (!row.is_array() || row.size() != 3) throw std::invalid_argument("fixed dependency needs two endpoints and an immutable delay");
            edges.push_back({ static_cast<std::size_t>(nonnegative(row[0])), static_cast<std::size_t>(nonnegative(row[1])), nonnegative(row[2]) });
        }
    }

    auto times = nlohmann::json::object();
    auto work = std::span(edges).subspan(plan.at("dependencies").size());
    std::vector<std::int64_t> fixed_delays;
    for (const auto & edge : work) fixed_delays.push_back(edge.delay_ns);
    for (const auto & [name, values] : scenarios.get_ref<const nlohmann::json::object_t &>()) {
        if (!values.is_array() || values.size() != work.size()) throw std::invalid_argument("fixed replay needs one service value per declared work interval");
        for (std::size_t index = 0; index < work.size(); ++index) {
            const auto service = nonnegative(values[index]);
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
        else work_ns.push_back(prologue::read_parameters({ { "service_ns", value } }).at("service_ns"));
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
    result.domain = prologue::sglang::NativeTemplates::domain(result.environment);
    result.pools = source.at("pools").get<std::map<std::string, std::string>>();
    if (source.contains("signals")) result.signals = source.at("signals").get<std::map<std::string, std::string>>();
    if (source.contains("state_inputs")) {
        for (const auto & [name, value] : source.at("state_inputs").items()) result.state_inputs.emplace(name, prologue::StateBindings::read(value));
    }
    for (const auto & [name, storage] : source.at("storage").get_ref<const nlohmann::json::object_t &>()) {
        const auto version = prologue::read_parameters({
                                                         { "version", storage.at("version") }
        })
                                 .at("version");
        result.storage.emplace(name, prologue::StorageBinding{ storage.at("identity").get<std::string>(), version });
    }
    for (const auto & resource : source.at("resources").get_ref<const nlohmann::json::array_t &>())
        result.resources.push_back(prologue::Resource::read(resource));
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << usage;
            return 0;
        }
        if (argc < 2) throw std::invalid_argument(std::string(usage));

        const std::string_view command = argv[1];
        std::filesystem::path destination = argv[argc - 1];
        nlohmann::json output;
        if (command == "project" && (argc == 4 || (argc == 5 && std::string_view(argv[4]) == "--evidence"))) {
            destination = argv[3];
            auto observation = prologue::NativeObservation::read(read_json(argv[2]));
            prologue::ProjectionEvidence evidence;
            output = prologue::CoarseProjection::project(std::move(observation), argc == 5 ? &evidence : nullptr).document();
            if (argc == 5) output["evidence"] = evidence.document(observation);
        }
        else if (((command == "templates" || command == "work-identities") && argc == 4)
                 || (command == "templates" && argc == 6 && std::string_view(argv[4]) == "--root-site")) {
            destination = argv[3];
            const auto document = read_json(argv[2]);
            auto source = prologue::sglang::NativeTemplates::describe(prologue::CoarseGraph::read(document));
            if (argc == 6) {

                // Compile the observed root and its children without publishing incidental calls from the same window.
                std::erase_if(source.entries, [&](const auto & entry) { return source.graph.scopes.at(entry.scope).info.site != argv[5]; });
                if (source.entries.empty()) throw std::invalid_argument("requested root site has no observed template entry");
            }
            output = command == "templates" ? source.compile().document() : work_identities(source, document.at("evidence").at("window_ns"));
        }
        else if ((command == "work-costs" || command == "cost-points") && (argc == 5 || (argc == 6 && std::string_view(argv[5]) == "--bindings"))) {
            destination = argv[4];
            const auto input = read_json(argv[2]);
            const auto costs = read_json(argv[3]);
            if (command == "work-costs") {
                const auto source = prologue::sglang::NativeTemplates::describe(prologue::CoarseGraph::read(input));
                output = work_costs(source, costs, argc == 6);
            }
            else output = prologue::WorkCosts::evaluate(input, costs, argc == 6);
        }
        else if (command == "replay" && argc == 5) {
            output = replay(read_json(argv[2]), read_json(argv[3]));
        }
        else if (command == "action-layouts" && argc == 4) {
            const auto source = prologue::sglang::NativeTemplates::describe(prologue::CoarseGraph::read(read_json(argv[2])));
            const prologue::EntryExtraction extraction(source.graph);
            output = nlohmann::json::object();
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
        }
        else if (command == "link-templates" && argc == 4) {
            std::vector<prologue::TemplateLibrary> libraries;
            for (const auto & path : read_json(argv[2])) libraries.push_back(prologue::TemplateLibrary::read(read_json(path.get<std::string>())));
            if (libraries.empty()) throw std::invalid_argument("template linking requires a source library set");
            std::vector<prologue::TemplateLibrary *> sources;
            for (auto & library : libraries) sources.push_back(&library);
            prologue::TemplateLibrary::link_local_implementations(sources);
            const std::filesystem::path directory(argv[3]);
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
            std::cerr << "Created " << directory << '\n';
            return 0;
        }
        else if (command == "query-template" && argc == 5) {
            const auto library = prologue::TemplateLibrary::read(read_json(argv[2]));
            output = prologue::TemplateSelection(library).query(read_json(argv[3]));
        }
        else if (command == "template-evidence" && argc == 6) {
            const auto library = prologue::TemplateLibrary::read(read_json(argv[2]));
            const auto observed = prologue::TemplateLibrary::read(read_json(argv[3]));
            output = observed.evidence(read_json(argv[4]), library);
        }
        else if (command == "join-observation" && argc == 5) {
            auto worker = prologue::TemplateLibrary::read(read_json(argv[2]));
            worker.join_observation(prologue::TemplateLibrary::read(read_json(argv[3])));
            output = worker.document();
        }
        else if (command == "supplement-template" && argc == 6) {
            const auto library = prologue::TemplateLibrary::read(read_json(argv[2]));
            output = library.supplement(read_json(argv[3]), read_json(argv[4])).document();
        }
        else if (command == "instantiate" && argc == 5) {
            const auto library = prologue::TemplateLibrary::read(read_json(argv[2]));
            const auto call = invocation(read_json(argv[3]));
            const prologue::TemplateSelection selection(library);
            output = library.instantiate(selection.select(call.action, call.condition, call.parameters, call.state_inputs, call.work), call).document();
        }
        else if (command == "serve" && argc >= 5 && (argc - 5) % 2 == 0) {
            destination = argv[4];
            std::map<std::string_view, std::filesystem::path> options;
            for (int argument = 5; argument < argc; argument += 2) {
                const std::string_view option = argv[argument];
                if ((option != "--trace" && option != "--execution-inputs" && option != "--costs")
                    || !options.emplace(option, argv[argument + 1]).second)
                    throw std::invalid_argument("unknown or repeated serving option: " + std::string(option));
            }
            if (options.contains("--execution-inputs") && options.contains("--costs"))
                throw std::invalid_argument("select either measured execution inputs or independent costs");
            const auto task = read_json(argv[3]);
            for (const auto & [field, _] : task.items()) {
                if (field != "template_sources" && field != "instances" && field != "workload"
                    && field != "given_work_ns")
                    throw std::invalid_argument("unknown serving input field: " + field);
            }

            // Timing assets are immutable and selected at task startup; the core owns their decoding.
            const auto timing = options.contains("--execution-inputs") ? read_json(options.at("--execution-inputs")) : nlohmann::json();
            const auto costs = options.contains("--costs") ? read_json(options.at("--costs")) : nlohmann::json();
            if (options.contains("--execution-inputs") && !timing.is_object()) throw std::invalid_argument("execution inputs require an object");
            if (options.contains("--costs") && !costs.is_object()) throw std::invalid_argument("cost asset requires an object");
            std::unique_ptr<prologue::ExecutionTrace> trace;
            if (options.contains("--trace")) trace = std::make_unique<prologue::ExecutionTrace>(options.at("--trace"));
            StreamPeer peer;
            try {
                std::map<std::size_t, prologue::TemplateLibrary> libraries;
                PublishedTemplates published(argv[2]);
                for (const auto & header : task.at("template_sources").get_ref<const nlohmann::json::array_t &>()) {
                    const auto index = libraries.size();
                    libraries.emplace(index, published.read(index, header));
                }

                // Auxiliary bodies can supply closed actions or local children; participants come from the role binding.
                const auto imports = prologue::sglang::NativeTemplates::share_closed_actions(libraries);
                std::vector<prologue::TemplateLibrary *> sources;
                for (auto & [_, library] : libraries) sources.push_back(&library);
                prologue::TemplateLibrary::link_local_implementations(sources);
                output = prologue::sglang::serve(libraries, task, timing, costs, peer, trace.get(), trace.get(), trace.get());
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
        }
        else if (command == "execute" && (argc == 5 || (argc == 7 && std::string_view(argv[5]) == "--trace"))) {
            destination = argv[4];
            std::unique_ptr<prologue::ExecutionTrace> trace;
            if (argc == 7) trace = std::make_unique<prologue::ExecutionTrace>(argv[6]);
            output = execute(prologue::CoarseGraph::read(read_json(argv[2])), read_json(argv[3]), trace.get());
            if (trace) trace->finish();
        }
        else throw std::invalid_argument(std::string(usage));

        Publication(destination).write(output);
        std::cerr << "Created " << destination.string() << '\n';
        return 0;
    }
    catch (const prologue::TemplateGap & gap) {
        std::cerr << gap.details() << '\n';
        return 1;
    }
    catch (const std::exception & error) {
        std::cerr << "prologue: " << error.what() << '\n';
        return 1;
    }
}
