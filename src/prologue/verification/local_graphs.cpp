#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "prologue/integrations/frameworks/sglang/templates.hpp"
#include "prologue/integrations/frameworks/sglang/workers.hpp"
#include "prologue/graph/graph.hpp"
#include "prologue/templates/library.hpp"
#include "prologue/templates/selection.hpp"

namespace {

nlohmann::json read_json(const char * path) {
    std::ifstream input(path);
    return nlohmann::json::parse(input);
}

// Only action metadata crosses this boundary; target work, ports and edges cannot enter the constructor.
class LocalGraphs {
public:
    LocalGraphs(const prologue::TemplateLibrary & library, nlohmann::json environment)
        : library_(library), selection_(library), environment_(std::move(environment)), worker_(library, environment_) {}

    prologue::CoarseGraph build(const prologue::EntrySpecification & action, const prologue::Parameters & parameters,
                             const nlohmann::json & work) const {
        const auto states = worker_.state_inputs(work.value("bindings", nlohmann::json::object()), work);
        auto entry = selection_.select(action.action, action.condition, parameters, states, work);
        auto graph_work = work;
        worker_.bind(entry.frame, "local", graph_work);
        prologue::GraphInvocation call{ action.action, action.condition, parameters, library_.resources, std::move(graph_work),
                                      "local/" + std::to_string(action.scope), environment_, library_.domain, {} };
        for (auto & resource : call.resources) resource.rank = environment_.at("parallel").at("rank").get<std::int64_t>();
        if (!states.empty()) {
            for (const auto & [name, state] : entry.frame.state_inputs()) call.state_inputs.emplace(name, states.at(name));
        }
        for (const auto & port : library_.definitions[entry.frame.definition].ports) {
            if (port.use == prologue::PortUse::Signal) call.signals.emplace(port.name, call.identity + "/signal/" + port.name);
        }
        bind(entry.frame, call);
        return library_.instantiate(entry, call);
    }

private:
    void bind(const prologue::TemplateFrame & frame, prologue::GraphInvocation & call) const {

        // Local checks need symbolic external slots, not the target's allocation history or other graphs.
        for (const auto & tensor : frame.tensors) {
            if (tensor && tensor->external) call.storage.try_emplace(tensor->storage, prologue::StorageBinding{ tensor->storage, 0 });
        }
        const auto & state = frame.state_input ? call.state_inputs.at(*frame.state_input) : frame.state;
        for (const auto & value : state.kv) call.pools.try_emplace(value.pool, value.pool);
        for (const auto & pool : state.slots) call.pools.try_emplace(pool, pool);
        for (const auto & child : frame.children) bind(child, call);
    }

    const prologue::TemplateLibrary & library_;
    prologue::TemplateSelection selection_;
    nlohmann::json environment_;
    prologue::sglang::WorkerBindings worker_;
};

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 4 && std::string_view(argv[1]) == "actions") {
            auto observed = prologue::CoarseGraph::read(read_json(argv[2]));
            const auto declarations = prologue::sglang::NativeTemplates::describe(std::move(observed));
            if (declarations.entries.empty()) throw std::invalid_argument("target contains no native actions");
            nlohmann::json actions = nlohmann::json::array();
            for (const auto & action : declarations.entries) {
                const auto & scope = declarations.graph.scopes[action.scope];
                const auto & input = scope.info;
                auto work = input.work;

                // Endpoint routes are native call arguments, independently recorded before the transfer executes.
                if (scope.communication && scope.communication->operation == "transfer") {
                    const auto & communication = *scope.communication;
                    work["transfer"] = { { "channel", communication.channel }, { "members", communication.members } };
                }
                actions.push_back({ { "scope", action.scope }, { "action", action.action }, { "condition", action.condition },
                                    { "parameters", input.parameters }, { "work", std::move(work) } });
            }

            // Export declared invocation inputs; generation runs without access to the observed graph.
            std::ofstream output(argv[3], std::ios::out | std::ios::noreplace);
            output.exceptions(std::ios::failbit | std::ios::badbit);
            output << nlohmann::json{ { "environment", declarations.graph.environment }, { "actions", std::move(actions) } }.dump() << '\n';
            output.close();
            return 0;
        }
        if (argc != 5 || std::string_view(argv[1]) != "expand")
            throw std::invalid_argument("prologue_local_graphs actions <observed-target.json> <action-inputs.json>\n"
                                        "prologue_local_graphs expand <source-libraries.json> <action-inputs.json> <local-graphs.jsonl>");
        std::map<std::size_t, prologue::TemplateLibrary> libraries;
        for (const auto & path : read_json(argv[2])) {
            libraries.emplace(libraries.size(), prologue::TemplateLibrary::read(read_json(path.get<std::string>().c_str())));
        }
        if (libraries.empty()) throw std::invalid_argument("local expansion requires a primary source library");

        // Assemble the same source candidates as serving; the first library owns this worker's action interface.
        prologue::sglang::NativeTemplates::share_closed_actions(libraries);
        std::vector<prologue::TemplateLibrary *> sources;
        for (auto & [_, library] : libraries) sources.push_back(&library);
        prologue::TemplateLibrary::link_local_implementations(sources);

        const auto inputs = read_json(argv[3]);
        const LocalGraphs graphs(libraries.at(0), inputs.at("environment"));
        std::ofstream output(argv[4], std::ios::out | std::ios::noreplace);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        std::size_t built = 0;
        for (const auto & input : inputs.at("actions")) {
            const prologue::EntrySpecification action{ input.at("scope").get<std::size_t>(), input.at("action"), input.at("condition") };
            nlohmann::json record{ { "kind", "local_graph" }, { "source_scope", action.scope } };
            try {

                // Invocation inputs own the initialized environment, as in the formal execution trace.
                record["graph"] = graphs.build(action, input.at("parameters").get<prologue::Parameters>(), input.at("work")).document(false);
                ++built;
            }
            catch (const std::exception & error) {
                record["error"] = error.what();
            }
            output << record.dump() << '\n';
        }
        output << nlohmann::json{ { "kind", "local_summary" }, { "attempted", inputs.at("actions").size() }, { "built", built } }.dump() << '\n';
        output.close();
        return built == inputs.at("actions").size() ? 0 : 1;
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
