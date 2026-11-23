#include "prologue/app/control_channel.hpp"
#include "prologue/app/timing.hpp"
#include "prologue/core/performance/costs.hpp"
#include "prologue/core/serving/cost_timing.hpp"
#include "prologue/core/serving/simulation.hpp"
#include "prologue/core/serving/workload.hpp"
#include "prologue/core/structure/templates/compiler.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using nlohmann::json;
using prologue::serving::ActionTiming;
using prologue::serving::Request;
using prologue::serving::Stopped;
using prologue::serving::TimingSource;
using prologue::serving::Workload;

std::int64_t integer(const json& object, const char* key) {
    const auto& value = object.at(key);
    constexpr auto max_value = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<std::uint64_t>() > max_value)) {
        throw std::invalid_argument(std::string(key) + " must be a signed 64-bit integer");
    }
    return value.get<std::int64_t>();
}

json read_record() {
    std::string line;
    if (!std::getline(std::cin, line)) {
        throw std::invalid_argument("simulation channel closed before a reply");
    }
    return json::parse(line);
}

class StreamTiming final : public TimingSource {
public:
    ActionTiming for_action(const json& action, const json& graph) override {
        const auto& work = graph.at("work");
        std::cout << json{{"timing_query", {{"action", action}, {"work", work}}}} << std::endl;
        const auto reply = read_record();
        if (const auto failure = reply.find("failure"); failure != reply.end()) {
            throw Stopped(failure->at("status"), failure->at("reason"));
        }

        return prologue::app::read_timing(reply, work);
    }
};

Workload read_workload(const json& source) {
    if (!source.is_object() || source.size() != 2) {
        throw std::invalid_argument("workload fields are id and requests");
    }
    const auto& entries = source.at("requests");
    if (!entries.is_array()) {
        throw std::invalid_argument("requests must be an array");
    }

    std::vector<Request> requests;
    for (const auto& entry : entries) {
        if (!entry.is_object() || entry.size() != 4) {
            throw std::invalid_argument("request fields are id, arrival_ns, prompt_tokens and output_tokens");
        }
        requests.push_back({entry.at("id").get<std::string>(), integer(entry, "arrival_ns"),
                            integer(entry, "prompt_tokens"), integer(entry, "output_tokens")});
    }
    return Workload(source.at("id").get<std::string>(), std::move(requests));
}

json normalize_workloads(const json& source) {
    if (!source.is_array() || source.empty()) {
        throw std::invalid_argument("at least one workload is required");
    }

    auto workloads = json::array();
    std::unordered_set<std::string> identities;
    for (const auto& entry : source) {
        const auto workload = read_workload(entry);
        if (!identities.insert(workload.id()).second) {
            throw std::invalid_argument("duplicate workload ID " + workload.id());
        }

        auto requests = json::array();
        for (const auto& request : workload.requests()) {
            requests.push_back({{"id", request.id}, {"arrival_ns", request.arrival_ns},
                                {"prompt_tokens", request.prompt_tokens}, {"output_tokens", request.output_tokens}});
        }
        workloads.push_back({{"id", workload.id()}, {"requests", std::move(requests)}});
    }
    return workloads;
}

json work_coordinates(const json& batches) {
    auto result = json::array();
    for (const auto& batch : batches) {
        prologue::performance::CostQueries queries(batch.at("action"), batch.at("graph"));
        result.push_back({{"host", std::move(queries.host)}, {"device", std::move(queries.device)}});
    }
    return result;
}

json cost_quotes(const json& input) {
    const prologue::performance::CostCatalog costs(input.at("costs"));
    auto result = json::array();
    for (const auto& coordinate : input.at("coordinates")) {
        result.push_back(costs.quote(input.at("domain"), coordinate));
    }
    return result;
}

}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: prologue-core workloads|templates|simulate|predict|coordinates|quote|replay [socket]\n";
        return 2;
    }

    try {
        const std::string_view command(argv[1]);
        if (argc == 2 && command == "workloads") {
            std::cout << normalize_workloads(json::parse(std::cin)) << '\n';
        } else if (argc == 2 && command == "templates") {
            std::cout << prologue::structure::Compiler{}.compile(json::parse(std::cin)) << '\n';
        } else if (argc == 3 && command == "simulate") {
            const auto job = read_record();
            StreamTiming timing;
            prologue::serving::Simulation simulation(job, read_workload(job.at("workload")), timing);
            std::cout << json{{"result", prologue::app::serve_control(argv[2], simulation)}} << '\n';
        } else if (argc == 3 && command == "predict") {
            const auto job = json::parse(std::cin);
            const prologue::performance::CostCatalog costs(job.at("costs"));
            prologue::serving::CostTiming timing(job.at("cost_domain"), costs);
            prologue::serving::Simulation simulation(job, read_workload(job.at("workload")), timing);
            std::cout << prologue::app::serve_control(argv[2], simulation) << '\n';
        } else if (argc == 2 && command == "coordinates") {
            std::cout << work_coordinates(json::parse(std::cin)) << '\n';
        } else if (argc == 2 && command == "quote") {
            std::cout << cost_quotes(json::parse(std::cin)) << '\n';
        } else if (argc == 2 && command == "replay") {
            std::cout << prologue::app::replay(json::parse(std::cin)) << '\n';
        } else {
            throw std::invalid_argument("unknown core command " + std::string(command));
        }
    } catch (const prologue::performance::CostGap& gap) {
        std::cout << gap.detail << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
