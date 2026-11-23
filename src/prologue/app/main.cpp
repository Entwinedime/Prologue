#include "prologue/app/control_channel.hpp"
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
using prologue::serving::Nanoseconds;
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
    ActionTiming for_action(const json& action, const json& work) override {
        std::cout << json{{"timing_query", {{"action", action}, {"work", work}}}} << std::endl;
        const auto reply = read_record();
        if (const auto failure = reply.find("failure"); failure != reply.end()) {
            throw Stopped(failure->at("status"), failure->at("reason"));
        }

        const auto& samples = reply.at("work");
        if (!samples.is_array() || samples.size() != work.size()) {
            throw Stopped("timing_gap", "one measured interval is required per selected work");
        }

        ActionTiming timing{{}, elapsed(reply, "model_return_ns"), elapsed(reply, "result_finished_ns"), {}};
        for (std::size_t index = 0; index < work.size(); ++index) {
            const auto& sample = samples.at(index);
            timing.work.push_back({work.at(index).at("lane"), elapsed(sample, "submitted_after_ns"),
                                   elapsed(sample, "duration_ns")});
        }
        const auto& publications = reply.at("publications_after_ns");
        for (const auto& entry : publications.items()) {
            timing.publications.emplace(entry.key(), elapsed(publications, entry.key().c_str()));
        }
        return timing;
    }

private:
    static Nanoseconds elapsed(const json& row, const char* key) {
        if (!row.contains(key)) {
            throw Stopped("timing_gap", std::string("missing measured ") + key);
        }
        const auto result = integer(row, key);
        if (result < 0) {
            throw Stopped("timing_gap", std::string("negative measured ") + key);
        }
        return result;
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

}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: prologue-core workloads|templates|simulate [socket] < input.json\n";
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
        } else {
            throw std::invalid_argument("unknown core command " + std::string(command));
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
