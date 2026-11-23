#include "prologue/core/serving/workload.hpp"

#include <nlohmann/json.hpp>

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
using prologue::serving::Request;
using prologue::serving::Workload;

std::int64_t integer(const json& object, const char* key) {
    const auto& value = object.at(key);
    constexpr auto max_value = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<std::uint64_t>() > max_value)) {
        throw std::invalid_argument(std::string(key) + " must be a signed 64-bit integer");
    }
    return value.get<std::int64_t>();
}

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
    if (argc != 2 || std::string_view(argv[1]) != "workloads") {
        std::cerr << "Usage: prologue-core workloads < workloads.json\n";
        return 2;
    }

    try {
        std::cout << normalize_workloads(json::parse(std::cin)) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
