#include "core/contracts/serving.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace prologue {

std::int64_t read_integer(const nlohmann::json & value, std::string_view name) {
    if (!value.is_number_integer() || (value.is_number_unsigned() && !std::in_range<std::int64_t>(value.get<std::uint64_t>())))
        throw std::invalid_argument("parameter must be a signed 64-bit integer: " + std::string(name));
    return value.get<std::int64_t>();
}

std::int64_t read_nonnegative_integer(const nlohmann::json & value, std::string_view name) {
    const auto integer = read_integer(value, name);
    if (integer < 0) throw std::invalid_argument("parameter must be nonnegative: " + std::string(name));
    return integer;
}

std::vector<Request> Request::read(const nlohmann::json & source) {
    if (!source.is_array() || source.empty()) throw std::invalid_argument("serving needs a nonempty request array");
    constexpr std::array fields{ "id", "arrival_ns", "prompt_tokens", "output_tokens", "prefix_id" };
    std::vector<Request> requests;
    for (const auto & request : source) {
        for (const auto & [name, _] : request.items()) {
            if (std::ranges::find(fields, name) == fields.end()) throw std::invalid_argument("request has unsupported field: " + name);
        }
        requests.push_back({ request.at("id").get<std::string>(), read_integer(request.at("arrival_ns"), "arrival_ns"),
                             read_integer(request.at("prompt_tokens"), "prompt_tokens"), read_integer(request.at("output_tokens"), "output_tokens") });
        if (request.contains("prefix_id")) requests.back().prefix_id = request.at("prefix_id").get<std::string>();
    }

    // Stable arrival order gives both simulation and recorded batches the same request identities.
    std::stable_sort(requests.begin(), requests.end(), [](const Request & left, const Request & right) { return left.arrival_ns < right.arrival_ns; });
    std::set<std::string> identities;
    for (const auto & request : requests) {
        if (request.id.empty() || !identities.insert(request.id).second) throw std::invalid_argument("request IDs must be nonempty and unique");
        if (request.arrival_ns < 0 || request.prompt_tokens <= 0 || request.output_tokens <= 0)
            throw std::invalid_argument("request needs nonnegative arrival and positive token lengths: " + request.id);
        if (request.output_tokens - 1 > std::numeric_limits<std::int64_t>::max() - request.prompt_tokens)
            throw std::invalid_argument("request token length overflows: " + request.id);
    }
    return requests;
}

ServingSettings::ServingSettings(std::int64_t requests, std::int64_t tokens, std::int64_t page, std::int64_t prefill)
    : max_running_requests(requests),
      max_total_tokens(tokens),
      page_size(page),
      max_prefill_tokens(prefill) {
    if (requests <= 0 || page <= 0 || tokens < page || prefill <= 0)
        throw std::invalid_argument("serving needs positive request/prefill capacity and at least one KV page");
}

} // namespace prologue
