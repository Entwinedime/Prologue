#include "prologue/core/serving/workload.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace prologue::serving {

Workload::Workload(std::string id, std::vector<Request> requests)
    : id_(std::move(id)), requests_(std::move(requests)) {
    if (id_.empty() || requests_.empty()) {
        throw std::invalid_argument("workload needs an ID and at least one request");
    }

    std::unordered_set<std::string> identities;
    constexpr auto max_tokens = std::numeric_limits<std::int64_t>::max();
    for (const auto& request : requests_) {
        if (request.id.empty() || !identities.insert(request.id).second) {
            throw std::invalid_argument("workload " + id_ + " requires unique nonempty request IDs");
        }
        if (request.arrival_ns < 0 || request.prompt_tokens < 1 || request.output_tokens < 1) {
            throw std::invalid_argument("invalid arrival or token count for " + id_ + "/" + request.id);
        }

        // Prefill produces the first token, so only the remaining outputs extend the context.
        if (request.output_tokens - 1 > max_tokens - request.prompt_tokens) {
            throw std::invalid_argument("token count overflows for " + id_ + "/" + request.id);
        }
    }

    // Equal arrivals retain their order in the supplied workload.
    std::ranges::stable_sort(requests_, {}, &Request::arrival_ns);
}

const std::string& Workload::id() const {
    return id_;
}

const std::vector<Request>& Workload::requests() const {
    return requests_;
}

}
