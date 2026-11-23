#include "prologue/core/serving/resources.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace prologue::serving {

Resources::Resources(const Workload& workload, ResourceLimits limits) : limits_(limits) {
    if (limits.page_size < 1 || limits.slots < 1 || limits.kv_tokens < limits.page_size) {
        throw std::invalid_argument("invalid resource capacity");
    }

    free_slots_.resize(static_cast<std::size_t>(limits.slots));
    free_pages_.resize(static_cast<std::size_t>(limits.kv_tokens / limits.page_size));
    page_in_use_.resize(free_pages_.size());
    std::iota(free_slots_.begin(), free_slots_.end(), std::int64_t{0});
    std::iota(free_pages_.begin(), free_pages_.end(), std::int64_t{0});
    for (const auto& request : workload.requests()) {
        residents_.emplace(request.id, Residence{});
    }
}

std::int64_t Resources::available_slots() const {
    return static_cast<std::int64_t>(free_slots_.size());
}

std::int64_t Resources::available_tokens() const {
    return static_cast<std::int64_t>(free_pages_.size() + released_pages_.size()) * limits_.page_size;
}

bool Resources::drained() const {
    return available_slots() == limits_.slots && free_pages_.size() + released_pages_.size() == page_in_use_.size();
}

const std::deque<std::int64_t>& Resources::free_slots() const {
    return free_slots_;
}

const Residence& Resources::residence(const std::string& request) const {
    return residents_.at(request);
}

std::vector<std::int64_t> Resources::bind_slots(const std::vector<std::string>& requests) {
    std::unordered_set<std::string> distinct;
    std::size_t required = 0;
    for (const auto& request : requests) {
        if (!distinct.insert(request).second) {
            throw std::invalid_argument("slot request repeats " + request);
        }
        required += !residence(request).slot.has_value();
    }
    if (required > free_slots_.size()) {
        throw std::invalid_argument("request slots exhausted");
    }

    std::vector<std::int64_t> slots;
    for (const auto& request : requests) {
        auto& resident = residents_.at(request);
        if (!resident.slot) {
            resident.slot = free_slots_.front();
            free_slots_.pop_front();
        }
        slots.push_back(*resident.slot);
    }
    return slots;
}

std::int64_t Resources::pages_for(std::int64_t tokens) const {
    return tokens == 0 ? 0 : (tokens - 1) / limits_.page_size + 1;
}

std::vector<std::int64_t> Resources::grow(const std::vector<KvGrowth>& members) {
    std::unordered_set<std::string> distinct;
    const auto total = static_cast<std::int64_t>(free_pages_.size() + released_pages_.size());
    auto available = total;
    for (const auto& growth : members) {
        const auto& resident = residence(growth.request);
        if (!distinct.insert(growth.request).second || !resident.slot || growth.begin != resident.tokens ||
            growth.end <= growth.begin) {
            throw std::invalid_argument("invalid KV extent for " + growth.request);
        }
        const auto required = pages_for(growth.end) - pages_for(growth.begin);
        if (required > available) {
            throw std::invalid_argument("KV pages exhausted");
        }
        available -= required;
    }

    // Pools with deferred reuse merge returned page identities only when the active supply is short.
    if (total - available > static_cast<std::int64_t>(free_pages_.size())) {
        free_pages_.insert(free_pages_.end(), released_pages_.begin(), released_pages_.end());
        released_pages_.clear();
        std::ranges::sort(free_pages_);
    }

    // Reserve all members only after the whole native allocation request is within capacity.
    std::vector<std::int64_t> indices;
    for (const auto& growth : members) {
        auto& resident = residents_.at(growth.request);
        while (static_cast<std::int64_t>(resident.pages.size()) < pages_for(growth.end)) {
            const auto page = free_pages_.front();
            free_pages_.pop_front();
            page_in_use_.at(static_cast<std::size_t>(page)) = true;
            resident.pages.push_back(page);
        }
        for (auto token = growth.begin; token < growth.end; ++token) {
            const auto page = resident.pages.at(static_cast<std::size_t>(token / limits_.page_size));
            indices.push_back(page * limits_.page_size + token % limits_.page_size);
        }
        resident.tokens = growth.end;
    }
    return indices;
}

void Resources::release_slot(const std::string& request) {
    auto& resident = residents_.at(request);
    free_slots_.push_back(resident.slot.value());
    resident = {};
}

void Resources::release_pages(const std::vector<std::int64_t>& pages) {
    std::unordered_set<std::int64_t> distinct;
    for (const auto page : pages) {
        if (!page_in_use_.at(static_cast<std::size_t>(page)) || !distinct.insert(page).second) {
            throw std::invalid_argument("returned KV page is not exclusively allocated");
        }
    }
    for (const auto page : pages) {
        page_in_use_.at(static_cast<std::size_t>(page)) = false;
    }

    if (limits_.sort_released_pages) {
        released_pages_.insert(released_pages_.end(), pages.begin(), pages.end());
    } else {
        free_pages_.insert(free_pages_.begin(), pages.begin(), pages.end());
    }
}

}
