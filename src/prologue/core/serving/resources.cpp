#include "prologue/core/serving/resources.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <unordered_set>

namespace prologue::serving {

Resources::Resources(const Workload& workload, ResourceLimits limits) : limits_(limits) {
    if (limits.page_size < 1 || limits.slots < 1 || limits.kv_tokens < limits.page_size) {
        throw std::invalid_argument("invalid resource capacity");
    }

    free_slots_.resize(static_cast<std::size_t>(limits.slots));
    free_pages_.resize(static_cast<std::size_t>(limits.kv_tokens / limits.page_size));
    pages_.resize(free_pages_.size());
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
    return available_slots() == limits_.slots && protected_pages_ == 0 &&
        free_pages_.size() + released_pages_.size() + static_cast<std::size_t>(cached_pages_) == pages_.size();
}

std::int64_t Resources::evictable_tokens() const {
    return (cached_pages_ - protected_pages_) * limits_.page_size;
}

std::int64_t Resources::protected_tokens() const {
    return protected_pages_ * limits_.page_size;
}

const std::deque<std::int64_t>& Resources::free_slots() const {
    return free_slots_;
}

const Residence& Resources::residence(const std::string& request) const {
    return residents_.at(request);
}

std::vector<std::int64_t> Resources::locations(const std::string& request) const {
    const auto& resident = residence(request);
    std::vector<std::int64_t> indices;
    indices.reserve(static_cast<std::size_t>(resident.tokens));
    append_locations(resident, 0, resident.tokens, indices);
    return indices;
}

std::int64_t Resources::pinned_prefix(const std::string& request) const {
    const auto& resident = residence(request);
    const auto tail = std::ranges::find_if(resident.pages, [this](const auto page) {
        return pages_.at(static_cast<std::size_t>(page)).pins == 0;
    });
    return (tail - resident.pages.begin()) * limits_.page_size;
}

void Resources::append_locations(const Residence& resident, std::int64_t begin, std::int64_t end,
                                std::vector<std::int64_t>& indices) const {
    for (auto token = begin; token < end; ++token) {
        const auto page = resident.pages.at(static_cast<std::size_t>(token / limits_.page_size));
        indices.push_back(page * limits_.page_size + token % limits_.page_size);
    }
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
        if (!distinct.insert(growth.request).second || !resident.slot || growth.begin < 0 ||
            growth.end <= growth.begin ||
            growth.prefix_pages.size() > static_cast<std::size_t>(growth.begin / limits_.page_size)) {
            throw std::invalid_argument("invalid KV extent for " + growth.request);
        }

        require_prefix(growth.prefix_pages);
        if (resident.tokens == 0) {
            if (growth.begin % limits_.page_size != 0 ||
                growth.prefix_pages.size() != static_cast<std::size_t>(growth.begin / limits_.page_size)) {
                throw std::invalid_argument("first KV extent needs a complete pinned prefix for " + growth.request);
            }
        } else if (growth.begin != resident.tokens || !std::ranges::equal(growth.prefix_pages,
            resident.pages | std::views::take(growth.prefix_pages.size()))) {
            throw std::invalid_argument("continued KV extent disagrees with residence for " + growth.request);
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
        if (resident.tokens == 0) {
            resident.pages = growth.prefix_pages;
        }
        while (static_cast<std::int64_t>(resident.pages.size()) < pages_for(growth.end)) {
            const auto page = free_pages_.front();
            free_pages_.pop_front();
            pages_.at(static_cast<std::size_t>(page)).owner = Ownership::request;
            resident.pages.push_back(page);
        }
        append_locations(resident, growth.begin, growth.end, indices);
        resident.tokens = growth.end;
    }
    return indices;
}

void Resources::release_slot(const std::string& request) {
    auto& resident = residents_.at(request);
    free_slots_.push_back(resident.slot.value());
    resident = {};
}

std::vector<std::int64_t> Resources::release_pages(const std::vector<std::int64_t>& pages) {
    std::unordered_set<std::int64_t> distinct;
    for (const auto page : pages) {
        const auto& state = pages_.at(static_cast<std::size_t>(page));
        if (state.owner == Ownership::free || state.pins != 0 || !distinct.insert(page).second) {
            throw std::invalid_argument("returned KV page is free, pinned, or repeated");
        }
    }

    std::vector<std::int64_t> evicted;
    for (const auto page : pages) {
        auto& state = pages_.at(static_cast<std::size_t>(page));
        if (state.owner == Ownership::cache) {
            --cached_pages_;
            evicted.push_back(page);
        }
        state.owner = Ownership::free;
    }

    if (limits_.sort_released_pages) {
        released_pages_.insert(released_pages_.end(), pages.begin(), pages.end());
    } else {
        free_pages_.insert(free_pages_.begin(), pages.begin(), pages.end());
    }
    return evicted;
}

void Resources::retain_prefix(const std::vector<std::int64_t>& pages) {
    std::unordered_set<std::int64_t> distinct;
    for (const auto page : pages) {
        if (pages_.at(static_cast<std::size_t>(page)).owner != Ownership::request || !distinct.insert(page).second) {
            throw std::invalid_argument("retained KV page is not a private allocation");
        }
    }
    for (const auto page : pages) {
        pages_.at(static_cast<std::size_t>(page)).owner = Ownership::cache;
    }
    cached_pages_ += static_cast<std::int64_t>(pages.size());
}

std::int64_t Resources::pin_prefix(const std::vector<std::int64_t>& pages, bool acquire) {
    std::unordered_set<std::int64_t> distinct;
    for (const auto page : pages) {
        const auto& state = pages_.at(static_cast<std::size_t>(page));
        if (state.owner != Ownership::cache || (!acquire && state.pins == 0) || !distinct.insert(page).second) {
            throw std::invalid_argument("invalid pin change for cached KV page");
        }
    }

    std::int64_t delta = 0;
    for (const auto page : pages) {
        auto& pins = pages_.at(static_cast<std::size_t>(page)).pins;
        if (acquire) {
            delta += pins++ == 0;
        } else {
            delta -= --pins == 0;
        }
    }
    protected_pages_ += delta;
    return -delta * limits_.page_size;
}

void Resources::require_prefix(const std::vector<std::int64_t>& pages) const {
    std::unordered_set<std::int64_t> distinct;
    for (const auto page : pages) {
        const auto& state = pages_.at(static_cast<std::size_t>(page));
        if (state.owner != Ownership::cache || state.pins == 0 || !distinct.insert(page).second) {
            throw std::invalid_argument("cached KV binding requires distinct pinned pages");
        }
    }
}

void Resources::rebind_prefix(const std::string& request, const std::vector<std::int64_t>& pages) {
    auto& resident = residents_.at(request);
    if (!resident.slot || pages.size() > static_cast<std::size_t>(resident.tokens / limits_.page_size)) {
        throw std::invalid_argument("cache rebind exceeds the committed KV prefix for " + request);
    }
    require_prefix(pages);

    // The native cache frees duplicate private pages before replacing their request mapping.
    for (std::size_t index = 0; index < pages.size(); ++index) {
        const auto previous = resident.pages.at(index);
        if (previous != pages.at(index) && pages_.at(static_cast<std::size_t>(previous)).owner != Ownership::free) {
            throw std::invalid_argument("cache rebind leaves an allocated duplicate for " + request);
        }
    }
    std::ranges::copy(pages, resident.pages.begin());
}

}
