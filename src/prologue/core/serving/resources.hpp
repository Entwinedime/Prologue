#pragma once

#include "prologue/core/serving/workload.hpp"

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace prologue::serving {

struct ResourceLimits final {
    std::int64_t slots;
    std::int64_t kv_tokens;
    std::int64_t page_size;
    bool sort_released_pages = false;
};

struct KvGrowth final {
    std::string request;
    std::int64_t begin;
    std::int64_t end;
    std::vector<std::int64_t> prefix_pages;
};

struct Residence final {
    std::optional<std::int64_t> slot;
    std::int64_t tokens = 0;
    std::vector<std::int64_t> pages;
};

class Resources final {
public:
    Resources(const Workload& workload, ResourceLimits limits);

    std::int64_t available_slots() const;
    std::int64_t available_tokens() const;
    std::int64_t evictable_tokens() const;
    std::int64_t protected_tokens() const;
    bool drained() const;
    const std::deque<std::int64_t>& free_slots() const;
    const Residence& residence(const std::string& request) const;
    std::vector<std::int64_t> locations(const std::string& request) const;
    std::int64_t pinned_prefix(const std::string& request) const;

    std::vector<std::int64_t> bind_slots(const std::vector<std::string>& requests);
    std::vector<std::int64_t> grow(const std::vector<KvGrowth>& members);
    void release_slot(const std::string& request);
    std::vector<std::int64_t> release_pages(const std::vector<std::int64_t>& pages);
    void retain_prefix(const std::vector<std::int64_t>& pages);
    std::int64_t pin_prefix(const std::vector<std::int64_t>& pages, bool acquire);
    void rebind_prefix(const std::string& request, const std::vector<std::int64_t>& pages);

private:
    enum class Ownership { free, request, cache };

    struct Page {
        Ownership owner = Ownership::free;
        std::int64_t pins = 0;
    };

    std::int64_t pages_for(std::int64_t tokens) const;
    void require_prefix(const std::vector<std::int64_t>& pages) const;
    void append_locations(const Residence& resident, std::int64_t begin, std::int64_t end,
                          std::vector<std::int64_t>& indices) const;

    ResourceLimits limits_;
    std::deque<std::int64_t> free_slots_;
    std::deque<std::int64_t> free_pages_;
    std::vector<std::int64_t> released_pages_;
    std::vector<Page> pages_;
    std::int64_t cached_pages_ = 0;
    std::int64_t protected_pages_ = 0;
    std::unordered_map<std::string, Residence> residents_;
};

}
