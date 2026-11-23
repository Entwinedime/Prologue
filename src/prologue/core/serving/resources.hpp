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
    bool drained() const;
    const std::deque<std::int64_t>& free_slots() const;
    const Residence& residence(const std::string& request) const;

    std::vector<std::int64_t> bind_slots(const std::vector<std::string>& requests);
    std::vector<std::int64_t> grow(const std::vector<KvGrowth>& members);
    void release_slot(const std::string& request);
    void release_pages(const std::vector<std::int64_t>& pages);

private:
    std::int64_t pages_for(std::int64_t tokens) const;

    ResourceLimits limits_;
    std::deque<std::int64_t> free_slots_;
    std::deque<std::int64_t> free_pages_;
    std::vector<std::int64_t> released_pages_;
    std::vector<bool> page_in_use_;
    std::unordered_map<std::string, Residence> residents_;
};

}
