#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/contracts/expression.hpp"

namespace prologue {

enum class StateAccess { Read, Write };
StateAccess state_access(std::string_view name);
std::string_view state_access_name(StateAccess access);

struct TokenRegion {
    std::string request;
    std::int64_t begin;
    std::int64_t end;

    static TokenRegion read(const nlohmann::json & source);
    nlohmann::json document() const;
    bool operator==(const TokenRegion &) const = default;
};

// Each member supplies logical token positions; physical pages remain owned by the resource service.
struct TokenRangeRule {
    Expression begin;
    Expression end;

    static TokenRangeRule read(const nlohmann::json & source);
    nlohmann::json document() const;
    std::vector<TokenRegion> evaluate(const nlohmann::json & work) const;
};

// Each member contributes a slice of its own table row to the output's leading dimension.
struct RowPacking {
    std::string table;
    Expression member_count;
    Expression row_extent;
    TokenRangeRule range;

    static RowPacking read(const nlohmann::json & source);
    nlohmann::json document(const std::map<std::string, Expression> & arguments = {}) const;
    nlohmann::json bind(const nlohmann::json & work, const Parameters & parameters, const std::vector<std::int64_t> & shape) const;
};

// The caller owns the table and destination; each child sees one admitted row and produces one slice.
struct RowCall {
    std::size_t member;
    std::string parent_port;
    std::string source_port;
    std::string destination_port;

    static RowCall read(const nlohmann::json & source);
    nlohmann::json document() const;
};

std::span<const nlohmann::json> admitted_members(const nlohmann::json & work);
Parameters member_parameters(const nlohmann::json & member);

// A token's K/V payload is one logical row; transfers may populate only part of it.
struct KvSlice {
    std::int64_t token_bytes;
    std::int64_t begin;
    std::int64_t end;

    static KvSlice read(const nlohmann::json & source);
    nlohmann::json document() const;
    void validate() const;
    std::vector<std::pair<std::int64_t, std::int64_t>> ranges(std::int64_t token_begin, std::int64_t token_end) const;
    bool operator==(const KvSlice &) const = default;
};

struct KvAccess {
    std::string pool;
    std::int64_t layer;
    StateAccess access;
    KvSlice slice;
    std::vector<TokenRegion> regions;
    std::optional<TokenRangeRule> rule;

    static KvAccess read(const nlohmann::json & source);
    nlohmann::json document() const;
};

// Logical request fields resolve through their declared resource pool, with independent physical lifetimes.
struct SlotAccess {
    std::string pool;
    std::string field;
    StateAccess access;
    std::string members;
    std::vector<std::string> requests;

    // Replicas of one request-scoped publication may write concurrently; different publications remain ordered.
    std::string publication;

    static SlotAccess read(const nlohmann::json & source);
    nlohmann::json document() const;
};

} // namespace prologue
