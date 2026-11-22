#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "prologue/contracts/expression.hpp"
#include "prologue/graph/graph.hpp"

namespace prologue {

class FeatureValues {
public:
    virtual std::optional<double> value(std::string_view axis) const = 0;

protected:
    ~FeatureValues() = default;
};

class WorkFeatures final : public FeatureValues {
public:
    WorkFeatures(const GraphScope & scope, WorkKind kind);
    WorkFeatures(const TensorCopy & copy, const GraphScope & scope, const Resource & resource);
    explicit WorkFeatures(Parameters parameters);

    std::string_view phase() const { return phase_; }
    std::optional<double> value(std::string_view axis) const override;
    const nlohmann::json * transport() const { return transport_.is_null() ? nullptr : &transport_; }
    nlohmann::json document() const;

private:
    enum BatchAxis { Members, QueryTokens, ContextTokens, MaxContextTokens, MaxQueryTokens, AttentionPairs, BatchAxisCount };
    static constexpr std::array<std::string_view, BatchAxisCount> batch_names_{
        "batch.members", "batch.query_tokens", "batch.context_tokens", "batch.max_context_tokens",
        "batch.max_query_tokens", "batch.attention_pairs",
    };
    static constexpr std::string_view parameter_prefix_ = "parameter.";

    std::span<const std::string_view> batch_axes() const { return std::span(batch_names_).first(batch_axis_count_); }

    std::map<std::string, std::int64_t, std::less<>> parameters_;
    std::string_view phase_;
    nlohmann::json transport_;
    std::optional<std::int64_t> copy_bytes_;
    std::array<double, BatchAxisCount> batch_{};
    std::size_t batch_axis_count_{};
};

class WorkCoordinates {
public:
    explicit WorkCoordinates(const CoarseGraph & graph) : graph_(graph), by_scope_(graph.scopes.size()) {}
    std::size_t index(const GraphWork & work);
    const WorkFeatures & operator[](std::size_t point) const { return points_[point]; }

private:
    const CoarseGraph & graph_;
    std::vector<std::array<std::optional<std::size_t>, 2>> by_scope_;
    std::vector<WorkFeatures> points_;
};

} // namespace prologue
