#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/serving/execution/timeline.hpp"
#include "core/serving/execution/timing.hpp"
#include "core/graph/types.hpp"

namespace prologue {

// Match device exchanges and retain the service frontier of ordered peer routes.
class CommunicationLinks {
public:
    std::optional<ExecutionInput> preceding_service(const CoarseGraph & graph) const;
    void insert(const CoarseGraph & graph, std::span<const std::size_t> events,
                std::span<const ServiceDuration> durations, Timeline & timeline);
    bool empty() const;
    std::set<std::size_t> retained_events() const;
    nlohmann::json payload_coverage() const;

private:
    struct ChannelPosition {
        std::size_t next = 0;
        std::optional<std::size_t> tail;
    };
    struct Collective {
        std::size_t ready;
        std::size_t finish;
        std::int64_t service_ns;
    };
    struct Endpoint {
        std::size_t ready;
        std::size_t completion;
        std::vector<std::int64_t> required;
        std::optional<Collective> collective;
        bool connected = false;
    };
    struct Exchange {
        std::string operation;
        std::vector<std::int64_t> members;
        std::map<std::tuple<Direction, std::string, bool>, TensorLayout> payload;
        bool collective;
        std::map<std::int64_t, Endpoint> endpoints;
    };

    std::map<std::pair<std::string, std::int64_t>, ChannelPosition> channels_;
    std::map<std::pair<std::string, std::size_t>, Exchange> pending_;
    std::size_t payload_endpoints_ = 0;
    std::size_t unknown_endpoints_ = 0;
    std::size_t shape_fields_ = 0;
    std::size_t dtype_fields_ = 0;
};

} // namespace prologue
