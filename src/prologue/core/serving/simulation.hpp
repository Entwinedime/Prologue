#pragma once

#include "prologue/core/serving/resources.hpp"
#include "prologue/core/serving/timeline.hpp"
#include "prologue/core/structure/templates/library.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace prologue::serving {

class Stopped final : public std::runtime_error {
public:
    Stopped(std::string status, const std::string& reason);
    const std::string& status() const;

private:
    std::string status_;
};

struct CutTiming final {
    Interval host;
    Nanoseconds ready_ns;
};

struct ProgramTiming final {
    std::vector<Interval> work;
    std::vector<std::vector<CutTiming>> steps;
    std::unordered_map<std::string, Nanoseconds> publications;
};

class TimingSource {
public:
    virtual ~TimingSource() = default;
    virtual ActionTiming for_action(const nlohmann::json& action, const nlohmann::json& graph) = 0;
    virtual ProgramTiming for_program(const nlohmann::json& action, const nlohmann::json& graph) = 0;
};

class Simulation final {
public:
    Simulation(const nlohmann::json& job, Workload workload, TimingSource& timing_source);
    ~Simulation();

    nlohmann::json request(const nlohmann::json& message);
    nlohmann::json result(std::string status, std::string reason);
    Nanoseconds now() const;
    bool done() const;

private:
    using Index = std::size_t;
    class Group;
    class Pipeline;
    class Disaggregation;

    struct ActiveAction {
        nlohmann::json record;
        ActionTiming timing;
        Nanoseconds work_ready_ns;
    };

    struct PendingRead {
        Nanoseconds earliest_ns;
        Index preceding_action;
    };

    struct Actor {
        Actor(const Workload& workload, ResourceLimits limits) : resources(workload, limits) {}

        Resources resources;
        Timeline clock;
        Index next_arrival = 0;
        std::vector<Nanoseconds> measured_arrivals;
        std::optional<ActiveAction> action;
        std::optional<PendingRead> read;
    };

    nlohmann::json arrivals(Index actor, bool finished);
    void measure_arrivals(const nlohmann::json& sources);
    std::optional<Nanoseconds> next_arrival(Index actor) const;
    void require_drained() const;
    nlohmann::json capacity(Index actor) const;
    nlohmann::json cache(Index actor, const nlohmann::json& message) const;
    void record_cache_effect(Index actor, const char* operation, const std::vector<std::int64_t>& pages);
    nlohmann::json resources(Index actor, const nlohmann::json& message);
    void require_residence(Index actor, const nlohmann::json& action) const;
    void record_publication(const nlohmann::json& message, Nanoseconds time_ns, nlohmann::json& target);
    void execute(Index actor, const nlohmann::json& message);
    void publish(Index actor, const nlohmann::json& message);
    void finish_action(Index actor);
    nlohmann::json direct(Index actor, const nlohmann::json& message);

    const nlohmann::json& job_;
    Workload workload_;
    std::vector<Actor> actors_;
    structure::Templates templates_;
    TimingSource& timing_source_;
    bool record_structure_;
    bool prefix_cache_;
    std::unordered_map<std::string, std::int64_t> output_limits_;
    std::unordered_map<std::string, std::int64_t> output_counts_;
    std::size_t finished_ = 0;
    nlohmann::json actions_ = nlohmann::json::array();
    nlohmann::json cache_effects_ = nlohmann::json::array();
    std::unique_ptr<Group> group_;
};

}
