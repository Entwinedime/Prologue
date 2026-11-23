#pragma once

#include "prologue/core/serving/resources.hpp"
#include "prologue/core/serving/timeline.hpp"
#include "prologue/core/structure/templates/library.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
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

struct ActionTiming final {
    std::vector<TimedWork> work;
    Nanoseconds prepare_ns;
    Nanoseconds model_return_ns;
    BoundaryTime result_finished;
    std::unordered_map<std::string, BoundaryTime> publications;
};

class TimingSource {
public:
    virtual ~TimingSource() = default;
    virtual ActionTiming for_action(const nlohmann::json& action, const nlohmann::json& graph) = 0;
};

class Simulation final {
public:
    Simulation(const nlohmann::json& job, Workload workload, TimingSource& timing_source);

    nlohmann::json request(const nlohmann::json& message);
    nlohmann::json result(std::string status, std::string reason);
    Nanoseconds now() const;
    bool done() const;

private:
    nlohmann::json arrivals();
    nlohmann::json capacity() const;
    nlohmann::json resources(const std::string& operation, const nlohmann::json& message);
    void execute(const nlohmann::json& message);
    void publish(const nlohmann::json& message);
    void finish_action();
    void wait_for(BoundaryTime boundary);

    const nlohmann::json& job_;
    Workload workload_;
    Resources resources_;
    structure::Templates templates_;
    Timeline timeline_;
    TimingSource& timing_source_;
    bool record_structure_;
    std::unordered_map<std::string, std::int64_t> output_limits_;
    std::unordered_map<std::string, std::int64_t> output_counts_;
    std::size_t next_arrival_ = 0;
    std::size_t finished_ = 0;
    nlohmann::json actions_ = nlohmann::json::array();
    nlohmann::json current_;
    std::optional<ActionTiming> timing_;
    Nanoseconds work_ready_ns_ = 0;
};

}
