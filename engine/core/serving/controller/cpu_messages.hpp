#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/serving/observer.hpp"
#include "core/serving/execution_inputs.hpp"
#include "core/serving/execution/timing.hpp"
#include "core/serving/execution/timeline.hpp"

namespace prologue {

class ServiceTimes;

struct CpuActor {
    std::string instance;
    Resource control;
};

// The broker owns numeric payloads and channel positions; this service owns their execution readiness.
class CpuMessages {
public:
    CpuMessages(Timeline & timeline, ServiceTimes & times, ControlObserver * observer);
    void add_actor(const std::string & group, CpuActor actor);
    void prime();
    void call(std::size_t call, const std::string & group, const nlohmann::json & message);
    void notify(std::size_t event);
    std::vector<std::pair<std::size_t, nlohmann::json>> take_replies();
    std::set<std::size_t> retained_events() const;
    bool shutdown_ready() const;
    nlohmann::json summary() const;

private:
    struct Endpoint {
        std::string group;
        std::size_t actor;
        nlohmann::json descriptor;
        nlohmann::json execution;
        std::optional<std::size_t> entered;
        std::optional<std::size_t> posted;
        std::optional<std::size_t> returned;
        std::optional<std::size_t> wait_call;
        std::optional<std::size_t> wait_return;
        std::optional<std::size_t> ready_event;
        nlohmann::json wait_execution;
        std::optional<InputPosition> input;
        EventTiming readiness;
        bool ready = false;
        bool consumed = false;
    };

    struct Message {
        std::array<std::optional<Endpoint>, 2> endpoints;
        std::vector<std::size_t> posts() const;
    };

    enum class Phase { Post, Return, Ready, Consume };
    struct Completion {
        Phase phase;
        nlohmann::json key;
        std::size_t endpoint;
        std::optional<std::size_t> call;
    };

    struct Service {
        std::size_t entered;
        std::optional<std::size_t> posted;
        std::size_t returned;
        std::optional<InputPosition> input;
    };

    void observe(const nlohmann::json & key, const Endpoint & endpoint, const nlohmann::json & fields);
    void declare_ready(const nlohmann::json & key, Endpoint & endpoint, std::size_t side);
    void publish_ready(const nlohmann::json & key, Endpoint & endpoint, std::span<const std::size_t> prerequisites);
    void resolve(const nlohmann::json & key, Message & message);
    ServiceDuration estimate(const Endpoint & owner, const nlohmann::json & execution, std::string_view operation, WorkKind kind);
    Service service(const nlohmann::json & key, const Endpoint & endpoint,
                    const nlohmann::json & execution, bool posting, std::optional<std::size_t> prerequisite);
    void consume(const nlohmann::json & key, Endpoint & endpoint);
    void post(std::size_t call, const std::string & group, const nlohmann::json & message);
    void wait(std::size_t call, const std::string & group, const nlohmann::json & message);

    Timeline & timeline_;
    ServiceTimes & times_;
    ControlObserver * observer_;
    std::map<std::string, CpuActor> actors_;
    std::map<nlohmann::json, Message> messages_;
    std::map<std::size_t, std::vector<Completion>> completions_;
    std::vector<std::pair<std::size_t, nlohmann::json>> replies_;
    bool priming_ = true;
    std::size_t service_count_ = 0;
    std::size_t ready_count_ = 0;
    std::size_t consumed_count_ = 0;
};

} // namespace prologue
