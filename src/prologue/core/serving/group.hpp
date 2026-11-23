#pragma once

#include "prologue/core/serving/simulation.hpp"

#include <optional>
#include <vector>

namespace prologue::serving {

class Simulation::Group {
public:
    explicit Group(Simulation& session);
    virtual ~Group() = default;

    nlohmann::json request(const nlohmann::json& message);
    Nanoseconds now() const;
    virtual bool done() const = 0;
    virtual void report(nlohmann::json& output) = 0;

protected:
    virtual std::optional<Nanoseconds> eligible(Index actor, const nlohmann::json& call) const = 0;
    virtual nlohmann::json dispatch(Index actor, const nlohmann::json& call) = 0;
    virtual void settle() {}
    virtual Nanoseconds drained_time(Nanoseconds actor_end) const { return actor_end; }

    Simulation& session_;

private:
    std::vector<std::optional<nlohmann::json>> pending_;
    Nanoseconds now_ns_ = 0;
};

}
