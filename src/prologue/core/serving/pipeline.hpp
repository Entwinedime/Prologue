#pragma once

#include "prologue/core/serving/simulation.hpp"

#include <deque>
#include <map>
#include <memory>
#include <tuple>

namespace prologue::serving {

class Simulation::Pipeline final {
public:
    Pipeline(Simulation& session, std::size_t width);
    ~Pipeline();

    nlohmann::json request(const nlohmann::json& message);
    Nanoseconds now() const;
    bool done() const;
    std::size_t peak_inflight() const;

private:
    using Index = std::size_t;
    using Channel = std::tuple<Index, Index, std::string>;
    using Queue = std::tuple<Index, std::string, Index>;
    class Program;

    struct Actor {
        Nanoseconds time_ns = 0;
        std::optional<nlohmann::json> pending;
    };

    struct Delivery {
        Channel channel;
        nlohmann::json payload;
        Nanoseconds posted_ns;
        bool received = false;
        bool waited = false;
    };

    std::optional<Nanoseconds> eligible(Index actor, const nlohmann::json& call) const;
    nlohmann::json dispatch(Index actor, const nlohmann::json& call);
    nlohmann::json launch(Index actor, const nlohmann::json& call);
    nlohmann::json send(Index actor, const nlohmann::json& call);
    nlohmann::json receive(Index actor, const nlohmann::json& call);
    void wait_send(Index actor, Index delivery);
    void retire_programs();

    Simulation& session_;
    std::vector<Actor> actors_;
    std::map<Index, std::unique_ptr<Program>> programs_;
    std::map<Channel, std::deque<Index>> inboxes_;
    std::map<Index, Delivery> deliveries_;
    std::map<Queue, Nanoseconds> lanes_;
    Nanoseconds now_ns_ = 0;
    Index next_delivery_ = 0;
    Index next_program_ = 0;
    Index peak_inflight_ = 0;
};

}
