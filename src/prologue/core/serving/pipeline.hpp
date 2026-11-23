#pragma once

#include "prologue/core/serving/group.hpp"

#include <deque>
#include <map>
#include <memory>
#include <tuple>

namespace prologue::serving {

class Simulation::Pipeline final : public Group {
public:
    explicit Pipeline(Simulation& session);
    ~Pipeline() override;

    bool done() const override;
    void report(nlohmann::json& output) override;

private:
    using Channel = std::tuple<Index, Index, std::string>;
    using Queue = std::tuple<Index, std::string, Index>;
    class Program;

    struct Delivery {
        Channel channel;
        nlohmann::json payload;
        Nanoseconds posted_ns;
        bool received = false;
        bool waited = false;
    };

    std::optional<Nanoseconds> eligible(Index actor, const nlohmann::json& call) const override;
    nlohmann::json dispatch(Index actor, const nlohmann::json& call) override;
    void settle() override;
    Nanoseconds drained_time(Nanoseconds actor_end) const override;
    nlohmann::json launch(Index actor, const nlohmann::json& call);
    nlohmann::json send(Index actor, const nlohmann::json& call);
    nlohmann::json receive(Index actor, const nlohmann::json& call);
    void wait_send(Index actor, Index delivery);
    std::map<Index, std::unique_ptr<Program>> programs_;
    std::map<Channel, std::deque<Index>> inboxes_;
    std::map<Index, Delivery> deliveries_;
    std::map<Queue, Nanoseconds> lanes_;
    Index next_delivery_ = 0;
    Index next_program_ = 0;
    Index peak_inflight_ = 0;
};

}
