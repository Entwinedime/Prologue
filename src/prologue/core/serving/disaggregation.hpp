#pragma once

#include "prologue/core/serving/group.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace prologue::serving {

class Simulation::Disaggregation final : public Group {
public:
    Disaggregation(Simulation& session, std::int64_t page_size);

    bool done() const override;
    void report(nlohmann::json& output) override;

private:
    static constexpr Index prefill = 0;
    static constexpr Index decode = 1;

    struct Transfer {
        Nanoseconds ready_ns;
        nlohmann::json metadata;
    };

    struct Handoff {
        Nanoseconds reserved_ns;
        nlohmann::json target;
        bool reservation_seen = false;
        std::optional<Transfer> transfer;
        std::array<bool, 2> retired{};
    };

    struct Admission {
        Nanoseconds at_ns;
        bool pause;
    };

    std::optional<Nanoseconds> eligible(Index actor, const nlohmann::json& call) const override;
    nlohmann::json dispatch(Index actor, const nlohmann::json& call) override;
    void measure_admission(const nlohmann::json& commands);
    nlohmann::json arrivals(Index actor);
    nlohmann::json endpoint(Index actor, const nlohmann::json& call) const;
    void reserve(const nlohmann::json& call);
    void transfer(const nlohmann::json& call);
    nlohmann::json poll(Index actor, const nlohmann::json& call);
    void retire(Index actor, const nlohmann::json& call);
    void admit(const nlohmann::json& call);

    std::int64_t page_size_;
    std::unordered_map<std::string, Handoff> pending_;
    std::unordered_set<std::string> waiting_for_reservation_;
    std::unordered_set<std::string> received_;
    std::deque<Admission> administration_;
    nlohmann::json receipts_ = nlohmann::json::object();
};

}
