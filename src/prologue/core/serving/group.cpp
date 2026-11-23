#include "prologue/core/serving/group.hpp"

#include <algorithm>
#include <utility>

namespace prologue::serving {

using nlohmann::json;

Simulation::Group::Group(Simulation& session) : session_(session), pending_(session.actors_.size()) {}

Nanoseconds Simulation::Group::now() const {
    return now_ns_;
}

json Simulation::Group::request(const json& message) {
    const auto& operation = message.at("operation");
    if (operation == "setup_actor") {
        return session_.resources(message.at("actor"), message.at("call"));
    }
    if (operation != "turn") {
        throw Stopped("controller_error", "group control requires actor turns");
    }
    for (const auto& envelope : message.at("calls")) {
        auto& pending = pending_.at(envelope.at("actor"));
        if (pending) {
            throw Stopped("controller_error", "a native actor has two outstanding calls");
        }
        pending = envelope.at("call");
    }
    if (done() && std::ranges::none_of(session_.actors_, [](const Actor& actor) { return actor.read.has_value(); })) {
        session_.require_drained();
        for (const auto& actor : session_.actors_) {
            now_ns_ = std::max(now_ns_, actor.clock.now());
        }
        now_ns_ = drained_time(now_ns_);
        return {{"done", true}};
    }

    std::optional<std::pair<Nanoseconds, Index>> selected;
    for (Index index = 0; index < pending_.size(); ++index) {
        const auto& call = pending_.at(index).value();
        const auto ready = eligible(index, call);
        if (!ready) {
            continue;
        }

        const auto& actor = session_.actors_.at(index);
        auto time = std::max({*ready, now_ns_, actor.clock.now()});
        if (actor.read && call.at("operation") == "arrivals") {
            time = std::max(time, actor.read->earliest_ns);
        }
        const auto candidate = std::pair{time, index};
        if (!selected || candidate < *selected) {
            selected = candidate;
        }
    }
    if (!selected) {
        throw Stopped("controller_stall", "native actors are waiting without an eligible input or delivery");
    }

    const auto [time, index] = *selected;
    now_ns_ = time;
    auto& clock = session_.actors_.at(index).clock;
    clock.advance_to(time);
    auto result = dispatch(index, pending_.at(index).value());
    pending_.at(index).reset();
    settle();
    return {{"done", false}, {"actor", index}, {"now_ns", clock.now()}, {"result", std::move(result)}};
}

}
