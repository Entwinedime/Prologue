#include "prologue/core/serving/disaggregation.hpp"

#include <algorithm>
#include <utility>

namespace prologue::serving {

using nlohmann::json;

Simulation::Disaggregation::Disaggregation(Simulation& session, std::int64_t page_size)
    : Group(session), page_size_(page_size) {
    if (session_.actors_.size() != 2) {
        throw std::invalid_argument("disaggregation requires one Prefill actor and one Decode actor");
    }
    if (const auto conditions = session_.job_.find("measured_ingress"); conditions != session_.job_.end()) {
        measure_admission(conditions->at("admission"));
    }
}

bool Simulation::Disaggregation::done() const {
    return session_.finished_ == session_.workload_.requests().size() && pending_.empty() && received_.empty() &&
        administration_.empty() &&
        std::ranges::none_of(session_.actors_, [](const Actor& actor) { return actor.action.has_value(); });
}

void Simulation::Disaggregation::report(json& output) {
    if (session_.record_structure_) {
        output["handoffs"] = std::move(receipts_);
    }
}

void Simulation::Disaggregation::measure_admission(const json& commands) {
    for (const auto& event : commands) {
        const Admission boundary{event.at("at_ns"), event.at("pause")};
        if (boundary.at_ns < 0 || (!administration_.empty() && boundary.at_ns <= administration_.back().at_ns)) {
            throw std::invalid_argument("PD admission commands must follow their measured order");
        }
        administration_.push_back(boundary);
    }
}

json Simulation::Disaggregation::arrivals(Index actor) {
    auto result = session_.arrivals(actor, false);
    auto commands = json::array();
    if (actor == decode) {
        while (!administration_.empty() && administration_.front().at_ns <= now()) {
            commands.push_back(administration_.front().pause ? "pause" : "resume");
            administration_.pop_front();
        }
    }
    result["admission"] = std::move(commands);
    return result;
}

std::optional<Nanoseconds> Simulation::Disaggregation::eligible(Index actor, const json& call) const {
    if (call.at("operation") != "idle") {
        return now();
    }

    auto earliest = session_.next_arrival(actor);
    if (actor == decode && !administration_.empty() && (!earliest || administration_.front().at_ns < *earliest)) {
        earliest = administration_.front().at_ns;
    }
    for (const auto& [identity, state] : pending_) {
        if (state.retired.at(actor)) {
            continue;
        }
        std::optional<Nanoseconds> ready;
        if (state.transfer) {
            ready = state.transfer->ready_ns;
        } else if (actor == prefill && waiting_for_reservation_.contains(identity)) {
            ready = state.reserved_ns;
        }
        if (ready && (!earliest || *ready < *earliest)) {
            earliest = ready;
        }
    }
    return earliest;
}

json Simulation::Disaggregation::dispatch(Index actor, const json& call) {
    const auto& operation = call.at("operation");
    if (operation == "arrivals") {
        return arrivals(actor);
    }
    if (operation == "idle") {
        return nullptr;
    }
    if (operation == "reserve_handoff" && actor == decode) {
        reserve(call);
        return nullptr;
    }
    if (operation == "send_handoff" && actor == prefill) {
        transfer(call);
        return nullptr;
    }
    if (operation == "handoff_status") {
        return poll(actor, call);
    }
    if (operation == "retire_handoff") {
        retire(actor, call);
        return nullptr;
    }
    if (operation == "admit_handoff" && actor == decode) {
        admit(call);
        return nullptr;
    }
    return session_.direct(actor, call);
}

json Simulation::Disaggregation::endpoint(Index actor, const json& call) const {
    const auto request = call.at("request").get<std::string>();
    const auto& resident = session_.actors_.at(actor).resources.residence(request);
    if (!resident.slot || resident.pages != call.at("pages").get<std::vector<std::int64_t>>()) {
        throw Stopped("resource_error", "handoff pages do not match the resident allocation for " + request);
    }
    return {{"slot", *resident.slot}, {"pages", resident.pages}, {"tokens", resident.tokens},
            {"metadata_slot", call.at("metadata_slot")}};
}

void Simulation::Disaggregation::reserve(const json& call) {
    const auto request = call.at("request").get<std::string>();
    auto target = endpoint(decode, call);
    const auto& tokens = target.at("tokens");
    const json query{{"phase", "pd_reservation"}, {"tokens", tokens}};
    const json member{{"request", request}, {"kv_committed_tokens", tokens}, {"output_tokens", 0}};
    const json action{{"phase", query.at("phase")}, {"query", query}, {"clock_ns", now()},
                      {"members", json::array({member})}};
    const auto timing = session_.timing_source_.for_action(action, {{"calls", json::array()}, {"work", json::array()}});

    auto& clock = session_.actors_.at(decode).clock;
    clock.advance_by(timing.prepare_ns);
    const auto began = clock.now();
    clock.advance_by(timing.model_return_ns);
    clock.satisfy(began, clock.now(), timing.result_finished);
    Handoff state{.reserved_ns = clock.now(), .target = std::move(target), .transfer = std::nullopt};
    if (!pending_.emplace(request, std::move(state)).second) {
        throw Stopped("action_mismatch", "Decode submitted a second live handoff reservation for " + request);
    }
}

void Simulation::Disaggregation::transfer(const json& call) {
    const auto request = call.at("request").get<std::string>();
    auto& state = pending_.at(request);
    const auto source = endpoint(prefill, call);
    const auto tokens = source.at("tokens").get<std::int64_t>();
    if (state.transfer || !state.reservation_seen || tokens != state.target.at("tokens") ||
        static_cast<Index>((tokens - 1) / page_size_ + 1) != source.at("pages").size()) {
        throw Stopped("action_mismatch", "Prefill did not supply the reserved KV extent for " + request);
    }

    // A native result owns the source writes. Transport can outlive its host submission.
    const auto& actor = session_.actors_.at(prefill);
    const auto& producing = actor.action.value().record.at("members");
    const auto member = std::ranges::find_if(producing, [&request](const json& item) {
        return item.at("request") == request;
    });
    if (member == producing.end() || member->at("kv_committed_tokens") != tokens) {
        throw Stopped("action_mismatch", "handoff has no completed Prefill write for " + request);
    }

    const json query{{"phase", "pd_handoff"}, {"tokens", tokens}};
    auto graph = session_.templates_.expand(query);
    const json action{{"phase", query.at("phase")}, {"query", query}, {"clock_ns", now()},
                      {"members", json::array({*member})}};
    auto timing = session_.timing_source_.for_action(action, graph);
    Timeline link;
    link.advance_to(now());
    link.advance_by(timing.prepare_ns);
    const auto started = link.now();
    const auto completed = link.schedule_work(timing.work, session_.record_structure_);
    link.advance_by(timing.model_return_ns);
    link.advance_to(std::max(link.now(), completed.all_work_ready_ns));
    const auto returned = link.now();
    link.satisfy(started, returned, timing.result_finished);
    state.transfer.emplace(Transfer{link.now(), call.at("metadata")});

    if (session_.record_structure_) {
        auto intervals = json::array();
        for (const auto& item : completed.intervals) {
            intervals.push_back({item.begin_ns, item.end_ns});
        }
        receipts_[request] = {{"source", source}, {"target", state.target}, {"reserved_ns", state.reserved_ns},
                              {"begin_ns", started}, {"ready_ns", link.now()}, {"graph", std::move(graph)},
                              {"intervals", std::move(intervals)}};
    }
}

json Simulation::Disaggregation::poll(Index actor, const json& call) {
    const auto request = call.at("request").get<std::string>();
    const auto found = pending_.find(request);
    if (found == pending_.end() || found->second.reserved_ns > now()) {
        if (actor == prefill) {
            // Reservation can precede Prefill ingress. Only an actual poll subscribes to its wakeup.
            waiting_for_reservation_.insert(request);
        }
        return {{"state", "bootstrapping"}};
    }

    auto& state = found->second;
    if (!state.transfer) {
        if (actor == prefill) {
            waiting_for_reservation_.erase(request);
            state.reservation_seen = true;
            return {{"state", "reserved"}};
        }
        return {{"state", "transferring"}};
    }
    if (state.transfer->ready_ns > now()) {
        return {{"state", "transferring"}};
    }
    if (actor == decode) {
        return {{"state", "ready"}, {"metadata", state.transfer->metadata}};
    }
    return {{"state", "ready"}};
}

void Simulation::Disaggregation::retire(Index actor, const json& call) {
    const auto request = call.at("request").get<std::string>();
    auto& state = pending_.at(request);
    if (!state.transfer || state.transfer->ready_ns > now() || state.retired.at(actor)) {
        throw Stopped("action_mismatch", "native actor retired an unfinished handoff for " + request);
    }
    state.retired.at(actor) = true;
    if (actor == decode) {
        received_.insert(request);
    }
    if (session_.record_structure_) {
        receipts_.at(request)[actor == prefill ? "released_ns" : "committed_ns"] = now();
    }
    if (std::ranges::all_of(state.retired, [](bool retired) { return retired; })) {
        pending_.erase(request);
    }
}

void Simulation::Disaggregation::admit(const json& call) {
    auto inputs = json::array();
    for (const auto& member : call.at("members")) {
        const auto request = member.at("request").get<std::string>();
        if (!received_.contains(request)) {
            throw Stopped("action_mismatch", "Decode admitted a request without a committed handoff " + request);
        }
        inputs.push_back(request);
    }
    session_.execute(decode, call);
    for (const auto& input : inputs) {
        received_.erase(input.get<std::string>());
    }
    if (session_.record_structure_) {
        session_.actors_.at(decode).action->record["handoff_inputs"] = std::move(inputs);
    }
}

}
