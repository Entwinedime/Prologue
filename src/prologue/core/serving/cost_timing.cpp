#include "prologue/core/serving/cost_timing.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace prologue::serving {

CostTiming::CostTiming(const nlohmann::json& domain, const performance::CostCatalog& costs)
    : domain_(domain), costs_(costs) {}

ActionTiming CostTiming::for_action(const nlohmann::json& action, const nlohmann::json& graph) {
    const performance::CostQueries queries(action, graph);
    std::vector<const nlohmann::json*> coordinates{&queries.host};
    coordinates.reserve(queries.device.size() + 1);
    for (const auto& coordinate : queries.device) {
        coordinates.push_back(&coordinate);
    }

    const auto estimates = costs_.require(domain_, coordinates);
    const auto& host = estimates.front();
    ActionTiming timing{{}, host.at("prepare_ns"), host.at("forward_ns"),
                        {0, host.at("result_ns")}, {}};
    timing.await_output_lane = true;
    std::size_t publication = 0;
    for (const auto& member : action.at("members")) {
        if (member.value("publishes", true)) {
            const auto request = member.at("request").get<std::string>();
            timing.publications.emplace(request, BoundaryTime{0, host.at("publications_ns").at(publication++)});
        }
    }

    const auto parallel = graph.contains("collectives");
    const auto rank_count = parallel ? graph.at("calls").front().at("parameters").at("ranks").size() : 1;
    std::vector<Nanoseconds> rank_submissions(rank_count, 0);
    for (std::size_t index = 0; index < queries.device.size(); ++index) {
        const auto& estimate = estimates.at(index + 1);
        const auto& work = graph.at("work").at(index);
        const auto rank = parallel ? work.at("path").front().get<std::size_t>() : 0;
        auto& submitted = rank_submissions.at(rank);
        const auto submission_ns = estimate.at("submission_ns").get<Nanoseconds>();
        if (submission_ns < 0 || submission_ns > std::numeric_limits<Nanoseconds>::max() - submitted) {
            throw std::invalid_argument("invalid driver submission duration");
        }
        submitted += submission_ns;
        timing.work.push_back({work.at("lane"), submitted, estimate.at("dispatch_ns"), estimate.at("duration_ns"),
                               work.value("after", std::vector<std::size_t>{})});
    }
    return timing;
}

ProgramTiming CostTiming::for_program(const nlohmann::json& action, const nlohmann::json& graph) {
    const auto geometry = structure::batch_geometry(action.at("phase"), action.at("members"));
    auto coordinates = nlohmann::json::array();
    for (const auto& step : graph.at("steps")) {
        coordinates.push_back({{"family", "host_thread"}, {"geometry", geometry},
                               {"stage", step.at("stage")}, {"cut", step.at("name")}});
    }
    throw performance::CostGap({{"status", "cost_gap"}, {"reason", "pipeline cuts need qualified timing models"},
                                {"domain", domain_}, {"coordinates", std::move(coordinates)}});
}

}
