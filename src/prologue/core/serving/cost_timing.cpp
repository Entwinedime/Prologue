#include "prologue/core/serving/cost_timing.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace prologue::serving {

CostTiming::CostTiming(const nlohmann::json& domain, const performance::CostCatalog& costs)
    : domain_(domain), costs_(costs) {}

ActionTiming CostTiming::for_action(const nlohmann::json& action, const nlohmann::json& graph) {
    const performance::CostQueries queries(action, graph);
    const auto& host = costs_.require(domain_, queries.host);
    ActionTiming timing{{}, host.at("prepare_ns"), host.at("forward_ns"),
                        {0, host.at("result_ns")}, {}};
    const auto& members = action.at("members");
    for (std::size_t index = 0; index < members.size(); ++index) {
        const auto request = members.at(index).at("request").get<std::string>();
        timing.publications.emplace(request, BoundaryTime{0, host.at("publications_ns").at(index)});
    }

    Nanoseconds submitted = 0;
    for (std::size_t index = 0; index < queries.device.size(); ++index) {
        const auto& estimate = costs_.require(domain_, queries.device.at(index));
        const auto& work = graph.at("work").at(index);
        const auto submission_ns = estimate.at("submission_ns").get<Nanoseconds>();
        if (submission_ns < 0 || submission_ns > std::numeric_limits<Nanoseconds>::max() - submitted) {
            throw std::invalid_argument("invalid driver submission duration");
        }
        submitted += submission_ns;
        timing.work.push_back({work.at("lane"), submitted, estimate.at("dispatch_ns"), estimate.at("duration_ns")});
    }
    return timing;
}

}
