#pragma once

#include <map>
#include <string>

#include "prologue/serving/action_graphs.hpp"
#include "prologue/serving/control.hpp"
#include "prologue/execution/dependencies.hpp"

namespace prologue {

class ServiceTimes;

struct ServingParticipant {
    ActionGraphs & graphs;
    GraphDependencies & dependencies;
};

class ServingSession {
public:
    ServingSession(ServingControl & control, Timeline & timeline, ServiceTimes & times, std::map<std::string, ServingParticipant> participants);
    nlohmann::json run();

private:
    ServingControl & control_;
    Timeline & timeline_;
    ServiceTimes & times_;
    std::map<std::string, ServingParticipant> participants_;
};

} // namespace prologue
