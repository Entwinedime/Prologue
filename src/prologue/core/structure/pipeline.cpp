#include "prologue/core/structure/pipeline.hpp"
#include "prologue/core/structure/parallel.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace prologue::structure {
namespace {

using nlohmann::json;
using Index = std::size_t;
using Endpoint = std::pair<Index, Index>;
using Wire = std::tuple<Index, Index, std::string>;

Endpoint endpoint(const json& address) {
    return {address.at("step").get<Index>(), address.at("rank").get<Index>()};
}

class CutOrder final {
public:
    explicit CutOrder(const json& steps) {
        for (Index step = 0; step < steps.size(); ++step) {
            for (const auto& rank : steps.at(step).at("ranks")) {
                inputs_.emplace(Endpoint{step, rank.get<Index>()}, 0);
            }
        }
    }

    void connect(Endpoint source, Endpoint target) {
        inputs_.at(source);
        ++inputs_.at(target);
        downstream_[source].push_back(target);
    }

    void verify() {
        std::deque<Endpoint> ready;
        for (const auto& [point, count] : inputs_) {
            if (count == 0) {
                ready.push_back(point);
            }
        }
        std::size_t visited = 0;
        while (!ready.empty()) {
            const auto point = ready.front();
            ready.pop_front();
            ++visited;
            for (const auto& target : downstream_[point]) {
                if (--inputs_.at(target) == 0) {
                    ready.push_back(target);
                }
            }
        }
        if (visited != inputs_.size()) {
            throw std::invalid_argument("pipeline cuts form a dependency cycle");
        }
    }

private:
    std::map<Endpoint, Index> inputs_;
    std::map<Endpoint, std::vector<Endpoint>> downstream_;
};

class Pipeline final {
public:
    explicit Pipeline(json& graph) : graph_(graph) {
        auto& metadata = graph_.at("calls").front().at("parameters");
        steps_ = std::move(metadata.at("steps"));
        dependencies_ = std::move(metadata.at("dependencies"));
        transfers_ = std::move(metadata.at("transfers"));
        metadata.erase("steps");
        metadata.erase("dependencies");
        metadata.erase("transfers");

        for (const auto& call : graph_.at("calls")) {
            const auto& path = call.at("path");
            if (path.size() == 1) {
                steps_.at(path.front().get<Index>())["ranks"] = call.at("parameters").at("ranks");
            } else if (call.at("operation") == "transfer_wait" && call.at("parameters").at("direction") == "receive") {
                fields_.emplace(path.dump(), call.at("parameters").at("field").get<std::string>());
            }
        }
    }

    void bind() {
        // Normalize communication inside one stage cut before joining any cross-stage work.
        std::vector<json> cuts(steps_.size(), {{"calls", json::array()}, {"work", json::array()}});
        for (const auto& row : graph_.at("calls")) {
            const auto& path = row.at("path");
            if (path.size() == 1 || row.at("operation") == "collective") {
                auto local = row;
                local.at("path").erase(local.at("path").begin());
                cuts.at(path.front().get<Index>()).at("calls").push_back(std::move(local));
            }
        }
        for (auto& row : graph_.at("work")) {
            auto& path = row.at("path");
            const auto step = path.front().get<Index>();
            path.erase(path.begin());
            cuts.at(step).at("work").push_back(std::move(row));
        }
        graph_["work"] = json::array();
        graph_["collectives"] = json::array();
        graph_["lanes"] = json::array();
        for (Index step = 0; step < cuts.size(); ++step) {
            append(step, cuts.at(step));
        }
        wire_dependencies();

        graph_["steps"] = std::move(steps_);
        graph_["dependencies"] = std::move(dependencies_);
        graph_["transfers"] = std::move(transfers_);
    }

private:
    void append(Index index, json& cut) {
        const auto grouped = std::ranges::any_of(cut.at("calls"), [](const auto& row) {
            return row.at("operation") == "collective";
        });
        if (grouped) {
            bind_collectives(cut);
        }

        auto& work = graph_.at("work");
        auto& collectives = graph_.at("collectives");
        const auto offset = work.size();
        const auto group_offset = collectives.size();
        auto& step = steps_.at(index);
        step["first_work"] = offset;
        step["work_count"] = cut.at("work").size();
        for (auto& entry : cut.at("work")) {
            auto& path = entry.at("path");
            const auto rank = step.at("ranks").at(path.front().get<Index>()).get<Index>();
            auto placed = json::array({index});
            for (const auto& part : path) {
                placed.push_back(part);
            }
            path = std::move(placed);
            const auto local = grouped ? 0 : entry.at("lane").get<Index>();
            const auto queue = step.at("queue").get<std::string>();
            const auto [lane, inserted] = lanes_.try_emplace(std::tuple{rank, queue, local}, lanes_.size());
            if (inserted) {
                graph_.at("lanes").push_back({rank, queue, local});
            }
            entry["lane"] = lane->second;
            entry["rank"] = rank;
            if (entry.contains("collective")) {
                entry["collective"] = entry.at("collective").get<Index>() + group_offset;
                for (auto& predecessor : entry.at("after")) {
                    predecessor = predecessor.get<Index>() + offset;
                }
            }
            if (entry.at("kind") == "transfer") {
                const auto& field = fields_.at(path.dump());
                if (!wire_results_.emplace(Wire{index, rank, field}, work.size()).second) {
                    throw std::invalid_argument("a pipeline wire has multiple receive results");
                }
            }
            work.push_back(std::move(entry));
        }
        if (grouped) {
            for (auto& group : cut.at("collectives")) {
                for (auto& result : group.at("results")) {
                    result = result.get<Index>() + offset;
                }
                collectives.push_back(std::move(group));
            }
        }
    }

    void wire_dependencies() {
        CutOrder order(steps_);
        for (const auto& edge : dependencies_) {
            order.connect(endpoint(edge.at(0)), endpoint(edge.at(1)));
        }

        for (auto& transfer : transfers_) {
            const auto source = endpoint(transfer.at("source"));
            const auto target = endpoint(transfer.at("target"));
            const auto sender_wait = transfer.at("sender_wait").get<Index>();
            const Wire wire{target.first, target.second, transfer.at("field").get<std::string>()};
            transfer["result_work"] = wire_results_.at(wire);
            wire_results_.erase(wire);

            // A posted send retains its backing until the matching wire reaches readiness.
            order.connect(target, {sender_wait, source.second});
        }
        if (!wire_results_.empty()) {
            throw std::invalid_argument("a pipeline result has no matching send");
        }
        order.verify();
    }

    json& graph_;
    json steps_;
    json dependencies_;
    json transfers_;
    std::map<std::string, std::string> fields_;
    std::map<std::tuple<Index, std::string, Index>, Index> lanes_;
    std::map<Wire, Index> wire_results_;
};

}

void bind_pipeline(nlohmann::json& graph) {
    Pipeline(graph).bind();
}

}
