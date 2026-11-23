#include "prologue/core/structure/parallel.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace prologue::structure {
namespace {

using nlohmann::json;
using Index = std::size_t;

struct RankWork final {
    std::vector<json> pending;
    Index next = 0;
    std::optional<Index> tail;
};

class Collectives final {
public:
    explicit Collectives(json& graph)
        : graph_(graph), ranks_(graph.at("calls").front().at("parameters").at("ranks")),
          workers_(ranks_.size()) {
        for (const auto& call : graph.at("calls")) {
            if (call.at("operation") == "collective") {
                endpoints_.emplace(call.at("path").dump(), &call);
            }
        }
        for (auto& work : graph.at("work")) {
            const auto rank = work.at("path").at(0).get<Index>();
            workers_.at(rank).pending.push_back(std::move(work));
        }
        graph_["work"] = json::array();
        graph_["collectives"] = json::array();
    }

    void bind() {
        while (true) {
            Index waiting = 0;
            for (Index rank = 0; rank < workers_.size(); ++rank) {
                auto& worker = workers_[rank];
                while (worker.next < worker.pending.size()) {
                    auto& work = worker.pending[worker.next];
                    if (work.at("kind") == "collective") {
                        ++waiting;
                        break;
                    }
                    if (work.at("lane") != 0) {
                        throw std::invalid_argument("the collective cut requires one observed compute stream per rank");
                    }
                    append(rank, std::move(work));
                }
            }
            if (waiting == 0) {
                return;
            }
            if (waiting != workers_.size()) {
                throw std::invalid_argument("a collective is missing a participant");
            }
            rendezvous();
        }
    }

private:
    Index append(Index rank, json work) {
        auto& bound = graph_.at("work");
        const auto index = bound.size();
        work["lane"] = rank;
        bound.push_back(std::move(work));
        auto& worker = workers_[rank];
        worker.tail = index;
        ++worker.next;
        return index;
    }

    void rendezvous() {
        const auto& first = *endpoints_.at(workers_.front().pending[workers_.front().next].at("path").dump());
        if (first.at("parameters").at("participants") != ranks_) {
            throw std::invalid_argument("collective membership does not match the observed rank group");
        }

        auto ready = json::array();
        for (const auto& worker : workers_) {
            const auto& call = *endpoints_.at(worker.pending[worker.next].at("path").dump());
            if (call.at("parameters") != first.at("parameters") ||
                call.at("inputs") != first.at("inputs") || call.at("outputs") != first.at("outputs")) {
                throw std::invalid_argument("collective operands or order disagree across ranks");
            }
            if (worker.tail) {
                ready.push_back(*worker.tail);
            }
        }

        // The group has one rendezvous. Each rank receives its own result on its compute stream.
        auto& groups = graph_.at("collectives");
        auto results = json::array();
        for (Index rank = 0; rank < workers_.size(); ++rank) {
            auto& worker = workers_[rank];
            auto work = std::move(worker.pending[worker.next]);
            work["after"] = ready;
            work["collective"] = groups.size();
            results.push_back(append(rank, std::move(work)));
        }
        groups.push_back({{"kind", first.at("parameters").at("kind")}, {"results", std::move(results)}});
    }

    json& graph_;
    const json& ranks_;
    std::vector<RankWork> workers_;
    std::unordered_map<std::string, const json*> endpoints_;
};

}

void bind_collectives(nlohmann::json& graph) {
    Collectives(graph).bind();
}

}
