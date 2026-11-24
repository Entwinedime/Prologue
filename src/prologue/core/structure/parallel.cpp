#include "prologue/core/structure/parallel.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
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

std::vector<std::int64_t> row_splits(const json& counts, const json& port, Index width) {
    if (!counts.is_array() || counts.size() != width) {
        throw std::invalid_argument("collective row splits must name every participant");
    }
    std::vector<std::int64_t> splits;
    auto remaining = port.at("shape").at(0).get<std::int64_t>();
    for (const auto& count : counts) {
        if (!count.is_number_integer() || count < 0 || count > remaining) {
            throw std::invalid_argument("collective row splits exceed the tensor extent");
        }
        const auto rows = count.get<std::int64_t>();
        splits.push_back(rows);
        remaining -= rows;
    }
    if (remaining != 0) {
        throw std::invalid_argument("collective row splits do not cover the tensor extent");
    }
    return splits;
}

json row_layout(json port) {
    auto& shape = port.at("shape");
    shape.erase(0);
    port.erase("name");
    return port;
}

json peer_transfers(const std::vector<const json*>& calls, const json& ranks) {
    const auto layout = row_layout(calls.front()->at("inputs").at(0));
    auto common = calls.front()->at("parameters");
    common.erase("send_rows");
    common.erase("receive_rows");

    std::vector<std::vector<std::int64_t>> sends;
    std::vector<std::vector<std::int64_t>> receives;
    for (const auto* call : calls) {
        auto parameters = call->at("parameters");
        const auto& inputs = call->at("inputs");
        const auto& outputs = call->at("outputs");
        if (inputs.size() != 1 || outputs.size() != 1 || row_layout(inputs.at(0)) != layout ||
            row_layout(outputs.at(0)) != layout) {
            throw std::invalid_argument("collective row layouts disagree across ranks");
        }
        sends.push_back(row_splits(parameters.at("send_rows"), inputs.at(0), ranks.size()));
        receives.push_back(row_splits(parameters.at("receive_rows"), outputs.at(0), ranks.size()));
        parameters.erase("send_rows");
        parameters.erase("receive_rows");
        if (parameters != common) {
            throw std::invalid_argument("collective order or membership disagrees across ranks");
        }
    }

    auto transfers = json::array();
    for (Index source = 0; source < ranks.size(); ++source) {
        for (Index target = 0; target < ranks.size(); ++target) {
            const auto rows = sends[source][target];
            if (rows != receives[target][source]) {
                throw std::invalid_argument("collective sender and receiver disagree on the peer row count");
            }
            if (rows != 0) {
                transfers.push_back({{"source", ranks.at(source)}, {"target", ranks.at(target)}, {"rows", rows}});
            }
        }
    }
    return transfers;
}

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

        const auto& kind = first.at("parameters").at("kind");
        const bool routed = kind == "all_to_all";
        auto ready = json::array();
        std::vector<const json*> calls;
        calls.reserve(workers_.size());
        for (const auto& worker : workers_) {
            const auto& call = *endpoints_.at(worker.pending[worker.next].at("path").dump());
            if (!routed && (call.at("parameters") != first.at("parameters") ||
                call.at("inputs") != first.at("inputs") || call.at("outputs") != first.at("outputs"))) {
                throw std::invalid_argument("collective operands or order disagree across ranks");
            }
            calls.push_back(&call);
            if (worker.tail) {
                ready.push_back(*worker.tail);
            }
        }

        // The group has one rendezvous. Each rank receives its own result on its compute stream.
        auto& groups = graph_.at("collectives");
        json group{{"kind", kind}};
        if (routed) {
            group["transfers"] = peer_transfers(calls, ranks_);
        }
        auto results = json::array();
        for (Index rank = 0; rank < workers_.size(); ++rank) {
            auto& worker = workers_[rank];
            auto work = std::move(worker.pending[worker.next]);
            work["after"] = ready;
            work["collective"] = groups.size();
            results.push_back(append(rank, std::move(work)));
        }
        group["results"] = std::move(results);
        groups.push_back(std::move(group));
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

std::optional<std::int64_t> publication_lane(const nlohmann::json& graph) {
    if (!graph.contains("collectives")) {
        return std::nullopt;
    }
    return graph.at("calls").front().at("parameters").at("publication_rank").get<std::int64_t>();
}

}
