#pragma once

#include <cstddef>
#include <vector>

namespace prologue {

// Both graph qualification and time propagation need the same dependency order.
template <class Adjacency>
std::vector<std::size_t> dependency_order(const Adjacency & successors) {
    std::vector<std::size_t> incoming(successors.size());
    for (const auto & targets : successors) {
        for (const auto target : targets) ++incoming[target];
    }

    std::vector<std::size_t> ready;
    ready.reserve(successors.size());
    for (std::size_t event = 0; event < incoming.size(); ++event) {
        if (incoming[event] == 0) ready.push_back(event);
    }

    for (std::size_t index = 0; index < ready.size(); ++index) {
        for (const auto target : successors[ready[index]]) {
            if (--incoming[target] == 0) ready.push_back(target);
        }
    }
    return ready;
}

} // namespace prologue
