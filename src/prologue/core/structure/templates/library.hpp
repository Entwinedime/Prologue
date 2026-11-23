#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace prologue::structure {

class Templates final {
public:
    explicit Templates(const nlohmann::json& libraries);

    std::optional<nlohmann::json> expand(const nlohmann::json& action) const;

private:
    struct Candidate final {
        std::size_t library;
        std::size_t definition;
    };

    const nlohmann::json& libraries_;
    std::unordered_map<std::string, std::vector<Candidate>> batches_;
};

}
