#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace prologue::structure {

nlohmann::json batch_geometry(const nlohmann::json& phase, const nlohmann::json& members);

class Templates final {
public:
    Templates(const nlohmann::json& libraries, const nlohmann::json& rule);

    std::optional<nlohmann::json> expand(const nlohmann::json& action) const;
    void qualify() const;

private:
    struct Candidate final {
        std::size_t library;
        std::size_t definition;
    };

    nlohmann::json select(const std::string& geometry, const std::vector<Candidate>& candidates) const;

    const nlohmann::json& libraries_;
    std::unordered_map<std::string, std::vector<Candidate>> batches_;
};

}
