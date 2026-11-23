#pragma once

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace prologue::performance {

class CostGap final : public std::runtime_error {
public:
    explicit CostGap(nlohmann::json detail);

    const nlohmann::json detail;
};

class CostQueries final {
public:
    CostQueries(const nlohmann::json& action, const nlohmann::json& graph);

    nlohmann::json host;
    nlohmann::json device;
};

class CostCatalog final {
public:
    explicit CostCatalog(const nlohmann::json& asset);

    const nlohmann::json& require(const nlohmann::json& domain, const nlohmann::json& coordinate) const;
    nlohmann::json quote(const nlohmann::json& domain, const nlohmann::json& coordinate) const;

private:
    nlohmann::json domain_;
    std::unordered_map<std::string, nlohmann::json> estimates_;
};

}
