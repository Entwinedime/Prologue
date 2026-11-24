#pragma once

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace prologue::performance {

class Estimator;

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
    CostCatalog(const nlohmann::json& asset, const nlohmann::json& rules);
    ~CostCatalog();

    std::vector<nlohmann::json> require(const nlohmann::json& domain,
                                        std::span<const nlohmann::json* const> coordinates) const;
    nlohmann::json quotes(const nlohmann::json& domain, const nlohmann::json& coordinates) const;

private:
    using Models = std::unordered_map<std::string, std::unique_ptr<Estimator>>;

    const Models* resolve(const nlohmann::json& domain) const;
    static std::optional<nlohmann::json> estimate(const Models* models, const nlohmann::json& coordinate);

    std::unordered_map<std::string, Models> domains_;
};

}
