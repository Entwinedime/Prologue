#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace prologue::structure {

nlohmann::json batch_geometry(const nlohmann::json& phase, const nlohmann::json& members);

class TemplateGap final : public std::runtime_error {
public:
    explicit TemplateGap(nlohmann::json detail);

    const nlohmann::json detail;
};

class Templates final {
public:
    Templates(const nlohmann::json& bank, const nlohmann::json& domain, const nlohmann::json& rule);

    nlohmann::json expand(const nlohmann::json& query, bool include_definitions = false) const;
    void qualify() const;

private:
    class Expansion;

    const nlohmann::json& definitions_;
    std::unordered_map<std::string, std::vector<std::size_t>> queries_;
};

}
