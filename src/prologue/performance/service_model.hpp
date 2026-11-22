#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace prologue {

class FeatureValues;

class ServiceModel {
public:
    struct Range {
        double lower;
        double upper;
    };

    virtual ~ServiceModel() = default;
    static std::unique_ptr<const ServiceModel> decode(const nlohmann::json & model, std::size_t binding,
                                                      const nlohmann::json & definition, std::size_t interval);
    std::optional<std::int64_t> estimate(const FeatureValues & features) const;

    const std::size_t binding;
    const std::map<std::string, Range, std::less<>> bounds;

protected:
    ServiceModel(const nlohmann::json & model, std::size_t binding);
    virtual std::int64_t predict(const FeatureValues & features) const = 0;
};

} // namespace prologue
