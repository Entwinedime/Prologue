#include "prologue/performance/service_model.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "prologue/performance/work_features.hpp"

#include "prologue/contracts/serving.hpp"
#include "prologue/performance/families.hpp"

namespace prologue {

namespace {

using Json = nlohmann::json;

std::int64_t service_ns(double value) {
    if (!std::isfinite(value) || value < 0 || value >= 0x1p63) throw std::out_of_range("service exceeds signed nanoseconds");

    // Match Python round(): halfway values go to the even integer, independent of the caller's rounding mode.
    const auto whole = static_cast<std::int64_t>(value);
    const auto fraction = value - static_cast<double>(whole);
    return whole + (fraction > 0.5 || (fraction == 0.5 && whole % 2 != 0));
}

auto measured_bounds(const Json & bounds) {
    std::map<std::string, ServiceModel::Range, std::less<>> result;
    for (const auto & [name, range] : bounds.get_ref<const Json::object_t &>()) {
        const auto & values = range.get_ref<const Json::array_t &>();
        if (values.size() != 2) throw std::invalid_argument("cost feature bounds require two coordinates");
        const ServiceModel::Range limits{ values[0].get<double>(), values[1].get<double>() };
        if (!std::isfinite(limits.lower) || !std::isfinite(limits.upper) || limits.lower > limits.upper)
            throw std::invalid_argument("cost feature bounds must be finite and ordered");
        result.emplace(name, limits);
    }
    return result;
}

class BoundedCurve final : public ServiceModel {
public:
    BoundedCurve(const Json & model, std::size_t ordinal) : ServiceModel(model, ordinal) {
        if (!model.at("axis").is_null()) axis_ = model.at("axis").get<std::string>();
        for (const auto & row : model.at("knots").get_ref<const Json::array_t &>()) {
            const auto & values = row.get_ref<const Json::array_t &>();
            if (values.size() != 2 || !values[1].is_number_integer())
                throw std::invalid_argument("cost knot requires a coordinate and integer service nanoseconds");
            const Point point{ values[0].get<double>(), read_nonnegative_integer(values[1], "service knot") };
            if (!std::isfinite(point.coordinate) || (!knots_.empty() && point.coordinate <= knots_.back().coordinate))
                throw std::invalid_argument("cost knots require increasing coordinates and nonnegative costs");
            knots_.push_back(point);
        }

        // Interpolation covers the entire declared axis. Constant services retain their measured feature bounds too.
        if (axis_) {
            const auto found = bounds.find(*axis_);
            if (knots_.size() < 2 || found == bounds.end() || knots_.front().coordinate != found->second.lower
                || knots_.back().coordinate != found->second.upper)
                throw std::invalid_argument("cost knots do not span their declared feature axis");
        } else if (knots_.size() != 1) {
            throw std::invalid_argument("a constant cost model requires one knot");
        }
    }

private:
    struct Point {
        double coordinate;
        std::int64_t ns;
    };
    std::optional<std::string> axis_;
    std::vector<Point> knots_;

    std::int64_t predict(const FeatureValues & features) const override {
        if (!axis_) return knots_.front().ns;
        const auto coordinate = *features.value(*axis_);
        const auto right = std::ranges::lower_bound(knots_, coordinate, {}, &Point::coordinate);
        if (right->coordinate == coordinate) return right->ns;
        const auto & left = *std::prev(right);
        const double offset = static_cast<double>(right->ns - left.ns) * (coordinate - left.coordinate) / (right->coordinate - left.coordinate);
        return service_ns(static_cast<double>(left.ns) + offset);
    }
};

class LinearRegression final : public ServiceModel {
public:
    LinearRegression(const Json & model, std::size_t ordinal, const Json & encoding)
        : ServiceModel(model, ordinal), startup_ns_(model.at("startup_ns").get<double>()) {
        if (!std::isfinite(startup_ns_) || startup_ns_ < 0) throw std::invalid_argument("invalid linear service startup");

        // Published named rates and newly registered coefficients both describe the same ordered sum.
        if (encoding.contains("rates")) {
            for (const auto & [axis, field] : encoding.at("rates").items())
                rates_.emplace(axis, model.at(field.get_ref<const std::string &>()).get<double>());
        } else {
            rates_ = model.at("coefficients").get<decltype(rates_)>();
        }
        if (rates_.empty()) throw std::invalid_argument("linear service requires at least one feature axis");
        double maximum = startup_ns_;
        for (const auto & [axis, rate] : rates_) {
            const auto span = bounds.find(axis);
            if (span == bounds.end() || span->second.lower <= 0 || span->second.lower >= span->second.upper || !std::isfinite(rate) || rate <= 0)
                throw std::invalid_argument("linear service requires a positive measured span and rate");
            maximum += span->second.upper * rate;
        }
        service_ns(maximum);
    }

private:
    std::map<std::string, double> rates_;
    double startup_ns_;

    std::int64_t predict(const FeatureValues & features) const override {
        double value = startup_ns_;
        for (const auto & [axis, rate] : rates_) value += *features.value(axis) * rate;
        return service_ns(value);
    }
};

} // namespace

ServiceModel::ServiceModel(const Json & model, std::size_t ordinal)
    : binding(ordinal), bounds(measured_bounds(model.at("bounds"))) {}

const Json & ServiceModelRegistry::encoding(const Json & model) {
    if (model.contains("form")) return model;
    static const Json catalog = Json::parse(published_family_contracts);
    const auto & family = model.at("family").get_ref<const std::string &>();
    const auto found = catalog.find(family);
    if (found == catalog.end()) throw std::invalid_argument("unknown service cost model family: " + family);
    return *found;
}

std::unique_ptr<const ServiceModel> ServiceModelRegistry::decode(const Json & model, std::size_t binding) {
    const auto & wire = encoding(model);
    const auto & form = wire.at("form").get_ref<const std::string &>();
    if (form == "curve") return std::make_unique<BoundedCurve>(model, binding);
    if (form == "linear") return std::make_unique<LinearRegression>(model, binding, wire);
    throw std::invalid_argument("unknown service model evaluator: " + form);
}

std::optional<std::int64_t> ServiceModel::estimate(const FeatureValues & features) const {
    for (const auto & [name, bound] : bounds) {
        const auto value = features.value(name);
        if (!value || !std::isfinite(*value) || *value < bound.lower || *value > bound.upper) return {};
    }
    return predict(features);
}

} // namespace prologue
