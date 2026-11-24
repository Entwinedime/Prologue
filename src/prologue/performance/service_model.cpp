#include "prologue/performance/service_model.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "prologue/performance/work_features.hpp"

namespace prologue {

namespace {

using Json = nlohmann::json;

struct Binding {
    const Json & model;
    std::size_t ordinal;
    const Json & definition;
    const Json & work;
};

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
    explicit BoundedCurve(const Binding & spec) : ServiceModel(spec.model, spec.ordinal) {
        const auto & model = spec.model;
        if (!model.at("axis").is_null()) axis_ = model.at("axis").get<std::string>();
        for (const auto & row : model.at("knots").get_ref<const Json::array_t &>()) {
            const auto & values = row.get_ref<const Json::array_t &>();
            if (values.size() != 2 || !values[1].is_number_integer())
                throw std::invalid_argument("cost knot requires a coordinate and integer service nanoseconds");
            const Point point{ values[0].get<double>(), values[1].get<std::int64_t>() };
            if (!std::isfinite(point.coordinate) || point.ns < 0 || (!knots_.empty() && point.coordinate <= knots_.back().coordinate))
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
        const long double fraction = (static_cast<long double>(coordinate) - left.coordinate)
            / (static_cast<long double>(right->coordinate) - left.coordinate);
        return static_cast<std::int64_t>(std::nearbyint(left.ns + fraction * (static_cast<long double>(right->ns) - left.ns)));
    }
};

class AxisRegression : public ServiceModel {
protected:
    AxisRegression(const Binding & spec, std::string_view axis, std::string_view rate)
        : ServiceModel(spec.model, spec.ordinal), axis_(axis), startup_ns_(spec.model.at("startup_ns").get<double>()),
          rate_(spec.model.at(rate).get<double>()) {
        const auto range = bounds.find(axis);
        if (range == bounds.end() || range->second.lower <= 0 || range->second.lower >= range->second.upper
            || !std::isfinite(startup_ns_) || startup_ns_ < 0 || !std::isfinite(rate_) || rate_ <= 0)
            throw std::invalid_argument("linear service requires a positive measured span, finite startup and positive rate");
        duration(range->second.upper);
    }

private:
    std::string_view axis_;
    double startup_ns_;
    double rate_;

    std::int64_t duration(double coordinate) const {
        const auto ns = std::nearbyint(startup_ns_ + static_cast<long double>(coordinate) * rate_);
        if (ns >= 0x1p63L) throw std::out_of_range("service exceeds signed nanoseconds");
        return static_cast<std::int64_t>(ns);
    }
    std::int64_t predict(const FeatureValues & features) const override { return duration(*features.value(axis_)); }
};

class StartupTransfer final : public AxisRegression {
public:
    explicit StartupTransfer(const Binding & spec) : AxisRegression(spec, "transfer.bytes", "ns_per_byte") {
        const auto external = spec.definition.find("external");
        const bool copy = spec.work.contains("copy") || (external != spec.definition.end() && external->at("operation") == "local_copy");
        if (spec.work.at("kind") != "transfer" || !copy || bounds.size() != 1)
            throw std::invalid_argument("a startup transfer model requires a declared local copy with a byte coordinate");
    }
};

class AttentionPairs final : public AxisRegression {
public:
    explicit AttentionPairs(const Binding & spec) : AxisRegression(spec, "batch.attention_pairs", "ns_per_pair") {
        if (spec.work.at("kind") != "device" || spec.definition.value("role", "") != "attention")
            throw std::invalid_argument("an attention-pair model requires device work in an attention scope");
    }
};

class LayerTokens final : public AxisRegression {
public:
    explicit LayerTokens(const Binding & spec) : AxisRegression(spec, "parameter.model_tokens", "ns_per_token") {
        if (spec.work.at("kind") != "device" || spec.definition.value("role", "") != "layer")
            throw std::invalid_argument("a layer-token model requires device work in a model layer");
    }
};

template<class Backend>
std::unique_ptr<const ServiceModel> load(const Binding & spec) { return std::make_unique<Backend>(spec); }

} // namespace

ServiceModel::ServiceModel(const Json & model, std::size_t ordinal)
    : binding(ordinal), bounds(measured_bounds(model.at("bounds"))) {}

std::unique_ptr<const ServiceModel> ServiceModel::decode(const Json & model, std::size_t binding, const Json & definition,
                                                        std::size_t interval) {
    using Factory = std::unique_ptr<const ServiceModel> (*)(const Binding &);
    static const std::map<std::string_view, Factory> backends{
        { "bounded_curve", load<BoundedCurve> }, { "startup_transfer", load<StartupTransfer> },
        { "attention_pairs", load<AttentionPairs> }, { "layer_tokens", load<LayerTokens> },
    };
    const auto & family = model.at("family").get_ref<const std::string &>();
    const auto found = backends.find(family);
    if (found == backends.end()) throw std::invalid_argument("unknown service cost model family: " + family);
    return found->second({ model, binding, definition, definition.at("work").at(interval) });
}

std::optional<std::int64_t> ServiceModel::estimate(const FeatureValues & features) const {
    for (const auto & [name, bound] : bounds) {
        const auto value = features.value(name);
        if (!value || !std::isfinite(*value) || *value < bound.lower || *value > bound.upper) return {};
    }
    return predict(features);
}

} // namespace prologue
