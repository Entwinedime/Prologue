#include "core/contracts/expression.hpp"

#include "core/contracts/serving.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace prologue {

struct ExpressionOperator {
    using Reduce = std::int64_t (*)(std::int64_t, std::int64_t);

    std::string_view name;
    Reduce reduce;
    bool binary = false;
    bool canonical = false;
    std::optional<std::int64_t> identity{};
    std::optional<std::int64_t> right_identity{};
};

namespace {

using Integer = std::int64_t;

std::int64_t add(std::int64_t left, std::int64_t right) {
    std::int64_t result;
    if (__builtin_add_overflow(left, right, &result)) throw std::overflow_error("shape addition overflow");

    return result;
}

std::int64_t multiply(std::int64_t left, std::int64_t right) {
    std::int64_t result;
    if (__builtin_mul_overflow(left, right, &result)) throw std::overflow_error("shape multiplication overflow");

    return result;
}

std::int64_t floor_divide(std::int64_t numerator, std::int64_t denominator) {
    if (denominator == 0 || (numerator == std::numeric_limits<std::int64_t>::min() && denominator == -1)) throw std::domain_error("invalid shape division");

    // Python-style shape arithmetic floors negative quotients instead of truncating them.
    auto quotient = numerator / denominator;
    if (numerator % denominator != 0 && (numerator < 0) != (denominator < 0)) --quotient;

    return quotient;
}

Integer modulo(Integer left, Integer right) {

    // Division and remainder both reject a zero divisor and the signed minimum divided by -1.
    floor_divide(left, right);
    const auto rest = left % right;
    return rest != 0 && (rest < 0) != (right < 0) ? add(rest, right) : rest;
}

constexpr std::array operations{
    ExpressionOperator{ .name = "add", .reduce = add, .canonical = true, .identity = 0 },
    ExpressionOperator{ .name = "mul", .reduce = multiply, .canonical = true, .identity = 1 },
    ExpressionOperator{ .name = "floor_div", .reduce = floor_divide, .binary = true, .right_identity = 1 },
    ExpressionOperator{ .name = "mod", .reduce = modulo, .binary = true },
    ExpressionOperator{ "min", [](Integer a, Integer b) -> Integer { return std::min(a, b); } },
    ExpressionOperator{ "max", [](Integer a, Integer b) -> Integer { return std::max(a, b); } },
    ExpressionOperator{ .name = "eq", .reduce = [](Integer a, Integer b) -> Integer { return a == b; },
                        .binary = true, .canonical = true },
    ExpressionOperator{ .name = "lt", .reduce = [](Integer a, Integer b) -> Integer { return a < b; }, .binary = true },
    ExpressionOperator{ .name = "le", .reduce = [](Integer a, Integer b) -> Integer { return a <= b; }, .binary = true },
    ExpressionOperator{ "and", [](Integer a, Integer b) -> Integer { return a != 0 && b != 0; } },
    ExpressionOperator{ "or", [](Integer a, Integer b) -> Integer { return a != 0 || b != 0; } },
};

} // namespace

Parameters read_parameters(const nlohmann::json & source) {
    if (!source.is_object()) throw std::invalid_argument("parameters must be an object");

    Parameters result;
    for (const auto & [name, value] : source.items()) result.emplace(name, read_integer(value, name));
    return result;
}

Expression Expression::read(const nlohmann::json & value) {
    Expression result;
    if (value.is_number_integer()) {
        if (value.is_number_unsigned() && value.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::out_of_range("expression integer exceeds signed 64-bit range");
        result.value_ = value.get<std::int64_t>();
    }
    else if (value.is_string()) result.value_ = value.get<std::string>();
    else {
        const auto & name = value.at(0).get_ref<const std::string &>();
        const auto definition = std::ranges::find(operations, name, &ExpressionOperator::name);
        if (definition == operations.end()) throw std::invalid_argument("unknown expression operator: " + name);

        const auto count = value.size() - 1;
        if (count == 0 || (definition->binary && count != 2))
            throw std::invalid_argument("invalid expression arity");

        std::vector<Expression> arguments;
        arguments.reserve(count);
        for (std::size_t index = 1; index < value.size(); ++index) arguments.push_back(read(value.at(index)));
        return make_operation(*definition, std::move(arguments));
    }

    return result;
}

Expression Expression::make_operation(const ExpressionOperator & definition, std::vector<Expression> arguments) {
    if (definition.identity) {

        // Removing other values, such as a zero multiplier, could hide an earlier overflow or division failure.
        std::erase_if(arguments, [&](const auto & argument) {
            const auto * constant = std::get_if<Integer>(&argument.value_);
            return constant && *constant == *definition.identity;
        });
        if (arguments.empty()) return read(*definition.identity);
    }
    if (definition.right_identity) {
        const auto * right = std::get_if<Integer>(&arguments[1].value_);
        if (right && *right == *definition.right_identity) return std::move(arguments.front());
    }
    if (arguments.size() == 1) return std::move(arguments.front());

    Expression result;
    result.value_ = Operation{ &definition, std::move(arguments) };
    return result;
}

std::set<std::string> Expression::variables() const {
    if (const auto * name = std::get_if<std::string>(&value_)) return { *name };

    std::set<std::string> result;
    if (const auto * operation = std::get_if<Operation>(&value_)) {
        for (const auto & argument : operation->arguments) result.merge(argument.variables());
    }
    return result;
}

std::int64_t Expression::evaluate(const Parameters & parameters) const {
    if (const auto * constant = std::get_if<std::int64_t>(&value_)) return *constant;

    if (const auto * parameter = std::get_if<std::string>(&value_)) {
        const auto found = parameters.find(*parameter);
        if (found == parameters.end()) throw std::invalid_argument("missing parameter: " + *parameter);

        return found->second;
    }

    const auto & operation = std::get<Operation>(value_);
    auto result = operation.arguments.front().evaluate(parameters);

    for (std::size_t index = 1; index < operation.arguments.size(); ++index) {

        // Every operand is checked, including the right side of a logical operation.
        const auto next = operation.arguments[index].evaluate(parameters);
        result = operation.definition->reduce(result, next);
    }

    return result;
}

nlohmann::json Expression::document() const {
    if (const auto * constant = std::get_if<std::int64_t>(&value_)) return *constant;
    if (const auto * parameter = std::get_if<std::string>(&value_)) return *parameter;

    const auto & operation = std::get<Operation>(value_);
    auto result = nlohmann::json::array({ operation.definition->name });
    for (const auto & argument : operation.arguments) result.push_back(argument.document());

    // Binary commutation preserves checked arithmetic; reordering longer sums/products can change overflow.
    if (operation.arguments.size() == 2 && operation.definition->canonical && result[2] < result[1])
        std::swap(result[1], result[2]);

    return result;
}

nlohmann::json Expression::document(const std::map<std::string, Expression> & arguments) const {
    return arguments.empty() ? document() : substitute(arguments).document();
}

Expression Expression::substitute(const std::map<std::string, Expression> & expressions) const {
    if (const auto * name = std::get_if<std::string>(&value_)) {
        const auto found = expressions.find(*name);
        return found == expressions.end() ? *this : found->second;
    }

    if (const auto * operation = std::get_if<Operation>(&value_)) {
        std::vector<Expression> arguments;
        for (const auto & argument : operation->arguments) arguments.push_back(argument.substitute(expressions));
        return make_operation(*operation->definition, std::move(arguments));
    }
    return *this;
}

bool Expression::parameterized() const {
    if (std::holds_alternative<std::string>(value_)) return true;

    if (const auto * operation = std::get_if<Operation>(&value_)) {
        for (const auto & argument : operation->arguments) {
            if (argument.parameterized()) return true;
        }
    }

    return false;
}

} // namespace prologue
