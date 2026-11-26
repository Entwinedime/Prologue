#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace prologue {

using Parameters = std::map<std::string, std::int64_t>;

Parameters read_parameters(const nlohmann::json & source);

struct ExpressionOperator;

// The same checked expressions bind both observed and projected geometry.
class Expression {
public:
    static Expression read(const nlohmann::json & value);

    std::int64_t evaluate(const Parameters & parameters) const;
    bool parameterized() const;
    std::set<std::string> variables() const;
    Expression substitute(const std::map<std::string, Expression> & expressions) const;
    nlohmann::json document() const;
    nlohmann::json document(const std::map<std::string, Expression> & arguments) const;

private:
    struct Operation {
        const ExpressionOperator * definition;
        std::vector<Expression> arguments;
    };

    static Expression make_operation(const ExpressionOperator & definition, std::vector<Expression> arguments);
    std::variant<std::int64_t, std::string, Operation> value_;
};

} // namespace prologue
