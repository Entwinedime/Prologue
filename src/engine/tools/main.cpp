#include <exception>
#include <iostream>
#include <span>

#include "tools/commands.hpp"
#include "core/templates/selection.hpp"

int main(int argc, char ** argv) {
    try {
        prologue::run_commands(std::span(argv, static_cast<std::size_t>(argc)).subspan(1));
        return 0;
    }
    catch (const prologue::TemplateGap & gap) {
        std::cerr << gap.details() << '\n';
        return 1;
    }
    catch (const std::exception & error) {
        std::cerr << "prologue: " << error.what() << '\n';
        return 1;
    }
}
