cmake_minimum_required(VERSION 3.20)
project(prologue LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json 3.11 REQUIRED)

add_library(prologue_core STATIC
    src/prologue/core/performance/costs.cpp
    src/prologue/core/serving/cost_timing.cpp
    src/prologue/core/serving/resources.cpp
    src/prologue/core/serving/simulation.cpp
    src/prologue/core/serving/timeline.cpp
    src/prologue/core/serving/workload.cpp
    src/prologue/core/structure/templates/bank.cpp
    src/prologue/core/structure/templates/compiler.cpp
    src/prologue/core/structure/templates/library.cpp
)
target_include_directories(prologue_core PUBLIC src)
target_link_libraries(prologue_core PUBLIC nlohmann_json::nlohmann_json)

add_executable(prologue-core
    src/prologue/app/control_channel.cpp
    src/prologue/app/main.cpp
    src/prologue/app/timing.cpp
)
target_link_libraries(prologue-core PRIVATE prologue_core)

foreach(target prologue_core prologue-core)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endforeach()

install(TARGETS prologue-core RUNTIME DESTINATION bin)
