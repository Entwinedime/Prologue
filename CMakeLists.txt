cmake_minimum_required(VERSION 3.20)
project(prologue LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json 3.11 REQUIRED)

add_library(prologue_core
    src/prologue/controller/native.cpp
    src/prologue/controller/cpu_messages.cpp
    src/prologue/controller/delivery.cpp
    src/prologue/resources/kv_pool.cpp
    src/prologue/resources/service.cpp
    src/prologue/resources/slot_pool.cpp
    src/prologue/resources/prefix_index.cpp
    src/prologue/contracts/expression.cpp
    src/prologue/contracts/serving.cpp
    src/prologue/graph/types.cpp
    src/prologue/graph/state.cpp
    src/prologue/graph/observation.cpp
    src/prologue/graph/graph.cpp
    src/prologue/execution/timeline.cpp
    src/prologue/execution/timing.cpp
    src/prologue/execution/dependencies.cpp
    src/prologue/execution/communication.cpp
    src/prologue/serving/session.cpp
    src/prologue/serving/service_times.cpp
    src/prologue/serving/execution_inputs.cpp
    src/prologue/performance/work_costs.cpp
    src/prologue/performance/work_features.cpp
    src/prologue/production/projection.cpp
    src/prologue/production/templates.cpp
    src/prologue/production/entries.cpp
    src/prologue/templates/library.cpp
    src/prologue/templates/interface.cpp
    src/prologue/templates/local.cpp
    src/prologue/templates/selection.cpp
    src/prologue/templates/validation.cpp
    src/prologue/templates/instance.cpp
)
target_include_directories(prologue_core PUBLIC src)
target_link_libraries(prologue_core PUBLIC nlohmann_json::nlohmann_json)

add_library(prologue_sglang
    src/prologue/adapters/sglang/templates.cpp
    src/prologue/adapters/sglang/workers.cpp
    src/prologue/adapters/sglang/actions.cpp
    src/prologue/adapters/sglang/serving.cpp
)
target_link_libraries(prologue_sglang PUBLIC prologue_core)

add_executable(prologue src/prologue/cli/main.cpp src/prologue/cli/execution_trace.cpp)
target_link_libraries(prologue PRIVATE prologue_sglang)

# Local verification expands supplied native actions without running the serving controller.
add_executable(prologue_local_graphs src/verification/local_graphs.cpp)
target_link_libraries(prologue_local_graphs PRIVATE prologue_sglang)
foreach(target prologue_core prologue_sglang prologue prologue_local_graphs)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endforeach()
