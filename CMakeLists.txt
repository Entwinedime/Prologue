cmake_minimum_required(VERSION 3.20)
project(prologue LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json 3.11 REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_library(prologue_core
    src/prologue/serving/controller/native.cpp
    src/prologue/serving/controller/cpu_messages.cpp
    src/prologue/serving/controller/delivery.cpp
    src/prologue/serving/resources/kv_pool.cpp
    src/prologue/serving/resources/service.cpp
    src/prologue/serving/resources/slot_pool.cpp
    src/prologue/serving/resources/prefix_index.cpp
    src/prologue/contracts/json_io.cpp
    src/prologue/contracts/expression.cpp
    src/prologue/contracts/serving.cpp
    src/prologue/graph/types.cpp
    src/prologue/graph/state.cpp
    src/prologue/graph/observation.cpp
    src/prologue/graph/graph.cpp
    src/prologue/serving/execution/timeline.cpp
    src/prologue/serving/execution/timing.cpp
    src/prologue/serving/execution/dependencies.cpp
    src/prologue/serving/execution/communication.cpp
    src/prologue/serving/session.cpp
    src/prologue/serving/service_times.cpp
    src/prologue/serving/execution_inputs.cpp
    src/prologue/performance/service_model.cpp
    src/prologue/performance/work_costs.cpp
    src/prologue/performance/work_features.cpp
    src/prologue/graph/projection.cpp
    src/prologue/templates/production.cpp
    src/prologue/templates/entries.cpp
    src/prologue/templates/library.cpp
    src/prologue/templates/bindings.cpp
    src/prologue/templates/interface.cpp
    src/prologue/templates/local.cpp
    src/prologue/templates/families.cpp
    src/prologue/templates/selection.cpp
    src/prologue/templates/supplement.cpp
    src/prologue/templates/validation.cpp
    src/prologue/templates/instance.cpp
)
target_include_directories(prologue_core PUBLIC src)
target_link_libraries(prologue_core PUBLIC nlohmann_json::nlohmann_json PRIVATE OpenSSL::Crypto)

add_library(prologue_sglang
    src/prologue/integrations/frameworks/sglang/templates.cpp
    src/prologue/integrations/frameworks/sglang/workers.cpp
    src/prologue/integrations/frameworks/sglang/actions.cpp
    src/prologue/integrations/frameworks/sglang/serving.cpp
)
target_link_libraries(prologue_sglang PUBLIC prologue_core)

add_executable(prologue src/prologue/app/main.cpp src/prologue/app/execution_trace.cpp)
target_link_libraries(prologue PRIVATE prologue_sglang)

# Local verification expands supplied native actions without running the serving controller.
add_executable(prologue_local_graphs src/prologue/verification/local_graphs.cpp)
target_link_libraries(prologue_local_graphs PRIVATE prologue_sglang)
foreach(target prologue_core prologue_sglang prologue prologue_local_graphs)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endforeach()
