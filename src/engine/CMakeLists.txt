find_package(nlohmann_json 3.11 REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

set(service_catalog "${CMAKE_CURRENT_SOURCE_DIR}/../shared/cost_families.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${service_catalog}")
file(READ "${service_catalog}" PROLOGUE_SERVICE_FAMILIES)
configure_file(core/performance/families.hpp.in generated/core/performance/families.hpp @ONLY)

add_library(prologue_core
    core/serving/controller/native.cpp
    core/serving/controller/cpu_messages.cpp
    core/serving/controller/delivery.cpp
    core/serving/resources/kv_pool.cpp
    core/serving/resources/service.cpp
    core/serving/resources/slot_pool.cpp
    core/serving/resources/prefix_index.cpp
    core/contracts/json_io.cpp
    core/contracts/expression.cpp
    core/contracts/serving.cpp
    core/graph/types.cpp
    core/graph/state.cpp
    core/graph/observation.cpp
    core/graph/graph.cpp
    core/serving/execution/timeline.cpp
    core/serving/execution/timing.cpp
    core/serving/execution/dependencies.cpp
    core/serving/execution/communication.cpp
    core/serving/session.cpp
    core/serving/service_times.cpp
    core/serving/execution_inputs.cpp
    core/performance/service_model.cpp
    core/performance/work_costs.cpp
    core/performance/work_features.cpp
    core/graph/projection.cpp
    core/templates/production.cpp
    core/templates/entries.cpp
    core/templates/library.cpp
    core/templates/bindings.cpp
    core/templates/interface.cpp
    core/templates/local.cpp
    core/templates/families.cpp
    core/templates/selection.cpp
    core/templates/supplement.cpp
    core/templates/validation.cpp
    core/templates/instance.cpp
)
target_include_directories(prologue_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
# Both evaluators use separate binary64 multiplication and addition before rounding nanoseconds.
set_source_files_properties(core/performance/service_model.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_link_libraries(prologue_core PUBLIC nlohmann_json::nlohmann_json PRIVATE OpenSSL::Crypto)

add_library(prologue_sglang
    adapters/registry.cpp
    adapters/sglang/templates.cpp
    adapters/sglang/workers.cpp
    adapters/sglang/actions.cpp
    adapters/sglang/serving.cpp
)
target_link_libraries(prologue_sglang PUBLIC prologue_core)

add_executable(prologue tools/main.cpp tools/commands.cpp tools/execution_trace.cpp)
target_link_libraries(prologue PRIVATE prologue_sglang)

# Local verification expands supplied native actions without running the serving controller.
add_executable(prologue_local_graphs tools/local_graphs.cpp)
target_link_libraries(prologue_local_graphs PRIVATE prologue_sglang)
foreach(target prologue_core prologue_sglang prologue prologue_local_graphs)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endforeach()
