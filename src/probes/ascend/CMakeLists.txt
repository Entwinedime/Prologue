cmake_minimum_required(VERSION 3.20)
project(prologue_ascend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# The bridge consumes only the SDK C ABI; model-selection and Torch workspace code are not part of observation.
add_library(prologue_ascend_observer SHARED native_calls.cpp runtime_timing.cpp thread_activity.cpp)
target_include_directories(prologue_ascend_observer SYSTEM PRIVATE "${PROLOGUE_ASCEND_INCLUDE_DIR}")
target_link_libraries(prologue_ascend_observer PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_options(prologue_ascend_observer PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

# Weight residency is loaded only by single-device workers whose native model exceeds their memory share.
execute_process(
    COMMAND python3 -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('torch').origin).parent / 'share/cmake/Torch')"
    OUTPUT_VARIABLE torch_config OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(Torch REQUIRED PATHS "${torch_config}")
find_library(ascend_runtime ascendcl REQUIRED PATHS "${PROLOGUE_ASCEND_INCLUDE_DIR}/../lib64")
add_library(prologue_ascend_weights SHARED weight_storage.cpp)
target_include_directories(prologue_ascend_weights SYSTEM PRIVATE "${PROLOGUE_ASCEND_INCLUDE_DIR}")
target_link_libraries(prologue_ascend_weights PRIVATE ${TORCH_LIBRARIES} ${ascend_runtime})
