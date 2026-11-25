cmake_minimum_required(VERSION 3.20)
project(prologue_gloo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
add_library(prologue_gloo SHARED gloo_timing.cpp)
target_link_libraries(prologue_gloo PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_options(prologue_gloo PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
