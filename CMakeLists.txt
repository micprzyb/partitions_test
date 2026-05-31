cmake_minimum_required(VERSION 3.20)

project(partitions_test
    VERSION 0.1.0
    DESCRIPTION "Correctness and performance test bench for custom partition functions"
    LANGUAGES CXX)

# ---- Defaults -------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

option(PARTITIONS_BUILD_TESTS      "Build the correctness test suite" ON)
option(PARTITIONS_BUILD_BENCHMARKS "Build the performance benchmarks" ON)
option(PARTITIONS_BUILD_TOOLS      "Build the statistics tools"       ON)
option(PARTITIONS_NATIVE           "Compile benchmarks/tools with -march=native" ON)

# ---- Header-only library --------------------------------------------------
add_library(partitions INTERFACE)
add_library(partitions::partitions ALIAS partitions)
target_include_directories(partitions INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(partitions INTERFACE cxx_std_23)

include(cmake/CompilerWarnings.cmake)

# ---- Subdirectories -------------------------------------------------------
if(PARTITIONS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(PARTITIONS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
if(PARTITIONS_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

message(STATUS "partitions_test: build=${CMAKE_BUILD_TYPE} "
    "tests=${PARTITIONS_BUILD_TESTS} benchmarks=${PARTITIONS_BUILD_BENCHMARKS} "
    "tools=${PARTITIONS_BUILD_TOOLS} native=${PARTITIONS_NATIVE}")
