# Shared warning flags exposed as an INTERFACE target.
# Link `partitions_warnings` into a target to opt in.

add_library(partitions_warnings INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(partitions_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused
        -Woverloaded-virtual -Wdouble-promotion)
elseif(MSVC)
    target_compile_options(partitions_warnings INTERFACE /W4 /permissive-)
endif()

# Helper: turn on aggressive optimisation (and -march=native when requested)
# for performance-sensitive targets such as the benchmarks.
function(partitions_enable_native target)
    if(PARTITIONS_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -O3 -march=native)
    endif()
endfunction()
