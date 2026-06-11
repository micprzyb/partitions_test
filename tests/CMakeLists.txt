# Each test_*.cpp is linked with the shared runner into its own executable and
# registered with CTest, so `ctest` runs them in parallel and reports them
# individually.

set(PARTITIONS_TEST_FILES
    test_partition_correctness.cpp
    test_quicksort.cpp
    test_quicksort_lr.cpp
    test_reverse_partition.cpp
    test_quicksort_rev.cpp
    test_pivot.cpp
    test_pivot_partition.cpp
    test_distributions.cpp
    test_statistics.cpp
    test_move_low_half.cpp)

foreach(test_src IN LISTS PARTITIONS_TEST_FILES)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src} framework_main.cpp)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${test_name} PRIVATE partitions::partitions partitions_warnings)
    # Tests are correctness checks over a large templated matrix; optimising
    # them at -O3 (the Release default) costs minutes of compile time for no
    # benefit.  Force light optimisation regardless of build type.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${test_name} PRIVATE -O1)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
