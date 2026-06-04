# Benchmarks are built with aggressive optimisation (and -march=native unless
# disabled).  They deliberately do NOT link the strict-warning interface: the
# (unsigned long long) printf casts etc. are benchmark-local noise.

foreach(bench_src bench_partition.cpp bench_pivot.cpp bench_large_n.cpp bench_small_sort.cpp)
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_include_directories(${bench_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${bench_name} PRIVATE partitions::partitions)
    partitions_enable_native(${bench_name})
endforeach()

# bench_small_sort additionally needs Boost.Sort headers.
find_package(Boost QUIET)
if(Boost_FOUND)
    target_include_directories(bench_small_sort PRIVATE ${Boost_INCLUDE_DIRS})
endif()

# bench_small_sort also compares against cpp-sort (Morwenn), vendored header-only
# under third_party/cpp-sort.  Optional: only wired in if the checkout exists.
set(_cppsort_inc ${CMAKE_SOURCE_DIR}/third_party/cpp-sort/include)
if(EXISTS ${_cppsort_inc}/cpp-sort/fixed/sorting_network_sorter.h)
    target_include_directories(bench_small_sort PRIVATE ${_cppsort_inc})
else()
    message(WARNING "cpp-sort not found at ${_cppsort_inc}; "
                    "run: git clone --depth 1 https://github.com/Morwenn/cpp-sort third_party/cpp-sort")
endif()
