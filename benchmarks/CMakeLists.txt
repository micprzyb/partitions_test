# Benchmarks are built with aggressive optimisation (and -march=native unless
# disabled).  They deliberately do NOT link the strict-warning interface: the
# (unsigned long long) printf casts etc. are benchmark-local noise.

foreach(bench_src bench_partition.cpp bench_pivot.cpp bench_large_n.cpp bench_small_sort.cpp bench_small_sort_dyn.cpp bench_small_merge.cpp)
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_include_directories(${bench_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${bench_name} PRIVATE partitions::partitions)
    partitions_enable_native(${bench_name})
endforeach()

# The small-sort benchmarks additionally need Boost.Sort and cpp-sort (Morwenn,
# vendored header-only under third_party/cpp-sort).
find_package(Boost QUIET)
set(_cppsort_inc ${CMAKE_SOURCE_DIR}/third_party/cpp-sort/include)
set(_cppsort_ok FALSE)
if(EXISTS ${_cppsort_inc}/cpp-sort/fixed/sorting_network_sorter.h)
    set(_cppsort_ok TRUE)
else()
    message(WARNING "cpp-sort not found at ${_cppsort_inc}; "
                    "run: git clone --depth 1 https://github.com/Morwenn/cpp-sort third_party/cpp-sort")
endif()
foreach(_ss bench_small_sort bench_small_sort_dyn)
    if(Boost_FOUND)
        target_include_directories(${_ss} PRIVATE ${Boost_INCLUDE_DIRS})
    endif()
    if(_cppsort_ok)
        target_include_directories(${_ss} PRIVATE ${_cppsort_inc})
    endif()
    # The pair_fi/pair_di element types compare on float/double first keys.  The
    # branchless lexicographic predicate issues real FP compares; under the
    # default IEEE rules each is a ucomisd that must also handle NaN/unordered.
    # We assume no NaNs in the data, so -ffast-math (-ffinite-math-only) lets the
    # compiler drop that handling -- applied to the whole TU, so EVERY candidate
    # (std::sort, Boost, cpp-sort and ours) gets it: an apples-to-apples flag.
    target_compile_options(${_ss} PRIVATE -ffast-math)
endforeach()
