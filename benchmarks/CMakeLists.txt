# Benchmarks are built with aggressive optimisation (and -march=native unless
# disabled).  They deliberately do NOT link the strict-warning interface: the
# (unsigned long long) printf casts etc. are benchmark-local noise.

# ---- Always-build benchmarks (no external dependencies) -------------------
foreach(bench_src bench_partition.cpp bench_partition_methods.cpp bench_partition_efficiency.cpp bench_pivot.cpp bench_pivot_total.cpp bench_large_n.cpp bench_small_merge.cpp bench_small_halve.cpp bench_quickselect.cpp bench_quicksort.cpp bench_halver_compare.cpp bench_pivot_tiers.cpp bench_reverse_partition.cpp)
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_include_directories(${bench_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${bench_name} PRIVATE partitions::partitions)
    partitions_enable_native(${bench_name})
endforeach()

# ---- Optional comparison benchmarks (Boost.Sort + cpp-sort) ---------------
# bench_small_sort / bench_small_sort_dyn compare our small-array sort against
# Boost.Sort (pdqsort, spreadsort) and cpp-sort (Morwenn).  Both are external,
# header-only dependencies.  They are OPTIONAL: if either is missing the two
# targets are skipped, so the library, tests, tools and the other benchmarks
# still build (this is what keeps CI/release green without Boost installed).
#
#   * Boost  : provided by `libboost-dev` (found via find_package).
#   * cpp-sort: vendored under third_party/cpp-sort -- NOT committed; fetch with
#       git clone https://github.com/Morwenn/cpp-sort third_party/cpp-sort
find_package(Boost QUIET)
set(_cppsort_inc ${CMAKE_SOURCE_DIR}/third_party/cpp-sort/include)
set(_cppsort_ok FALSE)
if(EXISTS ${_cppsort_inc}/cpp-sort/fixed/sorting_network_sorter.h)
    set(_cppsort_ok TRUE)
endif()

if(Boost_FOUND AND _cppsort_ok)
    foreach(bench_src bench_small_sort.cpp bench_small_sort_dyn.cpp)
        get_filename_component(bench_name ${bench_src} NAME_WE)
        add_executable(${bench_name} ${bench_src})
        target_include_directories(${bench_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${_cppsort_inc})
        target_link_libraries(${bench_name} PRIVATE partitions::partitions)
        partitions_enable_native(${bench_name})
        # pair_fi/pair_di compare on float/double first keys; the branchless lex
        # predicate issues real FP compares.  We assume no NaNs, so -ffast-math
        # (-ffinite-math-only) drops the NaN/unordered handling -- applied to the
        # whole TU so every candidate (std::sort, Boost, cpp-sort, ours) gets it.
        target_compile_options(${bench_name} PRIVATE -ffast-math)
    endforeach()
    message(STATUS "comparison benchmarks (bench_small_sort*): ENABLED")
else()
    message(STATUS "comparison benchmarks (bench_small_sort*): DISABLED -- "
                   "need Boost [found=${Boost_FOUND}] and cpp-sort headers at "
                   "${_cppsort_inc} [found=${_cppsort_ok}]. "
                   "Library/tests/tools/other benchmarks are unaffected.")
endif()
