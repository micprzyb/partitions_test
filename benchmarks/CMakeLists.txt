# Benchmarks are built with aggressive optimisation (and -march=native unless
# disabled).  They deliberately do NOT link the strict-warning interface: the
# (unsigned long long) printf casts etc. are benchmark-local noise.

foreach(bench_src bench_partition.cpp bench_pivot.cpp)
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_include_directories(${bench_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${bench_name} PRIVATE partitions::partitions)
    partitions_enable_native(${bench_name})
endforeach()
