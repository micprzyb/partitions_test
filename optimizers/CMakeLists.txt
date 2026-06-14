# Perfect-halver optimisation search tools (see optimizers/README.md).
#
# These are standalone (no library headers): the ILS network search with an
# exhaustive flat 0/1 check (search_halve, for N<=17) and the CEGAR + k-zero
# check (search_halve_cegar, for large N).  Built here so `cmake --build` keeps
# them compiling; they are normally driven by optimizers/optimize.sh.
#
# bench_pool.cpp is intentionally NOT a target: it #includes a net list generated
# at run time (/tmp/hpool.inc), so optimize.sh compiles it on demand.

add_executable(search_halve search_halve.cpp)
partitions_enable_native(search_halve)

add_executable(search_halve_cegar search_halve_cegar.cpp)
partitions_enable_native(search_halve_cegar)
