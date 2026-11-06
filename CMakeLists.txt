cmake_minimum_required(VERSION 3.16)
project(alysane CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Portable baseline: one binary, runtime SIMD dispatch (see include/simd.h).
# ALYSANE_NATIVE=ON opts into -march=native for local benchmarking only.
option(ALYSANE_NATIVE "Tune for the build host (non-portable)" OFF)
# -ffast-math is deliberately NOT used: it enables unsafe FP reassociation and
# was implicated in a GCC-13/EPYC Release-only segfault. Correctness of the
# parity and certificate gates takes precedence over the marginal speedup.
if(ALYSANE_NATIVE)
  add_compile_options(-O3 -march=native)
else()
  add_compile_options(-O3)
endif()
add_compile_options(-fopenmp -Wall -Wextra -Wno-unused-parameter)
add_link_options(-fopenmp)
find_package(OpenMP REQUIRED)
find_library(OPENBLAS openblas REQUIRED)
find_library(LAPACKE lapacke REQUIRED)
include_directories(include third_party /usr/include/x86_64-linux-gnu)
set(ALYSANE_SOURCES
  src/model.cpp
  src/self.cpp
  src/gates.cpp
  src/kv.cpp
  src/continuation.cpp
  src/structure.cpp
  src/quantize.cpp
  src/drift.cpp
  src/gates2.cpp
  tests/unit_tests.cpp
  src/probe.cpp
  tests/probe_tests.cpp
)
add_executable(alysane src/main.cpp ${ALYSANE_SOURCES})
target_link_libraries(alysane PRIVATE ${OPENBLAS} ${LAPACKE} OpenMP::OpenMP_CXX)

# Unit tests: no model weights needed, runs in CI on every push.
add_executable(alysane_unit_tests tests/unit_main.cpp ${ALYSANE_SOURCES})
target_link_libraries(alysane_unit_tests PRIVATE ${OPENBLAS} ${LAPACKE} OpenMP::OpenMP_CXX)
enable_testing()
add_test(NAME unit_tests COMMAND alysane_unit_tests)
add_test(NAME selftest COMMAND alysane selftest)

# Full-scale stress harness (no weights): exercises the post-trace gate path at
# production dimensions under sanitizers.
add_executable(alysane_stress tests/stress_gates.cpp ${ALYSANE_SOURCES})
target_link_libraries(alysane_stress PRIVATE ${OPENBLAS} ${LAPACKE} OpenMP::OpenMP_CXX)
