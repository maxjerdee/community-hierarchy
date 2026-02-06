cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Force Release mode with optimizations matching Makefile (-O3)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "Release flags" FORCE)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Add the MCMC_core Python module
pybind11_add_module(MCMC_core MODULE src/MCMC_core.cpp
                     src/MCMC_proposals.cpp
                     src/MCMC_bookkeeping.cpp
                     src/MCMC_merge_split.cpp
                     src/helpers.cpp)

target_compile_definitions(MCMC_core PRIVATE DIRECTEDSTRUCTURE_PYBIND)

set_property(TARGET MCMC_core PROPERTY CXX_STANDARD 17)
set_target_properties(
  MCMC_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                         ${CMAKE_SOURCE_DIR}/src/directedstructure)
install(TARGETS MCMC_core DESTINATION ${SKBUILD_PROJECT_NAME})