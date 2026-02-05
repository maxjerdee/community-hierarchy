cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Add the MCMC_core Python module
pybind11_add_module(MCMC_core MODULE src/MCMC_core.cpp
                     src/MCMC_proposals.cpp
                     src/MCMC_bookkeeping.cpp
                     src/MCMC_merge_split.cpp
                     src/helpers.cpp)
set_target_properties(
  MCMC_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                         ${CMAKE_SOURCE_DIR}/src/directedstructure)
install(TARGETS MCMC_core DESTINATION ${SKBUILD_PROJECT_NAME})