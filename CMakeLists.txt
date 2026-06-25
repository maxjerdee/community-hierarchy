cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

option(BUILD_TESTING "Build Catch2-based native tests" OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Build a reusable core library so other language bindings can link against it
add_library(directedstructure_core STATIC
    src/MCMC_core.cpp
    src/MCMC_proposals.cpp
    src/MCMC_bookkeeping.cpp
    src/helpers.cpp
    src/globals.cpp
    src/fit_result.cpp)
target_include_directories(directedstructure_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(directedstructure_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(directedstructure_core PUBLIC cxx_std_17)
target_compile_options(directedstructure_core PRIVATE -O3 -march=native)

# Python bindings: a small pybind11 module that links to the core library
pybind11_add_module(MCMC_core MODULE bindings/python/bindings.cpp)
target_link_libraries(MCMC_core PRIVATE directedstructure_core)
target_include_directories(MCMC_core PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(MCMC_core PRIVATE cxx_std_17)
set_target_properties(
  MCMC_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                         ${CMAKE_SOURCE_DIR}/python/directedstructure/src/directedstructure)
install(TARGETS MCMC_core LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

if(BUILD_TESTING)
  enable_testing()

  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.7.1
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  endif()

  add_executable(mcmc_core_test
    tests/mcmc_core_test.cpp
    src/MCMC_core.cpp
    src/MCMC_proposals.cpp
    src/MCMC_bookkeeping.cpp
    src/helpers.cpp
    src/globals.cpp
    src/fit_result.cpp
  )
  target_include_directories(mcmc_core_test PRIVATE tests src)
  target_compile_features(mcmc_core_test PRIVATE cxx_std_17)
  target_compile_options(mcmc_core_test PRIVATE -Wall -Wextra)
  target_link_libraries(mcmc_core_test PRIVATE Catch2::Catch2WithMain)

  include(CTest)
  add_test(NAME mcmc_core_test COMMAND mcmc_core_test)
endif()
