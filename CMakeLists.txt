cmake_minimum_required(VERSION 3.15...3.26)
# scikit-build-core passes SKBUILD_PROJECT_NAME; plain cmake / emcmake do not.
if(DEFINED SKBUILD_PROJECT_NAME)
  project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
else()
  project(directedstructure LANGUAGES CXX)
endif()

option(BUILD_TESTING "Build Catch2-based native tests" OFF)

# Under Emscripten there is no Python: build the wasm module instead.
if(EMSCRIPTEN)
  set(_ds_default_python OFF)
  set(_ds_default_wasm ON)
else()
  set(_ds_default_python ON)
  set(_ds_default_wasm OFF)
endif()
option(DS_BUILD_PYTHON "Build the pybind11 extension module (MCMC_core)" ${_ds_default_python})
option(DS_BUILD_WASM "Build the WebAssembly module (bindings/wasm; configure with emcmake)" ${_ds_default_wasm})

# Core sources: everything except fit.cpp (which uses std::async) so WASM can
# link the same library without threading.  fit.cpp is added back for native builds.
set(DS_CORE_SOURCES
    src/MCMC_core.cpp
    src/MCMC_proposals.cpp
    src/MCMC_merge_split.cpp
    src/MCMC_bookkeeping.cpp
    src/helpers.cpp
    src/globals.cpp
    src/fit_result.cpp)

# Build a reusable core library so other language bindings can link against it
add_library(directedstructure_core STATIC ${DS_CORE_SOURCES})
target_include_directories(directedstructure_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(directedstructure_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(directedstructure_core PUBLIC cxx_std_17)

if(EMSCRIPTEN)
  # C++ exceptions carry error messages to JavaScript.
  target_compile_options(directedstructure_core PUBLIC -fexceptions -O3)
  target_link_options(directedstructure_core PUBLIC -fexceptions)
else()
  target_compile_options(directedstructure_core PRIVATE -O3 -march=native)
  # Native build: include fit.cpp (uses std::async)
  target_sources(directedstructure_core PRIVATE src/fit.cpp)
endif()

if(DS_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  # Python bindings: a small pybind11 module that links to the core library
  pybind11_add_module(MCMC_core MODULE bindings/python/bindings.cpp)
  target_link_libraries(MCMC_core PRIVATE directedstructure_core)
  target_include_directories(MCMC_core PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_features(MCMC_core PRIVATE cxx_std_17)
  set_target_properties(
    MCMC_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                           ${CMAKE_SOURCE_DIR}/python/directedstructure/src/directedstructure)
  install(TARGETS MCMC_core LIBRARY DESTINATION ${PROJECT_NAME})
endif()

if(DS_BUILD_WASM)
  add_subdirectory(bindings/wasm)
endif()

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
    src/MCMC_merge_split.cpp
    src/MCMC_bookkeeping.cpp
    src/helpers.cpp
    src/globals.cpp
    src/fit_result.cpp
    src/fit.cpp
  )
  target_include_directories(mcmc_core_test PRIVATE tests src)
  target_compile_features(mcmc_core_test PRIVATE cxx_std_17)
  target_compile_options(mcmc_core_test PRIVATE -Wall -Wextra)
  target_link_libraries(mcmc_core_test PRIVATE Catch2::Catch2WithMain)

  # Plain-main() tests of the collective moves: the multi-node flip_nodes path
  # (two oracles) and merge/split against exact targets.
  foreach(t flip_nodes_multi_test merge_split_test)
    add_executable(${t} tests/${t}.cpp
      src/MCMC_core.cpp src/MCMC_proposals.cpp src/MCMC_merge_split.cpp
      src/MCMC_bookkeeping.cpp src/helpers.cpp src/globals.cpp)
    target_include_directories(${t} PRIVATE src)
    target_compile_features(${t} PRIVATE cxx_std_17)
  endforeach()

  include(CTest)
  # mcmc_core_test reads examples/data/... relative to the repository root.
  add_test(NAME mcmc_core_test COMMAND mcmc_core_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME flip_nodes_multi_test COMMAND flip_nodes_multi_test)
  add_test(NAME merge_split_test COMMAND merge_split_test)
endif()
