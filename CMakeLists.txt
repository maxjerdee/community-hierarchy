cmake_minimum_required(VERSION 3.15...3.26)
# scikit-build-core passes SKBUILD_PROJECT_NAME; plain cmake / emcmake do not.
if(DEFINED SKBUILD_PROJECT_NAME)
  project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
else()
  project(community_hierarchy LANGUAGES CXX)
endif()

option(BUILD_TESTING "Build the native C++ tests (tests/*.cpp)" OFF)

# Under Emscripten there is no Python: build the wasm module instead.
if(EMSCRIPTEN)
  set(_ds_default_python OFF)
  set(_ds_default_wasm ON)
else()
  set(_ds_default_python ON)
  set(_ds_default_wasm OFF)
endif()
option(DS_BUILD_PYTHON "Build the pybind11 extension module (MCMC_core)" ${_ds_default_python})
option(DS_NATIVE_ARCH "Optimise for the build machine's CPU (-march=native)" OFF)
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
    src/diagnostics.cpp
    src/fit_result.cpp
    src/initial_state.cpp
    src/interactions.cpp
    src/loo.cpp
    src/model_presets.cpp
    src/modularity.cpp)

# Build a reusable core library so other language bindings can link against it
add_library(community_hierarchy_core STATIC ${DS_CORE_SOURCES})
target_include_directories(community_hierarchy_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(community_hierarchy_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(community_hierarchy_core PUBLIC cxx_std_17)

if(EMSCRIPTEN)
  # C++ exceptions carry error messages to JavaScript.
  target_compile_options(community_hierarchy_core PUBLIC -fexceptions -O3)
  target_link_options(community_hierarchy_core PUBLIC -fexceptions)
else()
  # Portable by default, so that published wheels run on any CPU of their
  # architecture; -DDS_NATIVE_ARCH=ON tunes a local build to the build machine.
  if(NOT MSVC)
    target_compile_options(community_hierarchy_core PRIVATE -O3)
    if(DS_NATIVE_ARCH)
      target_compile_options(community_hierarchy_core PRIVATE -march=native)
    endif()
  endif()
  # Native build: include fit.cpp (uses std::async)
  target_sources(community_hierarchy_core PRIVATE src/fit.cpp)
endif()

if(DS_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  # Python bindings: a small pybind11 module that links to the core library
  pybind11_add_module(MCMC_core MODULE bindings/python/bindings.cpp)
  target_link_libraries(MCMC_core PRIVATE community_hierarchy_core)
  target_include_directories(MCMC_core PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_features(MCMC_core PRIVATE cxx_std_17)
  set_target_properties(
    MCMC_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                           ${CMAKE_SOURCE_DIR}/python/community_hierarchy/src/community_hierarchy)
  install(TARGETS MCMC_core LIBRARY DESTINATION community_hierarchy)
endif()

if(DS_BUILD_WASM)
  add_subdirectory(bindings/wasm)
endif()

if(BUILD_TESTING)
  enable_testing()

  # Plain-main() tests (no test framework), each exits non-zero on failure:
  #   mcmc_core_test             input validation, exact DL enumeration, scores_fixed,
  #                              Gaussian quantile / score proposal, plain Bradley-Terry
  #   weighted_bookkeeping_test  weighted-SBM (H_W) and two-layer bookkeeping
  #   flip_nodes_multi_test      the multi-node flip_nodes path (two oracles)
  #   merge_split_test           merge/split against exact targets
  foreach(t mcmc_core_test weighted_bookkeeping_test flip_nodes_multi_test merge_split_test)
    add_executable(${t} tests/${t}.cpp
      src/MCMC_core.cpp src/MCMC_proposals.cpp src/MCMC_merge_split.cpp
      src/MCMC_bookkeeping.cpp src/helpers.cpp src/globals.cpp)
    target_include_directories(${t} PRIVATE src)
    target_compile_features(${t} PRIVATE cxx_std_17)
    if(NOT MSVC)
      target_compile_options(${t} PRIVATE -O2)
    endif()
  endforeach()

  include(CTest)
  foreach(t mcmc_core_test weighted_bookkeeping_test flip_nodes_multi_test merge_split_test)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()
