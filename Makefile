# Makefile to build the pybind11 extension module and the C++ test harness

PYTHON ?= python3
CXX ?= g++
BUILD_DIR := build
SRC_DIR := src
PKG_DIR := $(SRC_DIR)/community_hierarchy
MODULE_NAME := MCMC_core
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')")
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)
# Prefer embed flags so executables link against libpython; fall back to standard ldflags
PY_LDFLAGS := $(shell $(PYTHON)-config --embed --ldflags 2>/dev/null || $(PYTHON)-config --ldflags 2>/dev/null || python3-config --embed --ldflags 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")
MKDIR_P := mkdir -p

CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -march=native -fPIC -I$(SRC_DIR) -DCOMMUNITY_HIERARCHY_PYBIND
DEBUG_CXXFLAGS := -std=c++17 -Wall -Wextra -O0 -g -fPIC -I$(SRC_DIR) -DCOMMUNITY_HIERARCHY_PYBIND
TEST_CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -march=native -fPIC -I$(SRC_DIR)
TEST_DEBUG_CXXFLAGS := -std=c++17 -Wall -Wextra -O0 -g -fPIC -I$(SRC_DIR)

SOURCES := $(SRC_DIR)/MCMC_core.cpp \
		   $(SRC_DIR)/MCMC_proposals.cpp \
		   $(SRC_DIR)/MCMC_bookkeeping.cpp \
		   $(SRC_DIR)/MCMC_merge_split.cpp \
		   $(SRC_DIR)/helpers.cpp \
		   $(SRC_DIR)/globals.cpp \
		   $(SRC_DIR)/fit_result.cpp \
		   $(SRC_DIR)/fit.cpp
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
TEST_CORE_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%_test.o,$(SOURCES))

TEST_SOURCES := tests/mcmc_core_test.cpp
TEST_OBJECTS := $(patsubst tests/%.cpp,$(BUILD_DIR)/tests_%.o,$(TEST_SOURCES))
DEBUG_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%_debug.o,$(SOURCES))
DEBUG_TEST_CORE_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%_test_debug.o,$(SOURCES))

MODULE_TARGET := $(PKG_DIR)/$(MODULE_NAME)$(EXT_SUFFIX)
TEST_BIN := $(BUILD_DIR)/mcmc_core_test
TEST_BIN_DEBUG := $(BUILD_DIR)/mcmc_core_test_debug
CSV_BIN := AddHealth_analysis/run_sampler_to_csv

.PHONY: all clean test csv

all: $(MODULE_TARGET) $(TEST_BIN)

$(BUILD_DIR):
	$(MKDIR_P) $(BUILD_DIR)

$(PKG_DIR):
	$(MKDIR_P) $(PKG_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

$(BUILD_DIR)/tests_%.o: tests/%.cpp | $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/addhealth_%.o: AddHealth_analysis/%.cpp | $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%_test.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@

$(MODULE_TARGET): $(OBJECTS) | $(PKG_DIR)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ $(PY_LDFLAGS)

$(TEST_BIN): $(TEST_OBJECTS) $(TEST_CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

$(CSV_BIN): $(BUILD_DIR)/addhealth_run_sampler_to_csv.o $(TEST_CORE_OBJECTS)
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

# Debug build for valgrind
$(BUILD_DIR)/%_debug.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(DEBUG_CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

$(BUILD_DIR)/tests_%_debug.o: tests/%.cpp | $(BUILD_DIR)
	$(CXX) $(TEST_DEBUG_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%_test_debug.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(TEST_DEBUG_CXXFLAGS) -c $< -o $@

DEBUG_TEST_OBJECTS := $(patsubst tests/%.cpp,$(BUILD_DIR)/tests_%_debug.o,$(TEST_SOURCES))

$(TEST_BIN_DEBUG): $(DEBUG_TEST_OBJECTS) $(DEBUG_TEST_CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(TEST_DEBUG_CXXFLAGS) $(DEBUG_TEST_OBJECTS) $(DEBUG_TEST_CORE_OBJECTS) -o $@

clean:
	rm -rf $(BUILD_DIR) $(MODULE_TARGET)
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.so $(PKG_DIR)/$(MODULE_NAME)*.pyd
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.dll
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.dylib
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.pdb
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.exp

# Plain-main() exactness tests of the collective moves (also built by CMake)
EXTRA_TESTS := $(BUILD_DIR)/merge_split_test $(BUILD_DIR)/flip_nodes_multi_test
$(BUILD_DIR)/merge_split_test $(BUILD_DIR)/flip_nodes_multi_test: $(BUILD_DIR)/%: $(BUILD_DIR)/tests_%.o $(TEST_CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

# Run the C++ harness and the exactness tests
test: $(TEST_BIN) $(EXTRA_TESTS)
	./$(TEST_BIN)
	./$(BUILD_DIR)/merge_split_test
	./$(BUILD_DIR)/flip_nodes_multi_test

# Build the C++ sampler-to-CSV binary only
csv: $(CSV_BIN)

# Run with valgrind (requires valgrind to be installed)
test-valgrind: $(TEST_BIN_DEBUG)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./$(TEST_BIN_DEBUG)
