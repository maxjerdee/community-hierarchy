# Makefile to build the pybind11 extension module and the C++ test harness

PYTHON ?= python3
CXX ?= g++
BUILD_DIR := build
SRC_DIR := src
PKG_DIR := $(SRC_DIR)/directedstructure
MODULE_NAME := MCMC_core
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')")
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)
# Prefer embed flags so executables link against libpython; fall back to standard ldflags
PY_LDFLAGS := $(shell $(PYTHON)-config --embed --ldflags 2>/dev/null || $(PYTHON)-config --ldflags 2>/dev/null || python3-config --embed --ldflags 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")
MKDIR_P := mkdir -p

CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -march=native -fPIC -I$(SRC_DIR) -DDIRECTEDSTRUCTURE_PYBIND

SOURCES := $(SRC_DIR)/MCMC_core.cpp \
		   $(SRC_DIR)/MCMC_proposals.cpp \
		   $(SRC_DIR)/MCMC_bookkeeping.cpp \
		   $(SRC_DIR)/MCMC_merge_split.cpp \
		   $(SRC_DIR)/helpers.cpp
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

TEST_SOURCES := tests/mcmc_core_test.cpp
TEST_OBJECTS := $(patsubst tests/%.cpp,$(BUILD_DIR)/tests_%.o,$(TEST_SOURCES))

MODULE_TARGET := $(PKG_DIR)/$(MODULE_NAME)$(EXT_SUFFIX)
TEST_BIN := $(BUILD_DIR)/mcmc_core_test

.PHONY: all clean test

all: $(MODULE_TARGET) $(TEST_BIN)

$(BUILD_DIR):
	$(MKDIR_P) $(BUILD_DIR)

$(PKG_DIR):
	$(MKDIR_P) $(PKG_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

$(BUILD_DIR)/tests_%.o: tests/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

$(MODULE_TARGET): $(OBJECTS) | $(PKG_DIR)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ $(PY_LDFLAGS)

$(TEST_BIN): $(TEST_OBJECTS) $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(PY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(MODULE_TARGET)
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.so $(PKG_DIR)/$(MODULE_NAME)*.pyd
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.dll
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.dylib
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.pdb
	rm -f $(PKG_DIR)/$(MODULE_NAME)*.exp

# Run the C++ harness
test: $(TEST_BIN)
	./$(TEST_BIN)
