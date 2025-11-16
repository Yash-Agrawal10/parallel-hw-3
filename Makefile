# Compilers
CXX      ?= clang++
MPICXX   := mpicxx

# Flags
COMMON_WARN := -Wall -Wextra -Wpedantic -Wshadow
OPT         := -O2
CXXFLAGS   := -std=c++20 $(COMMON_WARN) $(OPT)
MPICXXFLAGS:= -std=c++20 $(COMMON_WARN) $(OPT)

# OpenMP
OPENMP_FLAGS := -fopenmp

SRC_DIR := ./src
OUT_DIR := ./bin

.PHONY: all clean

all: hw4 hw3

hw4: $(OUT_DIR)/hw4/sequential $(OUT_DIR)/hw4/openmp $(OUT_DIR)/hw4/openmpi

hw3: $(OUT_DIR)/hw3/sequential $(OUT_DIR)/hw3/openmp $(OUT_DIR)/hw3/openmpi

$(OUT_DIR): $(OUT_DIR)/hw4 $(OUT_DIR)/hw3

$(OUT_DIR)/hw4:
	@mkdir -p $@

$(OUT_DIR)/hw3:
	@mkdir -p $@

$(OUT_DIR)/hw4/sequential: $(SRC_DIR)/hw4/sequential.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(OUT_DIR)/hw4/openmp: $(SRC_DIR)/hw4/openmp.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $< $(OPENMP_LIBS)

$(OUT_DIR)/hw4/openmpi: $(SRC_DIR)/hw4/openmpi.cpp | $(OUT_DIR)
	$(MPICXX) $(MPICXXFLAGS) -o $@ $<

$(OUT_DIR)/hw3/sequential: $(SRC_DIR)/hw3/sequential.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(OUT_DIR)/hw3/openmp: $(SRC_DIR)/hw3/openmp.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $< $(OPENMP_LIBS)

$(OUT_DIR)/hw3/openmpi: $(SRC_DIR)/hw3/openmpi.cpp | $(OUT_DIR)
	$(MPICXX) $(MPICXXFLAGS) -o $@ $<

clean:
	rm -rf $(OUT_DIR)/*
