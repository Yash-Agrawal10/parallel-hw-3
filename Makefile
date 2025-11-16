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

all: $(OUT_DIR)/sequential $(OUT_DIR)/openmp $(OUT_DIR)/openmpi

$(OUT_DIR):
	@mkdir -p $@

$(OUT_DIR)/sequential: $(SRC_DIR)/sequential.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(OUT_DIR)/openmp: $(SRC_DIR)/openmp.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $< $(OPENMP_LIBS)

$(OUT_DIR)/openmpi: $(SRC_DIR)/openmpi.cpp | $(OUT_DIR)
	$(MPICXX) $(MPICXXFLAGS) -o $@ $<

clean:
	rm -f $(OUT_DIR)/*
