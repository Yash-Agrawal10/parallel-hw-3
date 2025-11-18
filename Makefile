# Flags
COMMON_WARN := -Wall -Wextra -Wpedantic -Wshadow
OPT         := -O2
CXXFLAGS    := -std=c++20 $(COMMON_WARN) $(OPT) 

SRC_DIR := ./src
OUT_DIR := ./bin
OUTPUT_DIR      := ./output
OUTPUT_SUBDIRS  := $(OUTPUT_DIR)/sequential \
                   $(OUTPUT_DIR)/openmp     \
                   $(OUTPUT_DIR)/openmpi    \
                   $(OUTPUT_DIR)/shared_gpu

.PHONY: all clean

all: $(OUT_DIR)/sequential $(OUT_DIR)/openmp $(OUT_DIR)/openmpi $(OUT_DIR)/shared_gpu $(OUT_DIR)/distributed_gpu $(OUTPUT_SUBDIRS)

$(OUTPUT_DIR):
	@mkdir -p $@

$(OUTPUT_SUBDIRS): | $(OUTPUT_DIR)
	@mkdir -p $@

$(OUT_DIR):
	@mkdir -p $@

$(OUT_DIR)/sequential: $(SRC_DIR)/sequential.cpp | $(OUT_DIR)
	g++ $(CXXFLAGS) -o $@ $<

$(OUT_DIR)/openmp: $(SRC_DIR)/openmp.cpp | $(OUT_DIR)
	g++ $(CXXFLAGS) -fopenmp -o $@ $<

$(OUT_DIR)/openmpi: $(SRC_DIR)/openmpi.cpp | $(OUT_DIR)
	mpicxx $(CXXFLAGS) -o $@ $<

$(OUT_DIR)/shared_gpu: $(SRC_DIR)/shared_gpu.hip.cpp | $(OUT_DIR)
	hipcc $(CXXFLAGS)-o $@ $<

$(OUT_DIR)/distributed_gpu: $(SRC_DIR)/distributed_gpu.hip.cpp | $(OUT_DIR)
	OMPI_CXX=hipcc mpicxx $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(OUT_DIR)/*
