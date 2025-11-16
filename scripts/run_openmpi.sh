#!/bin/bash

#SBATCH -J parallel-hw5-openmpi
#SBATCH -o ./output/openmpi/%j-openmpi.out
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:10:00

echo "Running: ./bin/openmpi $@"
mkdir -p ./output/openmpi
srun ./bin/openmpi "$@"