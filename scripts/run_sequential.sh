#!/bin/bash

#SBATCH -J parallel-hw5-sequential
#SBATCH -o ./output/sequential/%j-sequential.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00

echo "Running: ./bin/sequential $@"
mkdir -p ./output/sequential
srun ./bin/sequential "$@"