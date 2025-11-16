#!/bin/bash

#SBATCH -J parallel-hw5-shared-gpu
#SBATCH -o ./output/shared_gpu/%j-shared_gpu.out
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

echo "Running: ./bin/shared_gpu $@"
srun ./bin/shared_gpu "$@"