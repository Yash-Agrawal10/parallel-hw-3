#!/bin/bash

#SBATCH -J parallel-hw5-distributed-gpu
#SBATCH -o ./output/distributed_gpu/%j-distributed_gpu.out
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -p mi2104x

echo "Running: ./bin/distributed_gpu $@"
srun ./bin/distributed_gpu "$@"