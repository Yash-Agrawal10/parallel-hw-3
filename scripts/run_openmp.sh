#!/bin/bash

#SBATCH -J parallel-hw5-openmp
#SBATCH -o ./output/openmp/%j-openmp.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:10:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running: ./bin/openmp $@"
mkdir -p ./output/openmp
srun ./bin/openmp "$@"