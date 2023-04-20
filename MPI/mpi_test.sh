#!/bin/bash
# Slurm Sbatch Options
#SBATCH --gres=gpu:volta:1
#SBATCH -n 5 -N 5
#SBATCH --output="./mpi_scatter_test.log-%j"
# Loading the required module

export JULIA_CUDA_MEMORY_POOL=none
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

source $HOME/.bashrc
module load cuda/11.6 mpi/openmpi-4.1.3

srun hostname > hostfile
#script
time mpiexec julia gpu_ode_mpi.jl
