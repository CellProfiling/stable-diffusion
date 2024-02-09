#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --output=Auto_train_%j.out
#SBATCH --cpus-per-task=8      # Number of OpenMP threads for each MPI process/rank
#SBATCH --mem-per-cpu=2000mb   # Per processor memory request
#SBATCH --time=2:00:00      # Walltime in hh:mm:ss or d-hh:mm:ss

conda activate ldm2

srun python main.py -t -b configs/autoencoder/jump_autoencoder__r45__fov512.yaml --gpus=0,