#!/bin/bash
#SBATCH --job-name=recon
#SBATCH --output=recon.out
#SBATCH --error=recon.err
#SBATCH --mem=512GB
#SBATCH --partition=emmalu
#SBATCH --gpus=2
#SBATCH --gpu_cmode=shared
#SBATCH --time=01:00:00


#srun python reconstruct.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/autoencoder/jump_autoencoder__r45__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-01-31T07-58-47_jump_autoencoder__r45__fov512/checkpoints/last.ckpt
srun python reconstruct.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/latent-diffusion/jump__ldm__vq4__ref45__o123__fov512_0.5.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-02-01T14-51-40_jump__ldm__vq4__ref45__o123__fov512_0.5/checkpoints/last.ckpt