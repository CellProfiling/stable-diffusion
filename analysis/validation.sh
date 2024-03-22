#!/bin/bash
#SBATCH --job-name=valid
#SBATCH --partition=emmalu
#SBATCH --gpus=1
#SBATCH --gpu_cmode=shared
#SBATCH --time=03:00:00


#srun python validation.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/autoencoder/jump_autoencoder__r45__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-01-31T07-58-47_jump_autoencoder__r45__fov512/checkpoints/last.ckpt

#srun python validation.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/autoencoder/jump_autoencoder__r35__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-02-02T10-36-01_jump_autoencoder__r35__fov512/checkpoints/last.ckpt

#srun python validation.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/autoencoder/jump_autoencoder__r15__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-02-02T10-36-01_jump_autoencoder__r15__fov512/checkpoints/last.ckpt

#srun python validation.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/autoencoder/jump_autoencoder__o123__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-01-31T11-20-10_jump_autoencoder__o123__fov512/checkpoints/last.ckpt

#srun python validation.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/autoencoder/jump_autoencoder__o234__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-02-02T10-32-54_jump_autoencoder__o234__fov512/checkpoints/last.ckpt

#srun python validation.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/autoencoder/jump_autoencoder__o124__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-02-02T10-32-54_jump_autoencoder__o124__fov512/checkpoints/last.ckpt

#srun python validation.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/latent-diffusion/jump__ldm__vq4__ref45__o123__fov512_0.5.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-02-01T14-51-40_jump__ldm__vq4__ref45__o123__fov512_0.5/checkpoints/last.ckpt

#srun python validation.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/latent-diffusion/jump__ldm__vq4__ref35__o124__fov512_0.5.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-02-12T17-47-16_jump__ldm__vq4__ref35__o124__fov512_0.5/checkpoints/last.ckpt

srun python validation.py --config=/home/users/zwefers/lundberg_lab/stable-diffusion/configs/latent-diffusion/jump__ldm__vq4__ref15__o234__fov512_0.5.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-02-12T17-46-57_jump__ldm__vq4__ref15__o234__fov512_0.5/checkpoints/last.ckpt

