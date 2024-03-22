import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from imageio import imread, imwrite
import glob
import argparse
import os
import tifffile


def order_img(img, channel_order):
    layers = []
    for i in range(1,6):
       j = channel_order.index(i)
       layers.append(img[:, :, j])

    layers = np.array(layers)
    assert np.shape(layers) == (5, 256, 256)
    return layers



def main(opt):
    path_to_imgs = opt.path_to_imgs

    ref = path_to_imgs.split("__ref")[1].split("__o")[0]
    ref_channels = [int(c) for c in ref]
    
    out = path_to_imgs.split("__o")[1].split("__")[0]
    out_channels = [int(c) for c in out]

    channel_order = ref_channels + out_channels

    assert (len(channel_order) == 5 or len(channel_order) == 6)

    real_imgs = glob.glob(f"{path_to_imgs}/real_*.png")

    for real_img in real_imgs:
        ref_img = real_img.replace("real", "ref")
        recon_img = real_img.replace("real", "fake")

        real_ar = np.array(imread(real_img))
        recon_ar = np.array(imread(recon_img))
        ref_ar = np.array(imread(ref_img))[:,:, 0:2]

        real_ar = np.concatenate([ref_ar, real_ar], axis=2)
        real_ar = order_img(real_ar, channel_order)

        recon_ar = np.concatenate([ref_ar, recon_ar], axis=2)
        recon_ar = order_img(recon_ar, channel_order)


        mask = 

        savedir= path_to_imgs.split("images")[0] + "DINO_images/"

        imdir = savedir+os.path.basename(real_img).replace(".png", ".tif")
        os.makedirs(os.path.dirname(imdir), exist_ok=True)
        tifffile.imwrite(imdir, real_ar)

        imdir = savedir+os.path.basename(recon_img).replace(".png", ".tif")
        os.makedirs(os.path.dirname(imdir), exist_ok=True)
        tifffile.imwrite(imdir, recon_ar)

  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Reconstruct test imgs with autoencoder or sample images from ldm. Example command: python analysis/reconstruct.py --config=configs/autoencoder/jump_autoencoder__r45__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-01-31T07-58-47_jump_autoencoder__r45__fov512/checkpoints/last.ckpt --savedir=/scratch/groups/emmalu/JUMP_HPA_validation/ --num_exs=100")
  parser.add_argument(
    "--path_to_imgs",
    type=str,
    nargs="?",
    help="path to images",
  )


  opt = parser.parse_args()
  main(opt)