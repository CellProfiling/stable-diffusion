from omegaconf import OmegaConf
import argparse, os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import gc
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

############# HELPER FUNCTIONS #################
def reconstruct_with_vqgan(x, model):
    z, _, [_, _, indices] = model.encode(x) 
    xrec = model.decode(z) 
    return z, xrec

def sample_from_ldm(x, model):
    #TO DO
    return

def normalize_array(array):
  return (array + 1)/2


def save_imgs(original_arrays, recon_arrays, imgdir, num_exs):
  #TO DO: Need to changnum_exs bc imgs will be saved in batches
  if len(original_arrays) == len(recon_arrays): #recons from autoencoder
    for i in range(1, num_exs+1):
      oringal_array = normalize_array(original_arrays[i])
      recon_array = normalize_array(recon_arrays[i])
      original_img = Image.fromarray(oringal_array).convert('RGB')
      recon_img = Image.fromarray(recon_array).convert('RGB')
      original_img.save(f"{imgdir}/real_{str(i)}.png")
      recon_img.save(f"{imgdir}/fake_{str(i)}.png")
  else: #samples from ldm
    assert len(recon_arrays) % len(original_arrays), "wrong number of samples "
    #TO DO save smampled imgs
  return
#################################################### 

def main(opt):
  config = OmegaConf.load(opt.config_path)
  checkpoint = opt.checkpoint
  savedir = opt.savedir
  num_exs = opt.num_exs
  scale = opt.scale
  steps = opt.steps

  #Constructing savedir names
  model_name = checkpoint.split("/checkpoints")[0].split("/")[-1]
  savedir = f"{savedir}/{model_name}"
  imgdir = f"{savedir}/images"
  
  
  #Get Dataloader
  data_config = config['data']
  data = instantiate_from_config(data_config)
  data.prepare_data()
  data.setup()


  #Get Model
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model = instantiate_from_config(config['model'])
  model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=False)
  model = model.to(device)
  model.eval()


  batch_size = 4
  originals = []

  with torch.no_grad():
    with model.ema_scope():
      for i in range(0, len(data.datasets["test"]), batch_size): #loop through number of batches
        print("Loop step: " + str(i))
        #get batch
        lim = min([batch_size*(i+1), len(data.datasets["test"])])
        batch = [data.datasets["test"][j] for j in range(i, lim)]

        #Reformat batch to dict
        collated_batch = dict()
        for k in batch[0].keys():
          collated_batch[k] = [x[k] for x in batch]
          if isinstance(batch[0][k], (np.ndarray, np.generic)):
            collated_batch[k] = torch.tensor(collated_batch[k]).to(device)
        
        #Get inputs
        #originals = np.array([ex["image"] for ex in batch])
        #originals = np.transpose(originals, (0, 3, 1, 2))
        #originals = torch.from_numpy(originals).to(device)

        if "auto" in model_name:
          #embeddings, recons = reconstruct_with_vqgan(originals, model)
          recons = model(collated_batch)
          m = torch.mean(recons).to('cpu')
          #del embeddings
          #del recons
          #del originals
          #gc.collect()

        if "ldm" in model_name:
          sampler = DDIMSampler(model)

          c = model.cond_stage_model(collated_batch)
          uc = dict()

          #I only use c_concat bc I condition with imgs but NOT text
          if "c_concat" in c:
              #c = conditioning
              #uc = unconditioning
              #uc helps to guide sampler away from random noise from generator
              #pushes model to be more specific towards reference img
              #From me uc is unconditioned on reference img --> set toall 0s 
              uc['c_concat'] = [torch.zeros_like(v) for v in c['c_concat']]

          shape = (c['c_concat'][0].shape[1],)+c['c_concat'][0].shape[2:] #shape of tensor to randomly sample
          samples_ddim, notsurewhatthisis = sampler.sample(S=steps, conditioning=c, batch_size=c['c_concat'][0].shape[0], shape=shape, unconditional_guidance_scale=scale, unconditional_conditioning=uc, verbose=False)
          sampled_recons = model.decode_first_stage(samples_ddim)

          #references = [ex["ref-image"] for ex in batch]
          #TO DO: set reference imgs properly
          #embeddings, recons = sample_from_ldm(originals, model)

        #if i < 6:
          #save_imgs(originals, recons, imgdir)
        #df = compute_differences(originals, recons, )
      

    #originals = np.array([sample["image"] for sample in data.datasets["test"]])
    #img_ids = [sample["image"] for sample in data.datasets["test"]["info"]["image_id"]]
    #originals = np.transpose(originals, (0, 3, 1, 2))
    #originals = torch.from_numpy(originals).to(device)
    #embeddings, recons = reconstruct_with_vqgan(originals, model)

    #saving
    #np.save(f"{savedir}/originals.npy", originals)
    #print(originals.shape)
    #np.save(f"{savedir}/reconstructed.npy", recons)
    #print(recons.shape)
    #np.save(f"{savedir}/embeddings.npy", embeddings)
    #print(embeddings.shape)
    #save_imgs(originals, recons, imgdir)
  

  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Reconstruct test imgs with autoencoder or sample images from ldm. Example command: python analysis/reconstruct.py --config=configs/autoencoder/jump_autoencoder__r45__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-01-31T07-58-47_jump_autoencoder__r45__fov512/checkpoints/last.ckpt --savedir=/scratch/groups/emmalu/JUMP_HPA_validation/ --num_exs=100")
  parser.add_argument(
    "--config_path",
    type=str,
    nargs="?",
    help="the model config",
    )
  parser.add_argument(
    "--checkpoint",
    type=str,
    nargs="?",
    help="the model checkpoint",
  )
  parser.add_argument(
    "--savedir",
    type=str,
    default="/scratch/groups/emmalu/JUMP_HPA_validation/",
    nargs="?",
    help="where to save npy file",
  )
  parser.add_argument(
    "--num_exs",
    type=int,
    default=100,
    nargs="?",
    help="number of example images to save",
  )
  parser.add_argument(
    "--num_samples",
    type=int,
    default=100,
    nargs="?",
    help="number of images to sample from ldm per input",
  )
  parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="unconditional guidance scale",
  )
  parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
  )

  opt = parser.parse_args()
  main(opt)