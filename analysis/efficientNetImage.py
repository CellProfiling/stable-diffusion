from omegaconf import OmegaConf
import argparse, os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.evaluation import metrics2
import scipy.ndimage as ndi
from imageio import imread, imwrite
import cv2
import tifffile



############# HELPER FUNCTIONS #################
def reconstruct_with_vqgan(x, model):
    z, _, [_, _, indices] = model.encode(x) 
    xrec = model.decode(z) 
    return xrec

def sample_from_ldm(x, model):
    #TO DO
    return

def normalize_array(array):
  return (array + 1)/2



def order_img(img, channel_order):
  layers = []
  for i in range(1,6):
      j = channel_order.index(i)
      layers.append(img[:, :, j])

  layers = np.array(layers)
  assert np.shape(layers) == (5, 256, 256)
  return layers


def dino_permute_chans(img):
  #DINO has different order of Cell Paint Channels than Jump
  #JUMP: Mito, Golgi (AGP), RNA, ER, Nuc (DAPI)
  #EfficientNet: Nuc (DAPI), RNA, ER, AGP, Mito
  # --> np.transpose(img, (4, 3, 2, 1, 0))
  return img[[4, 2, 3, 1, 0], :, :]

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
  DINOdir = f"{savedir}/DINO_images"
  
  os.makedirs(imgdir, exist_ok=True) 
  os.makedirs(DINOdir, exist_ok=True)

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
  image_evaluator = metrics2.ImageEvaluator(device=device)
  if "ldm" in model_name:
          sampler = DDIMSampler(model)
  model.eval()

  batch_size = 16
  originals = []

  with torch.no_grad():
    with model.ema_scope():
      for i in range(0, len(data.datasets["test"]), batch_size): #loop through number of batches
        print("Loop step: " + str(i))
        #get batch
        lim = min([i+batch_size, len(data.datasets["test"])])
        batch = [data.datasets["test"][j] for j in range(i, lim)]

        #Reformat batch to dict
        collated_batch = dict()
        for k in batch[0].keys():
          collated_batch[k] = [x[k] for x in batch]
          if isinstance(batch[0][k], (np.ndarray, np.generic)):
            collated_batch[k] = torch.tensor(collated_batch[k]).to(device)

        
        if "auto" in model_name:
          recons = reconstruct_with_vqgan(torch.permute(collated_batch['image'], (0, 3, 1, 2)), model)

        elif "ldm" in model_name:
          c = model.cond_stage_model(collated_batch)
          uc = dict()
          if "c_concat" in c: #I only use c_concat bc I condition with imgs but NOT text
              #c = conditioning, uc = unconditioning
              #uc helps to guide sampler away from random noise and to be more specific towards reference img
              uc['c_concat'] = [torch.zeros_like(v) for v in c['c_concat']] #set uc ref img to all 0s 

          #Sample and Decode
          shape = (c['c_concat'][0].shape[1],)+c['c_concat'][0].shape[2:] #shape of tensor to randomly sample
          samples_ddim, notsurewhatthisis = sampler.sample(S=steps, conditioning=c, batch_size=c['c_concat'][0].shape[0], shape=shape, unconditional_guidance_scale=scale, unconditional_conditioning=uc, verbose=False)
          recons = model.decode_first_stage(samples_ddim)
  

        #SAVE IMAGES
        recons = torch.clip(recons, min=-1, max=1) #Is this necessary?
        recons = torch.permute(recons, (0, 2, 3, 1)).to('cpu').detach().numpy()
        originals = collated_batch['image'].to('cpu').detach().numpy() #(bs, chans, h, w)
        masks =  collated_batch['cell-mask'].to('cpu').detach().numpy()
        info = [info for info in collated_batch['info']]
        image_ids = [info["image_id"] for info in collated_batch['info']]
        subtiles = [info["subtile"] for info in collated_batch['info']]
        if "ldm" in model_name:
          refs = collated_batch['ref-image'].to('cpu').detach().numpy()
          
        #normalize
        originals = (originals + 1) / 2 * 255 # [0, 255]
        if "ldm" in model_name:
          refs = (refs + 1) / 2 * 255 # [0, 255]
          if len(model_name.split("__ref")[1].split("__")[0]) == 2:
            refs[:, :, :, 2] = 0 #set last channel to 0 if only two channels
        elif "auto" in model_name:
          if "__r" in model_name and len(model_name.split("__r")[1].split("__")[0]) == 2:
              recons[:, :, :, 2] = 0
          elif "__o" in model_name and len(model_name.split("__o")[1].split("__")[0]) == 2:
              recons[:, :, :, 2] = 0
        recons = (recons + 1) / 2 * 255 # [0, 255]
          
        
        
        if "ldm" in model_name:
          ref = model_name.split("__ref")[1].split("__o")[0]
          ref_channels = [int(c) for c in ref]
      
          out = model_name.split("__o")[1].split("__")[0]
          out_channels = [int(c) for c in out]

          channel_order = ref_channels + out_channels
          assert (len(channel_order) == 5 or len(channel_order) == 6)

          #####Loop through subtiles#####
          for j in range(len(recons)):
            #Save single_cell images for feeding to EfficientNet
            recon = recons[j]
            original = originals[j]
            ref = refs[j][:,:, 0:2]

            original = np.concatenate([ref, original], axis=2)
            original = order_img(original, channel_order)
            orginal = dino_permute_chans(original)
            assert original.shape==(5, 256, 256)

            recon = np.concatenate([ref, recon], axis=2)
            recon = order_img(recon, channel_order)
            recon = dino_permute_chans(recon)
            assert recon.shape==(5, 256, 256)

            #label different cells
            mask = masks[j].astype(np.uint8)
            assert mask.shape == (250, 250)
            labels, num_labels = ndi.label(mask)

              ######Loop through cells########
            for k in range(1, num_labels+1): 
                #get single cell
                single_cell_mask = np.array(labels == k)
                single_cell_mask = single_cell_mask.astype(np.uint8)

                #layer mask like channels
                single_cell_mask = np.expand_dims(single_cell_mask,0)
                single_cell_mask = np.repeat(single_cell_mask, 5, axis=0)
                assert single_cell_mask.shape == (5, 250, 250)

                single_cell_mask = single_cell_mask.transpose((1,2,0))
                single_cell_mask = cv2.resize(single_cell_mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST) #resize
                single_cell_mask = single_cell_mask.transpose((2,0,1))
                assert single_cell_mask.shape == (5, 256, 256)

                #print("mask bits: " + str(np.unique(single_cell_mask)))

                #mask images
                original_sc_masked = original * single_cell_mask 
                recon_sc_masked = recon * single_cell_mask

                #flatten into 1 black and white image with all channels sides by side, last tile is mask
                original_dino_list = list(np.concatenate([original_sc_masked, single_cell_mask[0:1, :, :]*255]))
                recon_dino_list = list(np.concatenate([recon_sc_masked, single_cell_mask[0:1, :, :]*255]))
                
                assert len(original_dino_list) == 6
                assert len(recon_dino_list) == 6

                #original_dino_format = np.transpose(original_dino_format, (1,2,0)).reshape(256, 256*6).astype(np.uint8)
                #recon_dino_format = np.transpose(recon_dino_format, (1,2,0)).reshape(256, 256*6).astype(np.uint8)

                original_dino_format = np.concatenate(original_dino_list, axis=1).astype(np.uint8)
                recon_dino_format = np.concatenate(recon_dino_list, axis=1).astype(np.uint8)
                assert original_dino_format.shape == (256, 256*6)
                assert recon_dino_format.shape == (256, 256*6)

                #save images
                imname = DINOdir + "/real/" + image_ids[j] + "_" + str(subtiles[j]) + "_cell_" + str(k) + ".png"
                os.makedirs(os.path.dirname(imname), exist_ok=True)
                imwrite(imname, original_dino_format)
                imname = DINOdir + "/fake/" + image_ids[j] + "_" + str(subtiles[j]) + "_cell_" + str(k) + ".png"
                os.makedirs(os.path.dirname(imname), exist_ok=True)
                imwrite(imname, recon_dino_format)







  

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
    default="/scratch/groups/emmalu/JUMP_HPA_validation",
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