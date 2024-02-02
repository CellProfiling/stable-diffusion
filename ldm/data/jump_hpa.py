#from collections import defaultdict
import cv2
import albumentations
#import numpy as np
#from PIL import Image
#from tqdm import tqdm, trange
#import os
#import h5py
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

try:
   import cPickle as pickle
except:
   import pickle

from ldm.data import image_processing
#from ldm.data.serialize import TorchSerializedList
import torch.nn as nn
from einops import rearrange
from ldm.util import instantiate_from_config
import torch
from torchvision.utils import make_grid



class JUMP_HPA:
   
    def __init__(self, group='train', path_to_metadata=None, hpa_ref_channels=None, jump_ref_channels=None, output_channels=None, size=512, scale_factor=1, flip_and_rotate=True, include_smiles=False, inlcude_prots=False, return_info=False):
   
        #Define metadata (image path, datasource, indices)
        self.metadata = pd.read_csv(path_to_metadata).sample(frac=1.0, random_state=42, ignore_index=True) #shuffle metadata
        self.indices = self.metadata[self.metadata["split"]==group].index
        image_ids = set(self.metadata.loc[self.indices, "image_id"])
        self.datasources = self.metadata["datasource"]
        self.return_info = return_info

        #Define channels
        self.hpa_channels = hpa_ref_channels
        self.jump_channels = jump_ref_channels
        self.output_channels = output_channels

        #Define crop, resize, flip/rotate transformations
        self.crop_transforms = [albumentations.Crop(x_min=40+250*i, y_min=40+250*j, x_max=40+250*(i+1), y_max=40+250*(j+1)) for i in range(4) for j in range(4)]
        self.final_size = int(size*scale_factor)
        self.transforms = [albumentations.geometric.resize.Resize(height=self.final_size, width=self.final_size, interpolation=cv2.INTER_LINEAR)]
        self.flip_and_rotate = flip_and_rotate
        if self.flip_and_rotate: #will rotate by random angle and then horizontal flip with prob 0.5
            self.transforms.extend([albumentations.RandomRotate90(p=1.0),
                            albumentations.HorizontalFlip(p=0.5)])  
        self.data_augmentation = albumentations.Compose(self.transforms)
            

        print(f"Dataset group: {group}, length: {len(self.indices)}, jump channels: {self.jump_channels},  hpa channels: {self.hpa_channels}, output channels: {self.output_channels}")

    def __len__(self):
        return len(self.indices)
                   
                   
    def __getitem__(self, i):
        sample = {}

        #get image
        img_index = self.indices[i]
        info = self.metadata.iloc[img_index].to_dict()
        datasource = info["datasource"]
        
        #load ref img
        if datasource == "jump":
            ref_chans = self.jump_channels
        elif datasource == "hpa":
            ref_chans = self.hpa_channels
        
        ref_imarray = image_processing.load_image(datasource, info["image_id"], ref_chans, info["subtile"])
        
        #Augment image
        out_imarray = []
        if datasource == "jump":
            #tile = int(info["subtile"])
            #crop = self.crop_transforms[tile] #only crop jump
            #ref_imarray = crop(image=ref_imarray)['image']

            #load output img
            if self.output_channels is not None: # => jump img
                out_imarray = image_processing.load_image(datasource, info["image_id"], self.output_channels, info["subtile"])
                #out_imarray = crop(image=out_imarray)['image']
                out_imarray = self.data_augmentation(image=out_imarray)['image']
                out_imarray = image_processing.convert_to_minus1_1(out_imarray) #convert to (-1, 1)
                assert out_imarray.shape == (self.final_size, self.final_size, 3)  
       
        ref_imarray = self.data_augmentation(image=ref_imarray)['image']
        ref_imarray = image_processing.convert_to_minus1_1(ref_imarray)
        assert ref_imarray.shape == (self.final_size, self.final_size, 3)      

        
        if self.output_channels is not None: #output array not empty --> training ldm:
            sample.update({"image": out_imarray, "ref-image": ref_imarray, "info": info})
        else: #output imarray is empty --> training autoencoder
            sample.update({"image": ref_imarray, "ref-image": ref_imarray, "info": info})


        if self.return_info:
            sample["info"] = info


        #type of stuff to add later
        #if self.use_smile:
            #sample["smile"] = info["smile"]
        #if self.use_prot:
            #sample["prot_id"] = info["prot_id"]
        
        return sample


class JUMPHPAClassEmbedder(nn.Module):
    def __init__(self, include_smile=False, include_ref_image=True, include_prot=True, image_embedding_model=None):
        super().__init__()
        self.include_ref_image = include_ref_image
        self.include_smile = include_smile
        self.include_prot = include_prot
        
        if image_embedding_model:
            assert not isinstance(image_embedding_model, dict)
            self.image_embedding_model = instantiate_from_config(image_embedding_model)

    def forward(self, batch, key=None):
        conditions = dict()
        embed = []
        if self.include_smile and "smile" in batch:
            embed.append(batch["smile"])
        if self.include_prot and "prot_id" in batch:
            embed.append(batch["prot_id"])
        if embed:
            conditions["c_crossattn"] = embed

        if self.include_ref_image:
            image = batch["ref-image"]
            assert image.shape[3] == 3
            image = rearrange(image, 'b h w c -> b c h w').contiguous()
            with torch.no_grad():
                img_embed = self.image_embedding_model.encode(image)
            if torch.any(torch.isnan(img_embed)):
                raise Exception("NAN values encountered in the image embedding")
            conditions["c_concat"] = [img_embed]
        return conditions

    def decode(self, c):
        assert self.include_ref_image
        condition_row = c['c_concat']
        assert len(condition_row) == 1
        with torch.no_grad():
            condition_rec_row = [self.image_embedding_model.decode(cond) for cond in condition_row]
        n_imgs_per_row = len(condition_rec_row)
        condition_rec_row = torch.stack(condition_rec_row)  # n_log_step, n_row, C, H, W
        condition_grid = rearrange(condition_rec_row, 'n b c h w -> b n c h w')
        condition_grid = rearrange(condition_grid, 'b n c h w -> (b n) c h w')
        condition_grid = make_grid(condition_grid, nrow=n_imgs_per_row)
        return condition_grid