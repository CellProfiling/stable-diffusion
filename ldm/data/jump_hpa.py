from collections import defaultdict
import cv2
import albumentations
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import os
import h5py
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

try:
   import cPickle as pickle
except:
   import pickle

from ldm.data import image_processing
from ldm.data.serialize import TorchSerializedList


class JUMP_HPA:
   
    def __init__(self, group='train', path_to_metadata=None, hpa_ref_channels=None, jump_ref_channels=None, output_channels=None, size=512, flip_and_rotate=True, include_smiles=False, inlcude_prots=False, return_info=False):
   
        #Define metadata (image path, datasource, indices)
        self.metadata = pd.read_csv(path_to_metadata).sample(frac=1.0, random_state=42, ignore_index=True)[0:1000] #shuffle metadata
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
        self.transforms = [albumentations.geometric.resize.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR)]
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
                out_imarray = image_processing.load_image(datasource, info["image_id"], self.output_channels)
                #out_imarray = crop(image=out_imarray)['image']
                out_imarray = self.data_augmentation(image=out_imarray)['image']
                out_imarray = image_processing.convert_to_minus1_1(out_imarray) #convert to (-1, 1)
                assert out_imarray.shape == (512, 512, 3)  
       
        ref_imarray = self.data_augmentation(image=ref_imarray)['image']
        ref_imarray = image_processing.convert_to_minus1_1(ref_imarray)
        assert ref_imarray.shape == (512, 512, 3)      

        
        if out_imarray: #output array not empty --> training ldm:
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
