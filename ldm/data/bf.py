import cv2
import albumentations
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
from sklearn.model_selection import train_test_split
import tifffile

class BFPaint:

    def __init__(self, group='train', path_to_metadata=None, input_channels=None, output_channels=None, size=512, scale_factor=1, flip_and_rotate=True, return_info=False):

        #Define channels
        self.input_channels = input_channels
        if output_channels == None:
            self.output_channels = input_channels
        else:
            self.output_channels = output_channels
        
        #Define metadata (image path, datasource, indices)
        self.metadata = pd.read_csv(path_to_metadata)
        self.metadata['Type'] = self.metadata.Ch
        self.metadata.loc[self.metadata.Type.isin(['DIC','PC']),'Type'] = 'BF'
        
        self.metadata = self.metadata[self.metadata.Image_id !="image_2542"] # Study_29/image_2542 is empty on all channels
        if output_channels == ['Nucleus']:
            imgs_keep = self.metadata.groupby('Image_id').agg({'Type': set })
            imgs_keep = imgs_keep.iloc[[('Nucleus' in f) for f in imgs_keep.Type]]
            self.metadata = self.metadata[self.metadata.Image_id.isin(imgs_keep.index)]
        # TODO: redo split correctly
        train_data, test_data = train_test_split(self.metadata, test_size=0.05, stratify=self.metadata.Study, random_state=42)
        self.metadata["split"] = ["train" if idx in train_data.index else "validation" for idx in self.metadata.index]
        self.metadata.reset_index(drop=True, inplace=True)
        #print(self.metadata.split.value_counts())
        self.indices = self.metadata[(self.metadata.split==group) & self.metadata.Type.isin(self.input_channels + self.output_channels)].index
        self.image_ids = self.metadata[(self.metadata.split==group) & self.metadata.Type.isin(self.input_channels + self.output_channels)].Image_id
    
        self.return_info = return_info

        self.final_size = int(size*scale_factor)
        self.transforms = [albumentations.geometric.resize.Resize(height=self.final_size, width=self.final_size, interpolation=cv2.INTER_LINEAR)]
        self.flip_and_rotate = flip_and_rotate
        if self.flip_and_rotate: #will rotate by random angle and then horizontal flip with prob 0.5
            self.transforms.extend([albumentations.RandomRotate90(p=1.0),
                            albumentations.HorizontalFlip(p=0.5)])
        self.data_augmentation = albumentations.Compose(self.transforms)

        print(f"Dataset group: {group}, length: {len(self.indices)}, in channels: {self.input_channels},  output channels: {self.output_channels}")

    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, i):
        sample = {}

        #get image
        img_index = self.indices[i]
        info = self.metadata.iloc[img_index].to_dict()
        img_id = info["Image_id"]
        img_df = self.metadata[self.metadata.Image_id == img_id]
        img_df = img_df[img_df.Type=='BF']
        in_chs = []
        for ch in self.input_channels:
            if ch != 'BF':
                in_chs.append(ch)
            else:
                #print(img_df.Id.sample(1).values)
                in_chs.append(img_df.Id.sample(1).values[0].replace(img_id,'')[1:].replace('.ome.tiff',''))
        #print(self.input_channels, self.output_channels)
        if self.input_channels == self.output_channels:
            out_chs = in_chs
        else:
            out_chs = []
            for ch in self.output_channels:
                if ch != 'BF':
                    out_chs.append(ch)
                else:
                    #print(img_df.Id.sample(1).values)   
                    out_chs.append(img_df.Id.sample(1).values[0].replace(img_id,'')[1:].replace('.ome.tiff',''))
                    
        image_height, image_width = info["ImageHeight"], info["ImageWidth"]
        image_id = "/".join([info["Study"], img_id])
        #print(in_chs, out_chs) 
        imarray = image_processing.load_ometiff_image(image_id, in_chs, rescale=True)
        targetarray = image_processing.load_ometiff_image(image_id, out_chs, rescale=True)
        #print(imarray.shape, targetarray.shape, image_height, image_width)
        assert imarray.shape == (image_width, image_height, 3)
        assert targetarray.shape == (image_width, image_height, 3)
        assert image_processing.is_between_0_255(imarray)
        
        transformed = self.data_augmentation(image=imarray, mask=targetarray)
        imarray = transformed["image"]
        targetarray = transformed["mask"]
        
        #print(img_id, imarray.shape, imarray.max(), imarray.dtype, '>>>>>', targetarray.shape, targetarray.max(), targetarray.dtype)
        imarray = image_processing.convert_to_minus1_1(imarray)
        targetarray = image_processing.convert_to_minus1_1(targetarray)
        
        assert imarray.shape == (self.final_size, self.final_size, 3)
        assert targetarray.shape == (self.final_size, self.final_size, 3)
        
        sample.update({"image": imarray, "ref-image": targetarray}) 
        if self.return_info:
            sample["info"] = info

        return sample
