import cv2
import albumentations as A
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
#import random
import numpy as np
import tifffile
from analysis.get_embeddings import rescale_2_98
from skimage.filters import threshold_local

location_mapping = {"Mitochondria": 0, "Actin": 1, "Tubulin": 2}
tiff_ch_maping = {"BF": 0, "Nucleus": 1, "Mitochondria": 2, "Actin": 3, "Tubulin": 4}
def read_ome_tiff_rescale(file_id, mode='DIC'):
    d = "/scratch/groups/emmalu/lightmycell/datasets"
    img = tifffile.imread(f'{d}/{file_id}')
    """
    if mode=='DIC':
        if img.dtype == 'uint16':
            img[:,:,0] = 65535 - img[:,:,0] #process_dic(img[:,:,0])
    """
    img_rescale = []
    for ch in range(img.shape[2]):
        img_ = rescale_2_98(img[:,:,ch])
        img_rescale.append(img_)
    img_rescale = np.stack(img_rescale, axis=2)
    return img_rescale
    
def process_dic(img):
    img = 65535 - img # invert bacground
    block_size = 75
    local_thresh = threshold_local(img, block_size, offset=10)
    img_local = img*(img > local_thresh)
    return img_local
    
"""
def artificial_oversample(df, class='TL'):
    class_counts = df[class].value_counts()
    class_ratios = class_counts / len(df)
    # Find the class with the smallest and largest ratio
    smaller_class = class_ratios.idxmin()
    larger_class = class_ratios.idxmax()
    
    # Calculate the number of times to repeat the smaller class samples
    repeat_factor = int(class_ratios[larger_class] / class_ratios[smaller_class])
    
    # Initialize an empty DataFrame to store repeated samples
    repeated_df = pd.DataFrame()
    
    # Repeat the smaller class rows to match the ratio of the larger class
    for class_name, count in class_counts.items():
        if class_name == smaller_class:
            repeat_count = count * repeat_factor
        else:
            repeat_count = count
        class_rows = df[df['class'] == class_name]
        repeated_df = pd.concat([repeated_df, class_rows.sample(repeat_count, replace=True)], ignore_index=True)
    
    balanced_df = pd.concat([df, repeated_df], ignore_index=True)
    return balanced_df
"""

class BFPaint:

    def __init__(self, group='train', path_to_metadata=None, input_channels=None, output_channels=None, is_ldm=False, size=512, scale_factor=1, flip_and_rotate=True, return_info=False, refine=False):

        self.is_ldm = is_ldm # False: AE, True: use inputs as ref-image, outputs as stage_1re
        #Define channels
        self.input_channels = input_channels
        if output_channels == None:
            self.output_channels = input_channels
        else:
            self.output_channels = output_channels
        
        self.metadata = pd.read_csv(path_to_metadata, header=1)
        print(self.metadata.shape)
        print(output_channels, output_channels == ['Actin', 'Actin', 'Actin'])
        #self.metadata = self.metadata[self.metadata.Image_id !="image_2542"]
        if output_channels == ['Actin', 'Actin', 'Actin']:
            self.metadata = self.metadata[[orgs.split(',')[2]=='1' for orgs in self.metadata.organelles]]
        elif output_channels == ['Tubulin', 'Tubulin', 'Tubulin']:
            self.metadata = self.metadata[[orgs.split(',')[3]=='1' for orgs in self.metadata.organelles]]
        elif output_channels == ['Mitochondria','Mitochondria','Mitochondria']:
            self.metadata = self.metadata[[orgs.split(',')[1]=='1' for orgs in self.metadata.organelles]]
        elif output_channels == ['Nucleus','Nucleus','Nucleus']:
            self.metadata = self.metadata[[orgs.split(',')[0]=='1' for orgs in self.metadata.organelles]]
        print(self.metadata.TL.value_counts())
        #self.metadata = self.metadata[self.metadata.TL=='DIC']
        #self.metadata = self.metadata[self.metadata.organelles !="1,0,0,0"]
        print('After remove 1 image: ', self.metadata.shape)
        train_data, test_data = train_test_split(self.metadata, test_size=0.05, stratify=self.metadata.Study, random_state=42)
        #if refine:
            
        self.metadata["split"] = ["train" if idx in train_data.index else "validation" for idx in self.metadata.index]
        self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)   
        self.indices = self.metadata[(self.metadata.split==group)].index
        self.image_ids = self.metadata[(self.metadata.split==group)].Id
    
        self.return_info = return_info

        self.final_size = int(size*scale_factor)
        self.transforms = []#A.geometric.resize.Resize(height=self.final_size, width=self.final_size, interpolation=cv2.INTER_LINEAR)]
        self.flip_and_rotate = flip_and_rotate
        if self.flip_and_rotate:
            self.transforms.extend([A.RandomRotate90(p=1.0),
                                    A.HorizontalFlip(p=0.5),
                                    A.RandomResizedCrop(height=self.final_size, width=self.final_size, scale=(0.7,0.95), p=0.5),
                                    ])
                            
        self.transforms.extend([A.geometric.resize.Resize(height=self.final_size, width=self.final_size, interpolation=cv2.INTER_LINEAR)])
        self.data_augmentation = A.Compose(self.transforms)
        self.data_augmentation_simple = A.Compose([A.RandomRotate90(p=1.0),
                                                A.HorizontalFlip(p=0.5),
                                                A.geometric.resize.Resize(height=self.final_size, width=self.final_size, interpolation=cv2.INTER_LINEAR)
                                                ])
        
        print(f"Dataset group: {group}, length: {len(self.indices)}, in channels: {self.input_channels},  output channels: {self.output_channels}")

    def __len__(self):
        return len(self.image_ids) # len(set(self.image_ids)) * 3
        #return 16 # testing #len(self.image_ids)


    def __getitem__(self, i):
        sample = {}
        #get image
        img_index = self.indices[i]
        info = self.metadata.iloc[img_index].to_dict()
        
        file_id = info["Id"]
        
        cond = list(map(int, info["organelles"].split(',')))
        cond = torch.tensor(cond[-3:])
        
        img = read_ome_tiff_rescale(file_id, mode= info['TL'])
        in_chs = [tiff_ch_maping[ch] for ch in self.input_channels]
        out_chs = [tiff_ch_maping[ch] for ch in self.output_channels]
        imarray = img[:,:,in_chs]
        targetarray = img[:,:,out_chs]
        #(imarray.shape, targetarray.shape)
        
        assert image_processing.is_between_0_255(imarray)
        assert image_processing.is_between_0_255(targetarray)
        
        if True: #info['TL'] == 'DIC': #info['Study'] in ['Study_6', 'Study_8', 'Study_9']:
            transformed = self.data_augmentation(image=imarray, mask=targetarray)
        else:
            transformed = self.data_augmentation_simple(image=imarray, mask=targetarray)
        imarray = transformed["image"]
        targetarray = transformed["mask"]
        imarray = (imarray/255).astype('float32')
        targetarray = (targetarray/255).astype('float32')
        #print(imarray.dtype, targetarray.dtype)
        imarray = image_processing.convert_to_minus1_1(imarray)
        targetarray = image_processing.convert_to_minus1_1(targetarray)
        
        assert imarray.shape == (self.final_size, self.final_size, 3)
        assert targetarray.shape == (self.final_size, self.final_size, 3)
        
        if self.is_ldm:
            sample.update({"image": targetarray, 
                           "ref-image": imarray, 
                           "location_classes": cond})
        else:
            sample.update({"image": imarray, "ref-image": targetarray, "location_classes": cond}) 
        if self.return_info:
            sample["info"] = info

        return sample

class BFClassEmbedder(nn.Module):
    def __init__(self, include_location=False, include_ref_image=True, include_bf_mode=False, use_loc_embedding=True, image_embedding_model=None):
        super().__init__()
        self.include_location = include_location
        self.include_ref_image = include_ref_image
        self.include_bf_mode = include_bf_mode
        self.use_loc_embedding = use_loc_embedding
        if self.use_loc_embedding:
            self.loc_embedding = nn.Sequential(
                nn.Linear(len(location_mapping.keys()), 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
        
        if image_embedding_model:
            assert not isinstance(image_embedding_model, dict)
            self.image_embedding_model = instantiate_from_config(image_embedding_model)

    def forward(self, batch, key=None):
        conditions = dict()
        embed = []
        if self.include_bf_mode:
            embed.append(batch["BF_mode"])
        #if "seq_embed" in batch:
        #    embed.append(batch["seq_embed"])
        if self.include_location:
            if self.use_loc_embedding:
                embeder = self.loc_embedding.to(batch["location_classes"].device)
                embed.append(embeder(batch["location_classes"]))
                #print('Embeding locations: ', batch["location_classes"].shape,  batch["location_classes"] ,embeder(batch["location_classes"]))
            else:
                embed.append(batch["location_classes"])
                #print(batch["location_classes"])
            
        if "densent_avg" in batch:
            embed.append(batch["densent_avg"])
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