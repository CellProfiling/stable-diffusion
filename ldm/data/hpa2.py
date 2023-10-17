import cv2
import albumentations
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import os
import json
import h5py
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

try:
   import cPickle as pickle
except:
   import pickle

from ldm.data.serialize import TorchSerializedList

HPA_DATA_ROOT = os.environ.get("HPA_DATA_ROOT", "/data/wei/hpa-webdataset-all-composite")

TOTAL_LENGTH = 247678

location_mapping = {"Actin filaments": 0, "Aggresome": 1, "Cell Junctions": 2, "Centriolar satellite": 3, "Centrosome": 4, "Cytokinetic bridge": 5, "Cytoplasmic bodies": 6, "Cytosol": 7, "Endoplasmic reticulum": 8, "Endosomes": 9, "Focal adhesion sites": 10, "Golgi apparatus": 11, "Intermediate filaments": 12, "Lipid droplets": 13, "Lysosomes": 14, "Microtubule ends": 15, "Microtubules": 16, "Midbody": 17, "Midbody ring": 18, "Mitochondria": 19, "Mitotic chromosome": 20, "Mitotic spindle": 21, "Nuclear bodies": 22, "Nuclear membrane": 23, "Nuclear speckles": 24, "Nucleoli": 25, "Nucleoli fibrillar center": 26, "Nucleoplasm": 27, "Peroxisomes": 28, "Plasma membrane": 29, "Rods & Rings": 30, "Vesicles": 31, "nan": 32}


class HPA:

    all_densenet_avg_list = []

    def __init__(self, group='train', channels=None, include_location=False, include_densenet_embedding=False, return_info=False, filter_func=None, rotate_and_flip=False, use_uniprot_embedding=False, size=256, crop="None"):
        with open(f"{HPA_DATA_ROOT}/HPACombineDatasetInfo.pickle", 'rb') as fp:
            self.info_list = TorchSerializedList(pickle.load(fp))
        assert len(self.info_list) == TOTAL_LENGTH

        self.cell_centers_df = pd.read_csv(f"{HPA_DATA_ROOT}/cell_centers.csv")

        assert group in ['train', 'validation']

        train_indexes, valid_indexes = self.filter_and_split(filter_func)
        self.indexes = train_indexes if group == "train" else valid_indexes
        self.cell_centers_df = self.cell_centers_df[self.cell_centers_df['hpa_index'].isin(self.indexes)]

        self.use_uniprot_embedding = use_uniprot_embedding
        assert include_densenet_embedding in ["all", "flt", False]
        if include_densenet_embedding and not self.all_densenet_avg_list:
            all_densenet_avg_list = self.compute_avg_densenet_embeds(train_indexes, valid_indexes, include_densenet_embedding)
            assert len(all_densenet_avg_list) == TOTAL_LENGTH
            HPA.all_densenet_avg_list = TorchSerializedList(all_densenet_avg_list)

        self.include_densenet_embedding = include_densenet_embedding
        self.channels = np.array([1, 1, 1]) if channels is None else np.array(channels)
        # self.samples = TorchSerializedList(self.samples)
        self.indexes = TorchSerializedList(self.indexes)
        # assert "info" in self.samples[0]

        self.include_location = include_location
        self.return_info = return_info
        self.crop = crop
        self.size = size
        if crop == "None":
            transforms = [albumentations.SmallestMaxSize(max_size=size)]
        elif crop == "random":
            transforms = [albumentations.RandomCrop(height=size, width=size)]
        elif crop == "center":
            transforms = [albumentations.CenterCrop(height=size, width=size)]
        elif crop == "cells":
            transforms = []
        else:
            raise NotImplementedError(f"crop {crop} not implemented")
        if rotate_and_flip:
            transforms.extend([albumentations.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=1.0, interpolation=cv2.INTER_NEAREST),
                albumentations.HorizontalFlip(p=0.5)])
        self.preprocessor = albumentations.Compose(transforms)
        print(f"Dataset group: {group}, length: {len(self.indexes)}, image channels: {self.channels}")

    def filter_and_split(self, filter_func):
        # Construct / Load train and validation indices
        split_by_indexes = f"{HPA_DATA_ROOT}/stage1_data_split_flt4.json"
        with open(split_by_indexes, "r") as in_file:
            idcs = json.load(in_file)
        train_indexes = idcs["train"]
        valid_indexes = idcs["validation"]
        
        if filter_func == 'has_location':
            filter_func = lambda i: str(self.info_list[i]['locations']) != "nan"

            train_indexes = list(filter(filter_func, train_indexes))
            valid_indexes = list(filter(filter_func, valid_indexes))
        return train_indexes, valid_indexes
    
    def compute_avg_densenet_embeds(self, train_indexes, valid_indexes, include_densenet_embedding):
        # Load all densenet embeddings
        # if HPACombineDatasetMetadataInMemory.densenet_embeds is None:
        with open(f"{HPA_DATA_ROOT}/HPACombineDatasetInfo-densenet-features.pickle", "rb") as f:
            densenet_embeds = pickle.load(f)
        assert densenet_embeds.shape == (TOTAL_LENGTH, 1024)

        # Group images of the same protein
        gid_to_nonvalid_imgids = {}
        for index, info in tqdm(enumerate(self.info_list), total=TOTAL_LENGTH, desc="Grouping images of the same protein"):
            if index not in valid_indexes and (include_densenet_embedding == "all" or index in train_indexes):
                gid = info['ensembl_ids']
                if not isinstance(gid, str):
                    continue
                if gid not in gid_to_nonvalid_imgids:
                    gid_to_nonvalid_imgids[gid] = [index]
                else:
                    gid_to_nonvalid_imgids[gid].append(index)

        densent_features_avg = []
        zero_emd_count = 0
        for index in trange(TOTAL_LENGTH, desc="Calculating average densenet embeddings"):
            gid = self.info_list[index]['ensembl_ids']
            if not isinstance(gid, str):
                # no ensembl id
                densent_features_avg.append(np.zeros([1024], dtype='float32'))
                zero_emd_count += 1
                continue
            # assert index in ensembl_ids[gid], f"{index} not in {ensembl_ids[gid]}"
            ids = gid_to_nonvalid_imgids[gid].copy()
            if index in ids:
                ids.remove(index)
            if len(ids) == 0:
                densent_features_avg.append(np.zeros([1024], dtype='float32'))
                zero_emd_count += 1
            else:
                # multi_avg_embeddings.append(index)
                avg_emd = densenet_embeds[ids].mean(axis=0)
                assert avg_emd.sum() != 0
                densent_features_avg.append(avg_emd)
        print(f"There are {zero_emd_count} images with a zero avg densenet embedding.")

        return densent_features_avg

    def crop_and_pad(self, img, center_y, center_x, size):
        """
        Crop a square region from the image and pad with zeros if necessary.

        Parameters:
        - img: numpy array representing the image.
        - center_y: y-coordinate for the center of the crop.
        - center_x: x-coordinate for the center of the crop.
        - size: Side length of the square crop.

        Returns:
        - Cropped and possibly padded image.
        """
        
        h, w = img.shape[:2]
        half_size = size // 2
        
        top = max(0, center_y - half_size)
        left = max(0, center_x - half_size)
        bottom = min(h, center_y + half_size)
        right = min(w, center_x + half_size)

        # Cropping
        cropped = img[top:bottom, left:right]

        # Padding
        pad_top = abs(min(0, center_y - half_size))
        pad_bottom = abs(h - max(h, center_y + half_size))
        pad_left = abs(min(0, center_x - half_size))
        pad_right = abs(w - max(w, center_x + half_size))

        result = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=0)
        return result

    def __len__(self):
        return len(self.cell_centers_df) if self.crop == "cells" else len(self.indexes)

    def __getitem__(self, i):
        if self.crop == "cells":
            cell = self.cell_centers_df.iloc[i]
            com_y, com_x, hpa_index = cell[["com_y", "com_x", "hpa_index"]]
        else:
            hpa_index = self.indexes[i]
        info = self.info_list[hpa_index]
        plate_id = info["if_plate_id"]
        position = info["position"]
        sample = info["sample"]
        image_id = str(plate_id) + "_" + str(position) + "_" + str(sample)
        # sample = dd.io.load(self.cache_file, f'/data_0/data_{hpa_index}').copy()
        im = Image.open(f'{HPA_DATA_ROOT}/images2/{image_id}.tif')
        imarray = np.array(im)
        assert imarray.shape == (1024, 1024, 4)
        
        if self.crop == "cells":
            # Crop the image to the cell and pad with zeros if the borders are out of the image
            imarray = self.crop_and_pad(imarray, int(com_y), int(com_x), self.size)

        imarray = (imarray / 127.5 - 1.0).astype(np.float32) # Convert image to [-1, 1]
        image = imarray[:, :, self.channels]
        ref = imarray[:, :, [0, 3, 2]] # reference channels: MT, ER, DAPI
        assert ref.min() == -1 and ref.max() <= 1
        assert image.min() >= -1
        assert image.min() < 0
        assert image.max() <= 1
        sample = {"image": image, "ref-image": ref, "hpa_index": hpa_index}
        sample["condition_caption"] = f"{info['gene_names']}/{info['atlas_name']}"
        sample["location_caption"] = f"{info['locations']}"
        if self.include_location:
            loc_labels = list(map(lambda n: location_mapping[n] if n in location_mapping else -1, str(info["locations"]).split(',')))
            # create one-hot encoding for the labels
            locations_encoding = np.zeros((len(location_mapping), ), dtype=np.float32)
            locations_encoding[loc_labels] = 1
            sample["location_classes"] = locations_encoding
        # make sure the pixel values should be [0, 1], but the sample image is ranging from -1 to 1
        transformed = self.preprocessor(image=(sample["image"]+1)/2, mask=(sample["ref-image"]+1)/2)
        # restore the range from [0, 1] to [-1, 1]
        sample["image"] = transformed["image"]*2 -1
        sample["ref-image"] = transformed["mask"]*2 -1
        if self.return_info:
            sample["info"] = info

        if self.use_uniprot_embedding:
            # uniprot_indexes = []
            with h5py.File(self.use_uniprot_embedding, "r") as file:
                assert len(info['sequences']) > 0
                for seq in info['sequences']:
                    prot_id = seq.split('|')[1]
                    if prot_id in file:
                        sample['prot_id'] = prot_id
                        sample['embed'] = np.array(file[prot_id])
                                # uniprot_indexes.append(idx)
        if self.include_densenet_embedding:
            sample["densent_avg"] = self.all_densenet_avg_list[hpa_index]
        return sample
