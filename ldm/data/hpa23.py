import cv2
import albumentations
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import h5py
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from ldm.data import image_processing
from ldm.data.serialize import TorchSerializedList

HPA_DATA_ROOT = os.environ.get("HPA_DATA_ROOT", "/scratch/users/xikunz2/cell-generator/data/HPA")

location_mapping = {"Actin filaments": 0, "Aggresome": 1, "Cell Junctions": 2, "Centriolar satellite": 3, "Centrosome": 4, "Cytokinetic bridge": 5, "Cytoplasmic bodies": 6, "Cytosol": 7, "Endoplasmic reticulum": 8, "Endosomes": 9, "Focal adhesion sites": 10, "Golgi apparatus": 11, "Intermediate filaments": 12, "Lipid droplets": 13, "Lysosomes": 14, "Microtubule ends": 15, "Microtubules": 16, "Midbody": 17, "Midbody ring": 18, "Mitochondria": 19, "Mitotic chromosome": 20, "Mitotic spindle": 21, "Nuclear bodies": 22, "Nuclear membrane": 23, "Nuclear speckles": 24, "Nucleoli": 25, "Nucleoli fibrillar center": 26, "Nucleoplasm": 27, "Peroxisomes": 28, "Plasma membrane": 29, "Rods & Rings": 30, "Vesicles": 31, "nan": 32}

matched_location_mapping = {"Actin filaments": 9, "Aggresome": 15, "Centriolar satellite": 12, "Centrosome": 12, "Cytoplasmic bodies": 17, "Cytosol": 16, "Endoplasmic reticulum": 6, "Endosomes": 17, "Focal adhesion sites": 9, "Golgi apparatus": 7, "Intermediate filaments": 8, "Lipid droplets": 17, "Lysosomes": 17, "Microtubules": 10, "Mitochondria": 14, "Mitotic spindle": 11, "Nuclear bodies": 5, "Nuclear membrane": 1, "Nuclear speckles": 4, "Nucleoli": 2, "Nucleoli fibrillar center": 3, "Nucleoplasm": 0, "Peroxisomes": 17, "Plasma membrane": 13, "Vesicles": 17, "nan": 18}
matched_idx_to_location = {v:k for k,v in matched_location_mapping.items()}


def one_hot_encode_locations(locations, location_mapping):
    loc_labels = list(map(lambda n: location_mapping[n] if n in location_mapping else -1, str(locations).split(',')))
    # create one-hot encoding for the labels
    locations_encoding = np.zeros((max(location_mapping.values()) + 1, ), dtype=np.float32)
    locations_encoding[loc_labels] = 1
    return locations_encoding


def decode_one_hot_locations(locations_encoding, idx_to_location):
    loc_labels = [idx_to_location[i] for i, v in enumerate(locations_encoding) if v == 1]
    locations = ','.join(loc_labels)
    return locations


class HPA:

    def __init__(self, group='train', split="imputation", channels=None, include_location=False, include_densenet_embedding=False, return_info=False, rotate_and_flip=False, include_seq_embedding=False, crop_size=512, crop_type="cells", scale_factor=0.5):
        self.metadata = pd.read_csv(f"{HPA_DATA_ROOT}/v23/IF-image-w-splits.csv", index_col=0)
        self.total_length = len(self.metadata)
        cell_masks_metadata = pd.read_csv(f"{HPA_DATA_ROOT}/v23/IF-cells.csv")

        assert group in ['train', 'validation']

        train_indexes, valid_indexes = self.metadata[self.metadata[split + "_split"] == "train"].index, self.metadata[self.metadata[split + "_split"] == "validation"].index
        self.indexes = train_indexes if group == "train" else valid_indexes
        image_ids = set(self.metadata.loc[self.indexes, "image_id"])
        self.cell_masks_metadata = cell_masks_metadata[cell_masks_metadata['ID'].isin(image_ids)]

        assert include_seq_embedding in [False, "prott5-swissprot", "prott5-homo-sapiens"]
        self.include_seq_embedding = include_seq_embedding
        assert include_densenet_embedding in ["all", "flt", False]

        if include_densenet_embedding:
            self.densenet_embeddings, self.imageid_to_embedidx, self.same_prot_embedidcs = self.prepare_densenet_embeds(train_indexes, valid_indexes, include_densenet_embedding)

        self.include_densenet_embedding = include_densenet_embedding
        self.channels = np.array([1, 1, 1]) if channels is None else np.array(channels)
        # self.samples = TorchSerializedList(self.samples)
        self.indexes = TorchSerializedList(self.indexes)
        # assert "info" in self.samples[0]

        self.include_location = include_location
        self.return_info = return_info
        self.crop_type = crop_type
        self.crop_size = crop_size
        if crop_type == "random":
            transforms = [albumentations.RandomCrop(height=crop_size, width=crop_size)]
        elif crop_type == "center":
            transforms = [albumentations.CenterCrop(height=crop_size, width=crop_size)]
        elif crop_type == "cells" or crop_type == "None":
            transforms = []
        else:
            raise ValueError(f"Unknown crop type: {crop_type}")
        self.final_size = int(crop_size * scale_factor)
        transforms.append(albumentations.SmallestMaxSize(max_size=self.final_size, interpolation=cv2.INTER_LINEAR))
        if rotate_and_flip:
            # transforms.extend([albumentations.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=1.0, interpolation=cv2.INTER_NEAREST),
            #     albumentations.HorizontalFlip(p=0.5)])
            raise NotImplementedError("Rotation and flipping not implemented since cell bounding box positions are not rotated and flipped")
        self.preprocessor = albumentations.Compose(transforms)
        print(f"Dataset group: {group}, length: {len(self.indexes)}, image channels: {self.channels}")
    
    def prepare_densenet_embeds(self, train_indexes, valid_indexes, include_densenet_embedding):
        # Load all densenet embeddings
        with h5py.File("/scratch/groups/emmalu/densenet_image_embedding/densenet_embeddings.h5") as f:
            imageid_to_embedidx = dict()
            densenet_embeddings = []
            for i, imageid in enumerate(f.keys()):
                imageid_to_embedidx[imageid] = i
                densenet_embeddings.append(f[imageid])
            densenet_embeddings = np.stack(densenet_embeddings)

        if include_densenet_embedding == "all":
            nonvalid_metadata = self.metadata[~self.metadata.index.isin(valid_indexes)]
        else:
            nonvalid_metadata = self.metadata[self.metadata.index.isin(train_indexes)]
        gid_to_nonvalid_imgids = nonvalid_metadata.groupby('gene_names')['image_id'].apply(list).to_dict()

        same_prot_embedidcs = dict()
        for index in tqdm(self.indexes, desc="Finding images of the same protein"):
            gid = self.metadata.loc[index, 'gene_names']
            if not isinstance(gid, str):
                continue
            imageid = self.metadata.loc[index, "image_id"]
            imageids = gid_to_nonvalid_imgids[gid].copy()
            if imageid in imageids:
                imageids.remove(imageid)
            same_prot_embedidcs[index] = [imageid_to_embedidx[imageid] for imageid in imageids if imageid in imageid_to_embedidx]
        return densenet_embeddings, imageid_to_embedidx, same_prot_embedidcs

    def __len__(self):
        return len(self.cell_masks_metadata) if self.crop_type == "cells" else len(self.indexes)

    def __getitem__(self, i):
        # i = 1960
        if self.crop_type == "cells":
            cell = self.cell_masks_metadata.iloc[i]
            # cell = self.cell_masks_metadata.loc[1214]
            # com_y, com_x, hpa_index = cell[["com_y", "com_x", "hpa_index"]]
            image_height, image_width, bbox_label, hpa_index = cell[["ImageHeight", "ImageWidth", "maskid", "hpa_index"]]
        else:
            hpa_index = self.indexes[i]
            image_height, image_width = self.metadata.loc[hpa_index, ["ImageHeight", "ImageWidth"]]
        info = self.metadata.loc[hpa_index]
        image_id = info["image_id"]
        imarray = image_processing.load_intensity_rescaled_image(image_id)
        assert imarray.shape == (image_height, image_width, 4)
        mask = Image.open(f'/scratch/groups/emmalu/HPA_masks/{image_id}_cellmask.png')
        maskarray = np.array(mask)
        assert maskarray.shape == (image_height, image_width)
        # sample = {"ori_image": np.copy(imarray), "ori_mask": maskarray, "hpa_index": hpa_index, "bbox_label": bbox_label}
        sample = {"hpa_index": hpa_index}
        if self.crop_type == "cells":
            sample["bbox_label"] = bbox_label
            assert (maskarray == bbox_label).sum() > 0
            # Crop the image to the cell and pad with zeros if the borders are out of the image
            imarray, maskarray = image_processing.crop_around_object(imarray, maskarray, bbox_label, self.crop_size)

        assert image_processing.is_between_0_255(imarray)
        sample["condition_caption"] = f"{info['gene_names']}/{info['atlas_name']}"
        sample["location_caption"] = str(info['locations'])
        if self.include_location:
            sample["location_classes"] = one_hot_encode_locations(info["locations"], location_mapping)
        sample["matched_location_classes"] = one_hot_encode_locations(info["locations"], matched_location_mapping)

        transformed = self.preprocessor(image=imarray, mask=maskarray)
        assert transformed["image"].shape == (self.final_size, self.final_size, 4)
        assert transformed["mask"].shape == (self.final_size, self.final_size)
        imarray = transformed["image"]
        maskarray = transformed["mask"]
        prot_image = imarray[:, :, self.channels]
        prot_image = image_processing.convert_to_minus1_1(prot_image)
        ref_image = imarray[:, :, [0, 3, 2]] # reference channels: MT, ER, DAPI
        ref_image = image_processing.convert_to_minus1_1(ref_image)
        top, left, bottom, right = image_processing.get_bbox_from_mask(maskarray, bbox_label)
        bbox_coords = np.array([top, left, bottom - top, right - left])
        sample.update({"image": prot_image, "ref-image": ref_image, "mask": maskarray, "bbox_coords": bbox_coords})
        if self.return_info:
            sample["info"] = info

        if self.include_seq_embedding:
            prott5_version = self.include_seq_embedding[len("prott5-"):]
            with h5py.File(f"/scratch/groups/emmalu/protT5_embeddings/{prott5_version}-per-protein_HPA_agg.h5", "r") as file:
                sample["seq_embed"] = np.array(file[info["ensembl_ids"]])
        if self.include_densenet_embedding:
            if hpa_index not in self.same_prot_embedidcs or not self.same_prot_embedidcs[hpa_index]:
                print(f"The average densenet embedding to use for image {hpa_index} of gene {info['gene_names']} is zero.")
                densent_features_avg = np.zeros([1024], dtype='float32')
            else:
                avg_emd = self.densenet_embeddings[self.same_prot_embedidcs[hpa_index]].mean(axis=0)
                assert avg_emd.sum() != 0
                densent_features_avg = avg_emd
            sample["densent_avg"] = densent_features_avg
        return sample


FUCCI_ROOT = "/scratch/groups/emmalu/cellcycle/datasets"
class Fucci:

    def __init__(self, group='train', split="imputation", channels_in=None, channels_out=None, return_info=False, rotate_and_flip=False, crop_size=512, crop_type="cells", scale_factor=0.5):

        assert group in ['train', 'validation', 'test']
        self.group = group
        if self.group == 'test':
            self.metadata = pd.read_csv(f"{HPA_DATA_ROOT}/v23/IF-image-w-splits.csv", index_col=0)
            self.metadata = self.metadata[self.metadata.latest_version==23]
            self.indexes = self.metadata.index
        else:
            self.metadata = pd.read_csv(f"{FUCCI_ROOT}/fucci_meta.csv")
            train_indexes, valid_indexes = self.metadata[self.metadata[split + "_split"] == "train"].index[:5], self.metadata[self.metadata[split + "_split"] == "validation"].index[:5]
            self.indexes = train_indexes if group == "train" else valid_indexes
            
        self.total_length = len(self.metadata)
        image_ids = set(self.metadata.loc[self.indexes, "image_id"])
        
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.indexes = TorchSerializedList(self.indexes)
        self.return_info = return_info
        self.crop_type = crop_type
        self.crop_size = crop_size
        if crop_type == "random":
            transforms = [albumentations.RandomCrop(height=crop_size, width=crop_size)]
        elif crop_type == "center":
            transforms = [albumentations.CenterCrop(height=crop_size, width=crop_size)]
        elif crop_type == "cells" or crop_type == "None":
            transforms = []
        else:
            raise ValueError(f"Unknown crop type: {crop_type}")
        self.final_size = int(crop_size * scale_factor)
        transforms.append(albumentations.SmallestMaxSize(max_size=self.final_size, interpolation=cv2.INTER_LINEAR))
        if rotate_and_flip:
            # transforms.extend([albumentations.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=1.0, interpolation=cv2.INTER_NEAREST),
            #     albumentations.HorizontalFlip(p=0.5)])
            raise NotImplementedError("Rotation and flipping not implemented since cell bounding box positions are not rotated and flipped")
        self.preprocessor = albumentations.Compose(transforms)
        print(f"Dataset group: {group}, length: {len(self.indexes)}, image channels: {self.channels_in} > {self.channels_out}")

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, i):
        # i = 1960
        hpa_index = self.indexes[i]
        image_height, image_width = self.metadata.loc[hpa_index, ["ImageHeight", "ImageWidth"]]
        info = self.metadata.loc[hpa_index]
        image_id = info["image_id"]
        sample = {"hpa_index": hpa_index, 'image_id': image_id}
        if self.group == 'test':
            imarray = image_processing.load_raw_hpa_image(image_id, self.channels_in, rescale=True)
            targetarray = image_processing.load_raw_hpa_image(image_id, self.channels_out, rescale=False)
        else:
            imarray = image_processing.load_raw_fucci_image(image_id, self.channels_in, rescale=True)
            targetarray = image_processing.load_raw_fucci_image(image_id, self.channels_out, rescale=False)
        assert imarray.shape == (image_height, image_width, 3)
        assert targetarray.shape == (image_height, image_width, 3)
        assert image_processing.is_between_0_255(imarray)

        transformed = self.preprocessor(image=imarray, mask=targetarray)
        assert transformed["image"].shape == (self.final_size, self.final_size, 3)
        assert transformed["mask"].shape == (self.final_size, self.final_size, 3)
        imarray = transformed["image"]
        targetarray = transformed["mask"]
        imarray = image_processing.convert_to_minus1_1(imarray)
        targetarray = image_processing.convert_to_minus1_1(targetarray)
        top, left, bottom, right = (0, self.final_size, self.final_size, 0) #image_processing.get_bbox_from_mask(maskarray, bbox_label)
        bbox_coords = np.array([top, left, bottom - top, right - left])
        sample.update({"image": imarray, "ref-image": targetarray}) # "mask": maskarray, "bbox_coords": bbox_coords})
        if self.return_info:
            sample["info"] = info

        return sample
