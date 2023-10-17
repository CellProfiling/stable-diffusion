# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python [conda env:ldm2]
#     language: python
#     name: conda-env-ldm2-py
# ---

# +
from functools import partial
import itertools
import json
import math
from multiprocessing import Pool
import os
import pickle
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import cv2
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
np.bool = bool
import pandas as pd
from PIL import Image
from skimage import measure, morphology, segmentation
import torch.nn as nn
from tqdm import tqdm, trange
# -

# # Load meta data

TOTAL_LENGTH = 247678
smc_data_path = "/scratch/users/xikunz2/stable-diffusion"
image_path = f"{smc_data_path}/hpa-webdataset-all-composite/images2"
mask_path = f"{smc_data_path}/hpa-webdataset-all-composite/masks2"
with open(f"{smc_data_path}/hpa-webdataset-all-composite/HPACombineDatasetInfo.pickle", 'rb') as fp:
    info_list = pickle.load(fp)
assert len(info_list) == TOTAL_LENGTH

info_list[0]

with open(f"{smc_data_path}/hpa-webdataset-all-composite/stage1_data_split_flt4.json") as in_file:
    idcs = json.load(in_file)
train_indexes = idcs["train"]
valid_indexes = idcs["validation"]
all_indexes = train_indexes + valid_indexes
len(all_indexes), len(train_indexes)


# # Get segmentation masks

# +
def get_masks(segmentator, image_path, mask_path):
    # for idx, row in tqdm(no_mask_df.iterrows(), total=len(no_mask_df)):
    c = 0
    for idx in tqdm(all_indexes):
        info = info_list[idx]
        plate_id = info["if_plate_id"]
        position = info["position"]
        sample = info["sample"]
        name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)
        if not os.path.exists(f"{mask_path}/{name_str}_cellmask.png") or not os.path.exists(f"{mask_path}/{name_str}_nucleimask.png"):
            c += 1
            im = Image.open(f'{image_path}/{idx}.tif')
            imarray = np.array(im)
            assert imarray.shape == (1024, 1024, 4)
            image_mt, image_er, image_nuc = imarray[:, :, 0], imarray[:, :, 3], imarray[:, :, 2]

            image = [[image_mt], [image_er], [image_nuc]]

            nuc_segmentation = segmentator.pred_nuclei(image[2])
            cell_segmentation = segmentator.pred_cells(image)

            # post-processing
            nuclei_mask = label_nuclei(nuc_segmentation[0])
            nuclei_mask, cell_mask = label_cell(nuc_segmentation[0], cell_segmentation[0])

            assert np.max(nuclei_mask) > 0 and np.max(cell_mask) > 0, f"No nuclei or cell mask found for {idx}"
            # fig, axes = plt.subplots(1, 2)
            # axes[0].imshow(nuclei_mask)
            # axes[1].imshow(cell_mask)

            im = Image.fromarray(cell_mask)
            im.save(f"{mask_path}/{name_str}_cellmask.png")
            im = Image.fromarray(nuclei_mask)
            im.save(f"{mask_path}/{name_str}_nucleimask.png")
            # break
    print(c)


NUC_MODEL = f"{smc_data_path}/nuclei_model.pth"
CELL_MODEL = f"{smc_data_path}/cell_model.pth"
segmentor = cellsegmentator.CellSegmentator(
    NUC_MODEL, CELL_MODEL, device="cuda", padding=True, multi_channel_model=True, scale_factor=0.5
)
for m in itertools.chain(segmentor.nuclei_model.modules(), segmentor.cell_model.modules()):
    if isinstance(m, nn.Upsample):
        m.recompute_scale_factor = None
# get_masks(segmentor, image_path, mask_path)
# -

# # Clean up segmentation masks and get bounding box coordinates

# +
class MyPool:
    def __init__(self, processes, chunksize, initializer, initargs):
        assert type(processes) == int
        assert processes >= 1 or processes == -1
        if processes == -1:
            processes = None
        self.processes = processes
        if processes == 1:
            self.map_func = map
            if initializer is not None:
                initializer(*initargs)
        else:
            self.pool = Pool(processes, initializer=initializer, initargs=initargs)
            self.map_func = self.pool.imap
            if processes is not None:
                self.map_func = partial(self.map_func, chunksize=chunksize)
     
    def __enter__(self):
        if self.processes != 1:
            self.pool.__enter__()
        return self
 
    def __exit__(self, *args):
        if self.processes != 1:
            return self.pool.__exit__(*args)

def get_single_cell_mask(
    cell_mask,
    nuclei_mask,
    closing_radius=7,
    rm_border=True,
    remove_size=1000,
    merge_nuclei=True
):
    init_total_labels = len(np.unique(nuclei_mask)) - 1

    if rm_border:
        nuclei_mask = segmentation.clear_border(nuclei_mask, buffer_size=5)
        keep_value = np.unique(nuclei_mask)
        # borderedcellmask = np.array([[x_ in keep_value for x_ in x] for x in cell_mask]).astype("uint8")
        borderedcellmask = np.isin(cell_mask, keep_value).astype("uint8")
        cell_mask = cell_mask * borderedcellmask
        num_removed_border = init_total_labels - len(keep_value) + 1
        # print(f"Removed {num_removed} cells from border out of {init_total_labels} cells", flush=True)

    if merge_nuclei:
        ### see if nuclei are touching and merge them
        bin_nuc_mask = nuclei_mask > 0
        cls_nuc = morphology.closing(bin_nuc_mask, morphology.disk(closing_radius))
        # get the labels of touching nuclei
        new_label_map = morphology.label(cls_nuc)
        new_label_idx = np.unique(new_label_map)[1:]

        new_cell_mask = np.zeros_like(cell_mask)
        new_nuc_mask = np.zeros_like(nuclei_mask)
        for new_label in new_label_idx:
            # get the label of the touching nuclei
            old_labels = np.unique(nuclei_mask[new_label_map == new_label])
            old_labels = old_labels[old_labels != 0]

            new_nuc_mask[np.isin(nuclei_mask, old_labels)] = new_label
            new_cell_mask[np.isin(cell_mask, old_labels)] = new_label
    else:
        new_cell_mask = cell_mask
        new_nuc_mask = nuclei_mask
    num_remove_merge = init_total_labels - num_removed_border - len(np.unique(new_nuc_mask)) + 1

    # assert set(np.unique(new_nuc_mask)) == set(np.unique(new_cell_mask))

    region_props_cell = measure.regionprops(new_cell_mask, intensity_image=(new_cell_mask > 0).astype(np.uint8))
    region_props_nuc = measure.regionprops(new_nuc_mask, intensity_image=(new_nuc_mask > 0).astype(np.uint8))

    region_props = [region_props_cell[i] for (i, x) in enumerate(region_props_nuc) if x.area > remove_size]
    # print(f"Removed {len(region_props_cell) - len(region_props)} out of {len(region_props_cell)} cells", flush=True)
    num_remove_size = len(region_props_cell) - len(region_props)

    assert init_total_labels - num_removed_border - num_remove_merge - num_remove_size == len(region_props), (  # noqa
        init_total_labels,
        num_removed_border,
        num_remove_merge,
        num_remove_size,
        len(region_props),
    )

    if len(region_props) == 0:
        return new_cell_mask, new_nuc_mask, None, None, None, None, None, None
    else:
        bbox_array = np.array([x.bbox for x in region_props])
        ## convert x1,y1,x2,y2 to x,y,w,h
        bbox_array[:, 2] = bbox_array[:, 2] - bbox_array[:, 0]
        bbox_array[:, 3] = bbox_array[:, 3] - bbox_array[:, 1]

        com_array = np.array([x.weighted_centroid for x in region_props])
        bbox_label = np.array([x.label for x in region_props])

        cell_area = np.array([x.area for x in region_props])
        nuc_area = np.array([x.area for x in region_props_nuc if x.area > remove_size])
        return (
            new_cell_mask,
            new_nuc_mask,
            bbox_array,
            com_array,
            bbox_label,
            cell_area,
            nuc_area,
            [
                num_removed_border,
                num_remove_merge,
                num_remove_size,
            ],
        )

filtered_info_list = [info_list[idx] for idx in all_indexes]


# +
def get_bbox_coords(info):
    plate_id = info["if_plate_id"]
    position = info["position"]
    sample = info["sample"]
    name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)
    # if name_str in images_w_bboxes:
    #     return None
    # else:
    im = Image.open(f"{mask_path}/{name_str}_cellmask.png")
    cell_mask = np.array(im)
    im = Image.open(f"{mask_path}/{name_str}_nucleimask.png")
    nuclei_mask = np.array(im)
    new_cell_mask, new_nuc_mask, bbox_array, com_array, bbox_label, cell_area, nuc_area, removed = get_single_cell_mask(
        cell_mask, nuclei_mask)
    return bbox_array, com_array, bbox_label, cell_area, nuc_area, removed

# n_cpus = 1
# with MyPool(n_cpus, chunksize=16, initializer=None, initargs=None) as p:
#     bbox_coords_list = list(tqdm(p.map_func(get_bbox_coords, filtered_info_list), total=len(all_indexes)))


# +
# with open(f"{smc_data_path}/hpa-webdataset-all-composite/bbox_coords.pkl", "wb") as out_file:
#     pickle.dump(bbox_coords_list, out_file)
# -

with open(f"{smc_data_path}/hpa-webdataset-all-composite/bbox_coords.pkl", "rb") as in_file:
    bbox_coords_list = pickle.load(in_file)
len(bbox_coords_list), bbox_coords_list[:2]

image_id_list, com_y_list, com_x_list = [[] for _ in range(3)]
assert len(filtered_info_list) == len(bbox_coords_list)
for info, (bbox_array, com_array, bbox_label, cell_area, nuc_area, removed) in tqdm(zip(filtered_info_list, bbox_coords_list),
                                                                                   total=len(filtered_info_list)):
    # print(info)
    # print(com_array)
    plate_id = info["if_plate_id"]
    position = info["position"]
    sample = info["sample"]
    name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)
    image_id_list.extend([name_str] * len(com_array))
    im = Image.open(f"{mask_path}/{name_str}_cellmask.png")
    cell_mask = np.array(im)
    im = Image.open(f"{mask_path}/{name_str}_nucleimask.png")
    nuclei_mask = np.array(im)
    assert cell_mask.shape == nuclei_mask.shape
    downsample_factors = cell_mask.shape / np.array([1024, 1024])
    adjusted_com_array = com_array / downsample_factors
    # print(adjusted_com_array)
    com_y_list.extend(adjusted_com_array[:, 0].tolist())
    com_x_list.extend(adjusted_com_array[:, 1].tolist())
com_df = pd.DataFrame({"image_id": image_id_list, "com_y": com_y_list, "com_x": com_x_list})
com_df

com_df.to_csv(f"{smc_data_path}/hpa-webdataset-all-composite/cell_centers.csv")

# # Visualize bounding boxes

info = filtered_info_list[0]
plate_id = info["if_plate_id"]
position = info["position"]
sample = info["sample"]
name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)
# if name_str in images_w_bboxes:
#     return None
# else:
im = Image.open(f"{mask_path}/{name_str}_cellmask.png")
cell_mask = np.array(im)
im = Image.open(f"{mask_path}/{name_str}_nucleimask.png")
nuclei_mask = np.array(im)
new_cell_mask, new_nuc_mask, bbox_array, com_array, bbox_label, cell_area, nuc_area, [num_removed_border, num_remove_merge, num_remove_size] = get_single_cell_mask(
    cell_mask,
    nuclei_mask)
bbox_array[0]

fig, axes = plt.subplots(1, 2)
axes[0].imshow(cell_mask)
axes[1].imshow(nuclei_mask)

np.unique(cell_mask), np.unique(nuclei_mask)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(new_cell_mask)
axes[1].imshow(new_nuc_mask)
c = 0
# Create a Rectangle patch
for bbox, com in zip(bbox_array, com_array):
    # bbox: [y, x, height, width]
    rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2], linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    axes[0].add_patch(rect)
    circ = patches.Circle((com[1], com[0]), color="red")
    axes[0].add_patch(circ)
    c += 1
    if c >= 3:
        break
        

new_cell_mask.shape

# +
bbox = bbox_array[0]

fig, axes = plt.subplots(1, 2)
# axes[0].imshow(new_cell_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])
# axes[0].imshow(new_cell_mask[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]])
axes[0].imshow(new_cell_mask[:1024])
axes[1].imshow(new_cell_mask[:, :1024])
# axes[1].imshow(new_nuc_mask)
# -


