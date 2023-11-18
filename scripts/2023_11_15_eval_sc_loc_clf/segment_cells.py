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
from collections import Counter
import itertools
import json
import math
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
import seaborn as sns
from skimage import measure, morphology, segmentation
import torch.nn as nn
from tqdm import tqdm, trange

from ldm.util import MyPool
# -

# # Load meta data

TOTAL_LENGTH = 247678
smc_data_path = "/scratch/users/xikunz2/stable-diffusion"
image_path = f"{smc_data_path}/hpa-webdataset-all-composite/images2"
mask_path = f"{smc_data_path}/hpa-webdataset-all-composite/masks2"
cleaned_mask_path = f"{smc_data_path}/hpa-webdataset-all-composite/cleaned_masks"
with open(f"{smc_data_path}/hpa-webdataset-all-composite/HPACombineDatasetInfo.pickle", 'rb') as fp:
    info_list = pickle.load(fp)
assert len(info_list) == TOTAL_LENGTH

info_list[0]

with open(f"{smc_data_path}/hpa-webdataset-all-composite/stage1_data_split_flt4.json") as in_file:
    idcs = json.load(in_file)
train_indexes = idcs["train"]
valid_indexes = idcs["validation"]
all_indexes = train_indexes + valid_indexes
print(len(all_indexes), len(train_indexes))
filtered_info_list = [info_list[idx] for idx in all_indexes]


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


# NUC_MODEL = f"{smc_data_path}/nuclei_model.pth"
# CELL_MODEL = f"{smc_data_path}/cell_model.pth"
# segmentor = cellsegmentator.CellSegmentator(
#     NUC_MODEL, CELL_MODEL, device="cuda", padding=True, multi_channel_model=True, scale_factor=0.5
# )
# for m in itertools.chain(segmentor.nuclei_model.modules(), segmentor.cell_model.modules()):
#     if isinstance(m, nn.Upsample):
#         m.recompute_scale_factor = None
# get_masks(segmentor, image_path, mask_path)
# -

# # Clean up segmentation masks and get bounding box coordinates

# +
def clean_up_masks(
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
            bbox_array, # [y, x, height, width], each bbox is [y:y+height, x:x+height]
            com_array, # [com_y, com_x]
            bbox_label,
            cell_area,
            nuc_area,
            [
                num_removed_border,
                num_remove_merge,
                num_remove_size,
            ],
        )

def get_bbox_coords(info):
    plate_id = info["if_plate_id"]
    position = info["position"]
    sample = info["sample"]
    name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)
    if not os.path.exists(f"{cleaned_mask_path}/{name_str}_cellmask.png") or not os.path.exists(f"{cleaned_mask_path}/{name_str}_nucleimask.png"):
        im = Image.open(f"{mask_path}/{name_str}_cellmask.png")
        cell_mask = np.array(im)
        im = Image.open(f"{mask_path}/{name_str}_nucleimask.png")
        nuclei_mask = np.array(im)
        new_cell_mask, new_nuc_mask, bbox_array, com_array, bbox_label, cell_area, nuc_area, removed = clean_up_masks(
            cell_mask, nuclei_mask)
        im = Image.fromarray(new_cell_mask)
        im.save(f"{cleaned_mask_path}/{name_str}_cellmask.png")
        im = Image.fromarray(new_nuc_mask)
        im.save(f"{cleaned_mask_path}/{name_str}_nucleimask.png")
        return bbox_array, com_array, bbox_label, cell_area, nuc_area, removed

n_cpus = os.cpu_count() - 1
with MyPool(n_cpus, chunksize=16, initializer=None, initargs=None) as p:
    bbox_coords_list = list(tqdm(p.map_func(get_bbox_coords, filtered_info_list), total=len(all_indexes)))
# get_bbox_coords(filtered_info_list[0])

# +
# with open(f"{smc_data_path}/hpa-webdataset-all-composite/bbox_coords.pkl", "wb") as out_file:
#     pickle.dump(bbox_coords_list, out_file)
# -

# with open(f"{smc_data_path}/hpa-webdataset-all-composite/bbox_coords.pkl", "rb") as in_file:
#     bbox_coords_list = pickle.load(in_file)
# len(bbox_coords_list), bbox_coords_list[:2]

# # +
# assert len(filtered_info_list) == len(bbox_coords_list)
# def scale_bboxes(inputs):
#     info, (bbox_array, com_array, bbox_label, cell_area, nuc_area, removed) = inputs
#     plate_id = info["if_plate_id"]
#     position = info["position"]
#     sample = info["sample"]
#     name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)
#     cellline = info["atlas_name"]
#     if com_array is None:
#         return [], [], [], [], [], [], [], []
#     else:
#         image_id_list = [name_str] * len(com_array)
#         cellline_list = [cellline] * len(com_array)
#         im = Image.open(f"{mask_path}/{name_str}_cellmask.png")
#         cell_mask = np.array(im)
#         im = Image.open(f"{mask_path}/{name_str}_nucleimask.png")
#         nuclei_mask = np.array(im)
#         assert cell_mask.shape == nuclei_mask.shape
#         downsample_factors = cell_mask.shape / np.array([1024, 1024])
#         adjusted_com_array = com_array / downsample_factors
#         # print(adjusted_com_array)
#         com_y_list = adjusted_com_array[:, 0].tolist()
#         com_x_list = adjusted_com_array[:, 1].tolist()
#         bbox_y_list = (bbox_array[:, 0] / downsample_factors[0]).tolist()
#         bbox_x_list = (bbox_array[:, 1] / downsample_factors[1]).tolist()
#         bbox_height_list = (bbox_array[:, 2] / downsample_factors[0]).tolist()
#         bbox_width_list = (bbox_array[:, 3] / downsample_factors[1]).tolist()
#         return image_id_list, cellline_list, com_y_list, com_x_list, bbox_y_list, bbox_x_list, bbox_height_list, bbox_width_list

# n_cpus = os.cpu_count() - 1
# with MyPool(n_cpus, chunksize=16, initializer=None, initargs=None) as p:
#     scaled_bboxes_list = list(tqdm(p.map_func(scale_bboxes, zip(filtered_info_list, bbox_coords_list)), total=len(all_indexes)))
# # scale_bboxes((filtered_info_list[0], bbox_coords_list[0]))
# # -

# len(scaled_bboxes_list)

# image_id_list, cellline_list, com_y_list, com_x_list, bbox_y_list, bbox_x_list, bbox_height_list, bbox_width_list = [[] for _ in range(8)]
# for image_id_list_, cellline_list_, com_y_list_, com_x_list_, bbox_y_list_, bbox_x_list_, bbox_height_list_, bbox_width_list_ in scaled_bboxes_list:
#     image_id_list.extend(image_id_list_)
#     cellline_list.extend(cellline_list_)
#     com_y_list.extend(com_y_list_)
#     com_x_list.extend(com_x_list_)
#     bbox_y_list.extend(bbox_y_list_)
#     bbox_x_list.extend(bbox_x_list_)
#     bbox_height_list.extend(bbox_height_list_)
#     bbox_width_list.extend(bbox_width_list_)
# scaled_bboxes_df = pd.DataFrame({"image_id": image_id_list, "cell_line": cellline_list, "com_y": com_y_list, "com_x": com_x_list, "bbox_y": bbox_y_list, "bbox_x": bbox_x_list, "bbox_height": bbox_height_list, "bbox_width": bbox_width_list})
# scaled_bboxes_df

# # # Plot the distribution of cell sizes

# scaled_bboxes_df["cell_line"].unique()

# # +
# # scaled_bboxes_df.to_csv(f"{smc_data_path}/hpa-webdataset-all-composite/scaled_bboxes.csv")
# # -

# scaled_bboxes_df[scaled_bboxes_df["bbox_height"] == 1024]

# scaled_bboxes_df[scaled_bboxes_df["bbox_width"] == 1024]

# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# ax = axes[0]
# sns.violinplot(bbox_height_list, ax=ax)
# ax.set_ylabel("Cell height (# pixels in a 1024^2 image)")
# ax.set_title(f"Max: {max(bbox_height_list)}")
# ax = axes[1]
# sns.violinplot(bbox_width_list, ax=ax)
# ax.set_ylabel("Cell width (# pixels in a 1024^2 image)")
# ax.set_title(f"Max: {max(bbox_width_list)}")
# fig.tight_layout()

# small_cells_df = scaled_bboxes_df[(scaled_bboxes_df["bbox_height"] <= 256) & (scaled_bboxes_df["bbox_width"] <= 256)]
# cellline_freq = pd.Series(Counter(scaled_bboxes_df["cell_line"])).sort_values(ascending=False)
# small_cells_cellline_freq = pd.Series(Counter(small_cells_df["cell_line"])).loc[cellline_freq.index]
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# ax = axes[0]
# cellline_freq.plot(kind='bar', ax=ax)
# ax.set_ylabel("Total #cells")
# ax = axes[1]
# small_cells_cellline_freq.plot(kind='bar', ax=ax)
# ax.set_ylabel("#cells smaller than 256^2 in a 1024^2 image")

# # # Visualize bounding boxes

# info = filtered_info_list[0]
# plate_id = info["if_plate_id"]
# position = info["position"]
# sample = info["sample"]
# name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)
# # if name_str in images_w_bboxes:
# #     return None
# # else:
# im = Image.open(f"{mask_path}/{name_str}_cellmask.png")
# cell_mask = np.array(im)
# im = Image.open(f"{mask_path}/{name_str}_nucleimask.png")
# nuclei_mask = np.array(im)
# new_cell_mask, new_nuc_mask, bbox_array, com_array, bbox_label, cell_area, nuc_area, [num_removed_border, num_remove_merge, num_remove_size] = clean_up_masks(
#     cell_mask,
#     nuclei_mask)
# bbox_array[0], com_array[0]

# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(cell_mask)
# axes[1].imshow(nuclei_mask)

# np.unique(cell_mask), np.unique(nuclei_mask)

# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(new_cell_mask)
# axes[1].imshow(new_nuc_mask)
# c = 0
# # Create a Rectangle patch
# for bbox, com in zip(bbox_array, com_array):
#     # bbox: [y, x, height, width]
#     rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2], linewidth=1, edgecolor='r', facecolor='none')
#     # Add the patch to the Axes
#     axes[0].add_patch(rect)
#     circ = patches.Circle((com[1], com[0]), color="red")
#     axes[0].add_patch(circ)
#     c += 1
#     if c >= 1:
#         break
        

# new_cell_mask.shape

# # +
# bbox = bbox_array[0]

# fig, axes = plt.subplots(1, 2)
# # axes[0].imshow(new_cell_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])
# # axes[0].imshow(new_cell_mask[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]])
# axes[0].imshow(new_cell_mask[:1024])
# axes[1].imshow(new_cell_mask[:, :1024])
# # axes[1].imshow(new_nuc_mask)

# # +
# # Each bbox:

# #    [y, x] --------------- [y, x+width]
# #       |                        |
# #       |                        |
# #       |                        |
# #       |                        |
# #       |                        |
# #       |                        |
# # [y+height, x] --------- [y+height, x+width]
# # -

# # # Visualize segmented cells

# # +
# # with open(f"{smc_data_path}/one_sample.pkl", "rb") as in_file:
# #     sample = pickle.load(in_file)
# # sample

# big_bboxes_df = scaled_bboxes_df[(scaled_bboxes_df["bbox_height"] > 512) & (scaled_bboxes_df["bbox_width"] > 512)]
# image_id, com_y, com_x, bbox_y, bbox_x, bbox_height, bbox_width = big_bboxes_df.iloc[0]
# im = Image.open(f'{smc_data_path}/hpa-webdataset-all-composite/images2/{image_id}.tif')
# imarray = np.array(im)
# imarray.shape
# # -

# fig, axes = plt.subplots(1, 2)
# ref = imarray[:, :, [0, 3, 2]]
# prot = imarray[:, :, 1]
# axes[0].imshow(ref)
# axes[1].imshow(prot, cmap="gray")

# fig, axes = plt.subplots(1, 2)
# cell_ref = ref[int(bbox_y):int(bbox_y+bbox_height), int(bbox_x):int(bbox_x+bbox_width)]
# cell_prot = prot[int(bbox_y):int(bbox_y+bbox_height), int(bbox_x):int(bbox_x+bbox_width)]
# axes[0].imshow(cell_ref, vmin=0, vmax=255)
# axes[1].imshow(cell_prot, cmap="gray", vmin=0, vmax=255)

# fig, axes = plt.subplots(1, 2)
# cell_ref = ref[int(com_y)-128:int(com_y)+128, int(com_x)-128:int(com_x)+128]
# cell_prot = prot[int(com_y)-128:int(com_y)+128, int(com_x)-128:int(com_x)+128]
# axes[0].imshow(cell_ref, vmin=0, vmax=255)
# axes[1].imshow(cell_prot, cmap="gray", vmin=0, vmax=255)

# # +
# # with open(f"{smc_data_path}/one_sample.pkl", "rb") as in_file:
# #     sample = pickle.load(in_file)
# # sample

# big_bboxes_df = scaled_bboxes_df[(scaled_bboxes_df["bbox_height"] == 1024) | (scaled_bboxes_df["bbox_width"] == 1024)]
# image_id, com_y, com_x, bbox_y, bbox_x, bbox_height, bbox_width = big_bboxes_df.iloc[0]
# im = Image.open(f'{smc_data_path}/hpa-webdataset-all-composite/images2/{image_id}.tif')
# imarray = np.array(im)
# imarray.shape
# # -

# fig, axes = plt.subplots(1, 2)
# ref = imarray[:, :, [0, 3, 2]]
# prot = imarray[:, :, 1]
# axes[0].imshow(ref)
# axes[1].imshow(prot, cmap="gray")

# fig, axes = plt.subplots(1, 2)
# cell_ref = ref[int(bbox_y):int(bbox_y+bbox_height), int(bbox_x):int(bbox_x+bbox_width)]
# cell_prot = prot[int(bbox_y):int(bbox_y+bbox_height), int(bbox_x):int(bbox_x+bbox_width)]
# axes[0].imshow(cell_ref, vmin=0, vmax=255)
# axes[1].imshow(cell_prot, cmap="gray", vmin=0, vmax=255)

# fig, axes = plt.subplots(1, 2)
# cell_ref = ref[int(com_y)-128:int(com_y)+128, int(com_x)-128:int(com_x)+128]
# cell_prot = prot[int(com_y)-128:int(com_y)+128, int(com_x)-128:int(com_x)+128]
# axes[0].imshow(cell_ref, vmin=0, vmax=255)
# axes[1].imshow(cell_prot, cmap="gray", vmin=0, vmax=255)

# with open("/scratch/users/xikunz2/HPA_SingleCellClassification/kaggle-dataset/CAM_images/image_mask.pkl", "rb") as in_file:
#     [image, mask] = pickle.load(in_file)
# image.shape, mask.shape

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(image[:, :, [0, 3, 2]])
# axes[1].imshow(image[:, :, 1], cmap="gray")
# axes[2].imshow(mask, cmap="gray")

# top, bottom, left, right = (20, 20, 20, 20)
# label_image = measure.label(mask)
# label_image, label_image.shape, np.unique(label_image)

# fig, ax = plt.subplots()
# ax.imshow(label_image, cmap="gray")

# # +
# max_area = 0
# for region in measure.regionprops(label_image):
#     print(region)
#     if region.area > max_area:
#         print(region.area, region.bbox)
#         max_area = region.area
#         min_row, min_col, max_row, max_col = region.bbox

# min_row, min_col = max(min_row - top, 0), max(min_col - left, 0)
# max_row, max_col = min(max_row + bottom, mask.shape[0]), min(max_col + right, mask.shape[1])
# min_row, min_col, max_row, max_col
# # -

# image = image[min_row:max_row, min_col:max_col]
# mask = mask[min_row:max_row, min_col:max_col]

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(image[:, :, [0, 3, 2]])
# axes[1].imshow(image[:, :, 1], cmap="gray")
# axes[2].imshow(mask, cmap="gray")


