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

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ldm.data import image_processing
from ldm import util
# -

metadata = pd.read_csv("/scratch/users/xikunz2/stable-diffusion/hpa-webdataset-all-composite/v23/IF-image-w-splits.csv", index_col=0)
metadata


# +
def rescale_image(image_id):
    plate_id = image_id.split("_")[0]
    save_dir = "/scratch/groups/emmalu/HPA_rescaled"
    rescaled_image_path = f"{save_dir}/{image_id}.tif"
    if not os.path.exists(rescaled_image_path):
    # print(image_id)
    # if True:
        try:
            full_res_image = image_processing.load_raw_image(plate_id, image_id)
            Image.fromarray(full_res_image).save(rescaled_image_path)
        except FileNotFoundError:
            pass

# rescale_image(metadata.loc[0, "image_id"])
with util.MyPool(processes=15, chunksize=1000) as p:
    list(tqdm(p.map_func(rescale_image, metadata["image_id"]), total=len(metadata)))
# -


