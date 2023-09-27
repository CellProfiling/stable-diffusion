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
import os

from PIL import Image
from tqdm import tqdm
import webdataset as wds
# -

HPA_DATA_ROOT = os.environ.get("HPA_DATA_ROOT", "/data/wei/hpa-webdataset-all-composite")

url = f"{HPA_DATA_ROOT}/webdataset_img.tar"
dataset1 = wds.WebDataset(url, nodesplitter=wds.split_by_node).decode().to_tuple("__key__", "img.pyd")

c = 0
for i, ret in tqdm(enumerate(dataset1), total=247678):
    # imgd, infod, embedd = ret
    # assert imgd[0] == infod[0] and imgd[0] == embedd[0]
    # yield {"file_path_": imgd[0], "image": imgd[1], "info": infod[1], "embed": embedd[1]}
    arr = ret[1]
    im = Image.fromarray(arr)
    im.save(f"{HPA_DATA_ROOT}/images2/{i}.tif")
    c += 1
    # if c >= 2:
    #     break

ret[1].shape
