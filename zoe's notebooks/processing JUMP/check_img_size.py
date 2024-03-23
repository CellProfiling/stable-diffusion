import pandas as pd
import numpy as np
from PIL import Image
from imageio import imread




ids = pd.read_csv("../../metadata_for_training/hpa_metadata.csv").image_id
count = 0
s = set()
for i, id in enumerate(ids):
    if i % 500 == 0: print("500 imgs done")
    path = "/scratch/groups/emmalu/HPA_rescaled/" + id + ".tif"
    imarray = imread(path)
    sh = imarray.shape
    s.add(sh)
    if sh != (2048, 2048, 4):
        count += 1

print(s)
print(count)
print(len(ids))