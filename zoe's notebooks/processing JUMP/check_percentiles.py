import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


paths = open("test_paths.txt", "r")

for path in paths:
    path = path.strip()
    image_name = path.split("/")[2] + "_" + path.split("/")[4]

    print("Working on : " + path)
    full_path = "/scratch/groups/emmalu/" + path
    imarray = np.array(Image.open(full_path))
    imarray = imarray.flatten()
    imarray_nonzero = imarray[imarray != 0]
    
    percentiles = [i for i in range(100)]
    y = np.percentile(imarray, percentiles)
    y_nonzero = np.percentile(imarray_nonzero, percentiles)

    
    plt.clf()
    plt.plot(percentiles, y)
    plt.ylabel("Pixel Value")
    plt.title("Pixel Value percentiles")
    plt.savefig("percentile_figs/pix_percentiles_"+ image_name + ".png")

    plt.clf()
    plt.plot(percentiles, y_nonzero)
    plt.ylabel("Pixel Value")
    plt.title("Pixel Value (nonzero) percentiles")
    plt.savefig("percentile_figs/pix_percentiles_non_zero"+ image_name + ".png")

    plt.clf()
    hist, bins = np.histogram(imarray, bins=100)
    plt.hist(imarray, bins=200)
    plt.title("Pixel Value histogram")
    plt.savefig("percentile_figs/pix_his_" + image_name + ".png")