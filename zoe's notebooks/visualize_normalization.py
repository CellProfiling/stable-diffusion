import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

#This is a script to test different normalization schemes on ten randomly selected images

def combine_images(columns, space, images, savename):
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([image.shape[1] for image in images])
    height_max = max([image.shape[0] for image in images])
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, imarray in enumerate(images):
        img = Image.fromarray(imarray)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
    background.save(savename)


def pick_imgs(num):
    paths = [[]]
    with open("test_paths.txt", "r") as file:
        all_paths = file.readlines()
        num_paths = len(all_paths)
        random_nums = [random.randrange(num_paths) for i in range(num)]
        paths = [all_paths[i].strip()[5:] for i in random_nums]
    return paths


paths = pick_imgs(10) #may be duplicates, but unlikely

paths_d = {str(i): [] for i in range(1,9)}
for path in paths:
    fov = path.split("ch")[0]
    for i in range(1, 9):
        paths_d[str(i)].append(fov + "ch" + str(i) + "sk1fk1fl1.tiff")

percentiles = [90, 95, 97, 99, 100]
lower_percentiles = [1, 2, 5, 10]
data_dir = "/scratch/groups/emmalu/JUMP/"
#minmax
print("Doing minmax normalization scheme")
for p in percentiles:
    print("Percentile: " + str(p))
    for chan in paths_d.keys():
        imgs_minmax = [] #min-max normalization
        
        for path in paths_d[chan]:
            imarray = np.array(Image.open(data_dir + path))
            pix_max = np.percentile(imarray, 99) 
            pix_min = np.min(imarray)
            
            #lower thresholding for ch1
            #if chan == "1":
                #pix_min = np.percentile(imarray, p) 
                #imarray[imarray < pix_min] = pix_min
            
            imarray[imarray > pix_max] = pix_max
            imarray = (imarray - pix_min) / (pix_max-pix_min) #(0,1) pixel range
            imarray = imarray * 255 #convert to (0,255) pixel range
            assert np.all(imarray <= 255) and np.all(imarray >= 0)
            imgs_minmax.append(imarray)

        savename = "visualize_percentiles/minmaxnorm2_" + chan + "_p" + str(p) + ".png"
        combine_images(columns=5, space=20, images=imgs_minmax, savename=savename)

#zscore
print("Doing zscore normalization scheme")
for chan in paths_d.keys():
    imgs_zscore = [] # zscore normalization
    for path in paths_d[chan]:
        imarray = np.array(Image.open(data_dir + path))
        mean = np.mean(imarray)
        std = np.std(imarray)
        imarray = (imarray - mean)/std
        imarray = (imarray - np.min(imarray))/(np.max(imarray)-np.min(imarray)) # convert to (0,1)
        imarray = imarray * 255
        assert np.all(imarray <= 255) and np.all(imarray >= 0)
        imgs_zscore.append(imarray)

    savename = "visualize_percentiles/zscore_" + chan + ".png"
    #combine_images(columns=5, space=20, images=imgs_zscore, savename=savename)


        

