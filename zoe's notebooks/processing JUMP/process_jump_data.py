import numpy as np
from imageio import imread, imwrite
import albumentations
from glob import glob
import os
import cv2
import pandas as pd


data_dir = "/scratch/groups/emmalu/JUMP/raw/"
#save_dir = "scratch/groups/emmalu/JUMP/processed/"


crop_tiles = [albumentations.Crop(x_min=40+250*i, y_min=40+250*j, x_max=40+250*(i+1), y_max=40+250*(j+1)) for i in range(4) for j in range(4)]

channels = ["ch"+ str(i) for i in range(1, 9)]

channel_mean_thresholds = [6000, 4000, 6000, 3500, 2500]

bad_imgs = []
reasons = []

for path in glob(data_dir + "*/*/*ch1*.tiff"): #get channel 1 image
    imgs_by_channel = [path.split("ch1")[0] + "ch" + str(i) + "sk1fk1fl1.tiff" for i in range(1, 9)]
    for i, img in enumerate(imgs_by_channel):
        chan = i + 1
        imarray = imread(img)

        #Check if mean of fov is an outlier
        if chan < 6:
            if np.mean(imarray) > channel_mean_thresholds[i]:
                bad_imgs.append(img)
                reasons.append("mean")

        if chan == 5:
            if np.std(imarray) < 1000:
                bad_imgs.append(img)
                reasons.append("ch5 uniform (empty img?)")

        p_99 = np.percentile(imarray, 99)
        binary = imarray > p_99

        binary = binary.astype('uint8')

        #min area of blob (in pixels)
        area_of_img = imarray.shape[0] * imarray.shape[1]

        #create blob detector
        blobParams = cv2.SimpleBlobDetector_Params()
        blobParams.filterByArea = True
        blobParams.minArea = area_of_img * 0.015625 #if blob is 1/64th area of img
        blobParams.maxArea = area_of_img
        blobDetector = cv2.SimpleBlobDetector_create(blobParams)
        
        #detect blobs
        blobs = blobDetector.detect(binary)

        if len(blobs) > 0:
            bad_imgs.append(img)
            reasons.append("brightspot")

        #minmax normalize intensity with max = 99th percentile
        pix_min = np.min(imarray)
        imarray[imarray > p_99] = p_99
        imarray = (imarray - pix_min) / (p_99-pix_min)
        imarray = imarray * 255 #(0,255) pixel range
        imarray = np.uint8(imarray)#convert to unit8, need for albumentations
        
        #get stats
        pix_mean = np.mean(imarray)
        pix_std = np.std(imarray)


        assert np.all(imarray <= 255) and np.all(imarray >= 0)


        #save whole image
        img = img.replace("raw", "processed_tiled")
        os.makedirs(os.path.dirname(img), exist_ok=True)
        savename = img.split(".")[0] + ".png"



        for j, crop in enumerate(crop_tiles):
            savetile = savename.replace(".png", "_" + str(j) + ".png")
            tile_array = crop(image=imarray)['image']
            #tile_array = cv2.resize(tile_array, (512, 512))

            #check if tile is empty, if so add to bad list
            if sum((tile_array < (pix_mean - pix_std)).flatten())/len(tile_array)**2 > 0.9: #if more thabn 90% of tile is empty
                bad_imgs.append(savetile)
                reasons.append("empty tile #" + str(j))
            
            imwrite(savetile, tile_array)
                
                


df = pd.DataFrame(columns = [["bad image", "reason"]])
df["bad image"] = bad_imgs
df["reason"] = reasons
df.to_csv("bad_imgs.csv", index=False)
