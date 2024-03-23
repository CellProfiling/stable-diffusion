import pandas as pd
import numpy as np
import click
from imageio import imread
from glob import glob
import matplotlib.pyplot as plt



@click.command()
@click.argument('datasource')
def get_statistcs(datasource):

    if datasource == "jump":
        data_dir = "/scratch/groups/emmalu/JUMP/raw/"
        channels = ["ch" + str(i) for i in range(1,9)]
    elif datasource == "hpa":
        data_dir = "/scratch/groups/emmalu/HPA_temp/"
        channels = ["red", "blue", "yellow", "green"]
    else:
        print(datasource + " is not a valid datasource")
        return [], [], []
   
    mean_df = pd.DataFrame(columns=channels)
    std_df = pd.DataFrame(columns=channels)
    range_df = pd.DataFrame(columns=channels)

    for chan in channels:
        print(chan)
        means = []
        stds = []
        ranges = []
        for path in glob(data_dir + "*/images/*" + chan + "*.tiff"):

            #get mean pixel value + std for each fov
            imarray = np.array(imread(path))
            if datasource == "jump":
                imarray = imarray #convert to be between 0 and 1
            elif datasource == "hpa":
                imarray = imarray/255
            mean = np.mean(imarray)
            std = np.std(imarray)
            range_len = np.max(imarray)-np.min(imarray)

            #add statistics to dictionaries
            means.append(mean)
            stds.append(std)
            ranges.append(range_len)

        plt.clf()
        plt.figure()
        plt.hist(means, bins=200)
        plt.title("Histogram of JUMP means (per fov)")
        plt.savefig("/home/users/zwefers/lundberg_lab/stable-diffusion/jump-" + chan + "-means.png")
        plt.close()

        plt.clf()
        plt.figure()
        plt.hist(stds, bins=200)
        plt.title("Histogram of JUMP stds (per fov)")
        plt.savefig("/home/users/zwefers/lundberg_lab/stable-diffusion/jump-"+ chan + "-stds.png")
        plt.close()


        plt.clf()
        plt.figure()
        plt.hist(ranges, bins=200)
        plt.title("Histogram of JUMP ranges (per fov)")
        plt.savefig("/home/users/zwefers/lundberg_lab/stable-diffusion/jump-"+ chan + "-ranges.png")
        plt.close()


if __name__ == '__main__':
    get_statistcs()