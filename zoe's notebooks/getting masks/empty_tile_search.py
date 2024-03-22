import glob
import numpy as np


paths = glob.glob("/scratch/groups/emmalu/JUMP/processed_tiled/*/outlines/*cell_mask*.npy")
print("total num tiles " + str(len(paths)))
num_empty = 0
with open("empty_tiles.txt", "a") as f:
    for path in paths:
        mask = np.load(path).astype(np.uint8)
        assert mask.shape == (250, 250)
        
        if list(np.unique(mask)) != [0,1]:
            f.write(path + "\n")
            num_empty+=1
        else:
            frac = np.sum(mask)/(mask.shape[0]*mask.shape[1]) #fraction of area covered by cell
            if frac < 0.15:
                f.write(path + "\n")
                num_empty+=1

print("num empty tiles " + str(num_empty))


    
    