import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import yaml
import numpy as np
import h5py
import pickle

data_config_yaml = """
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 8
    num_workers: 16
    wrap: false
    train:
      target: ldm.data.hpa.HPACombineDatasetMetadataInMemory
      params:
        seed: 123
        group: train
        train_split_ratio: 0.95
        cache_file: /data/wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256-1000-t5.pickle
        channels:
        - 1
        - 1
        - 1
        filter_func: has_location
        rotate_and_flip: true
        include_location: true
        use_uniprot_embedding: /data/wei/stable-diffusion/data/per-protein.h5
    validation:
      target: ldm.data.hpa.HPACombineDatasetMetadataInMemory
      params:
        seed: 123
        group: validation
        train_split_ratio: 0.95
        cache_file: /data/wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256-1000-t5.pickle
        channels:
        - 1
        - 1
        - 1
        filter_func: has_location
        include_location: true
        use_uniprot_embedding: /data/wei/stable-diffusion/data/per-protein.h5
"""




if __name__ == "__main__":
    config = yaml.safe_load(data_config_yaml)
    data_config = config['data']

    # data
    data = instantiate_from_config(data_config)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    ids = []
    with h5py.File("./data/per-protein.h5", "r") as file:
        print(f"number of entries: {len(file.items())}")
        # print(np.array(file['A0A024R1R8']))
        for idx, sample in enumerate(tqdm(data.datasets['validation'])):
            if len(sample['info']['sequences']) > 0:
                prot_id = sample['info']['sequences'][0].split('|')[1]
                if prot_id not in file:
                  print(f"protein id {prot_id} not found in per-protein.h5")
                else:
                  ids.append({'id': idx, 'prot_id': prot_id})
                # assert np.array_equal(sample['embed'], np.array(file[prot_id])), f"embedding mismatch for protein {prot_id}"
            else:
                print(f"skipping sample {sample['info']['filename']}...")
            
            # check if the numpy array equals
        print("done!")
    
    print(f"number of valid ids: {len(ids)}/ {len(data.datasets['validation'])}")
    pickle.dump(ids, open("./data/valid_ids_per_protein_embedding.pkl", "wb"))