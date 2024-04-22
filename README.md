# Generative model for insilico painting
*This is the algorithm that currently places first in ISBI 2024 Light [My Cell competition](https://lightmycells.grand-challenge.org/evaluation/phase-2/leaderboard/)*
  
This code base is modified from [Stable-diffusion](https://github.com/CompVis/stable-diffusion) code base, thanks the authors who made their work public!

## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environmentclean.yaml
conda activate ldmbf
```

## Dataset
The models were trained on ~56700 previously unpublished images from 30 data acquisition sites. This dataset is very heterogeneous, including 3 imaging modality (Bright Field, Phase Contrast, Differential Inference Contrast microscopy), 40x - 100x objective, multiple different cell lines etc. Most importantly is the class imbalance (i.e. there's a few order of magnitude difference in number of training data for Actin vs Nucleus), and vaying combination of channels per image (1-3 organelles for each input). There's no field of view with all organelles.
More about the training dataset [here](https://lightmycells.grand-challenge.org/database/).

Final evaluation was performed on a hidden ~300 FOVs, which a submitted docker container will be evaluated for.

## VQGAN

1 VQGAN model was trained for each organnelle, with varying performance. These models can be used separately for your organelle of interest.


### Weights

You can access all organelle checkpoints from here: until April 19, 2024 

```
wget https://ell-vault.stanford.edu/dav/trangle/www/ISBI2024_lmc_checkpoints.zip
unzip ISBI2024_lmc_checkpoints.zip
```

This should contains:
- `BF_to_Nucleus.ckpt`: 722MB.
- `BF_to_Mitochondria.ckpt`: 722MB.
- `BF_to_Actin.ckpt`: 722MB.
- `BF_to_Tubulin.ckpt`: 722MB.


### Docker container:
You can test the winning algorithm on grand challenge platform [here](https://grand-challenge.org/algorithms/lmc_control/).
The code to build and run docker container is also provided in this repo. This docker container takes transmitted light tiff as input, and output 4 same size predicted organelle tiff as outputs.

### Example results

TODO: attached some results here.

## BibTeX
This is a place holder, have not had time to finish the manuscript yet.
```
@misc{le2024tlpainting,
      title={High-Resolution In Silico painting with generative model}, 
      author={Trang Le and Emma Lundberg},
      year={2024},
      eprint={2112.10752}, # TODO: Update this
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Train VQGAN
This command train an vqgan to predict organelle from transmitted light inputs
```
python main.py -t -b configs/autoencoder/lmc_BF_<organelle>.yaml --gpus=0,
```

Each model configuration:
```
  | Name            | Type                     | Params
-------------------------------------------------------------
0 | encoder         | Encoder                  | 22.3 M
1 | decoder         | Decoder                  | 33.0 M
2 | loss            | VQLPIPSWithDiscriminator | 17.5 M
3 | quantize        | VectorQuantizer2         | 24.6 K
4 | quant_conv      | Conv2d                   | 12    
5 | post_quant_conv | Conv2d                   | 12    
-------------------------------------------------------------
58.1 M    Trainable params
14.7 M    Non-trainable params
72.8 M    Total params
291.218   Total estimated model params size (MB)
```