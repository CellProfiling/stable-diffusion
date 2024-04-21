# Generative model for insilico painting
*This is the algorithm that won first place in ISBI 2024 Light [My Cell competition](https://lightmycells.grand-challenge.org/evaluation/phase-2/leaderboard/)*
  
This code base is modified from [Stable-diffusion](https://github.com/CompVis/stable-diffusion) code base, thanks the authors who made their work public!

## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environmentclean.yaml
conda activate ldmbf
```

## Dataset


## VQGAN

1 VQGAN model was trained for each organnelle, with varying performance. These models can be used separately for your organelle of interest.


### Weights

You can access all organelle checkpoints from here: ntil April 19, 2024 

```
wget https://ell-vault.stanford.edu/dav/trangle/www/ISBI2024_lmc_checkpoints.zip
unzip ISBI2024_lmc_checkpoints.zip
```

This should contains:
- `BF_to_Nucleus.ckpt`: 722MB.
- `BF_to_Mitochondria.ckpt`: 722MB.
- `BF_to_Actin.ckpt`: 722MB.
- `BF_to_Tubulin.ckpt`: 722MB.

Evaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0,
5.0, 6.0, 7.0, 8.0) and 50 PLMS sampling
steps show the relative improvements of the checkpoints:
![sd evaluation results](assets/v1-variants-scores.jpg)

### Docker container:
You can test the winning algorithm on grand challenge platform [here](https://grand-challenge.org/algorithms/lmc_control/).
The code to build and run docker container is also provided in this repo. This docker container takes transmitted light tiff as input, and output 4 same size predicted organelle tiff as outputs.

### Image Modification with Stable Diffusion

## BibTeX

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