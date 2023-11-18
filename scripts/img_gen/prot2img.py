import argparse, os, datetime
from collections import defaultdict
import random
import colorsys
import pickle

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score
import torch
from main import instantiate_from_config
from ldm.data.hpa2 import matched_idx_to_location, decode_one_hot_locations
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.parse import str2bool
from ldm.util import instantiate_from_config
from ldm.evaluation.metrics import ImageEvaluator
# import yaml
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt

"""
Command example: CUDA_VISIBLE_DEVICES=0 python scripts/img_gen/prot2img.py --config=configs/latent-diffusion/hpa2__ldm__vq4__densenet_all__splitcpp__cell256-debug.yaml --checkpoint=/scratch/users/xikunz2/stable-diffusion/logs/2023-10-15T10-18-18_hpa2__ldm__vq4__densenet_all__splitcpp__cell256/checkpoints/last.ckpt --scale=2 -d

"""

# data_config_yaml = """
# data:
#   target: main.DataModuleFromConfig
#   params:
#     batch_size: 8
#     num_workers: 16
#     wrap: false
#     train:
#       target: ldm.data.hpa.HPACombineDatasetMetadataInMemory
#       params:
#         seed: 123
#         group: train
#         train_split_ratio: 0.95
#         cache_file: /data/wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256-t5.pickle
#         channels:
#         - 1
#         - 1
#         - 1
#         filter_func: has_location
#         rotate_and_flip: true
#         include_location: true
#         use_uniprot_embedding: /data/wei/stable-diffusion/data/per-protein.h5
#     validation:
#       target: ldm.data.hpa.HPACombineDatasetMetadataInMemory
#       params:
#         seed: 123
#         group: validation
#         train_split_ratio: 0.95
#         cache_file: /data/wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256-t5.pickle
#         channels:
#         - 1
#         - 1
#         - 1
#         filter_func: has_location
#         include_location: true
#         use_uniprot_embedding: /data/wei/stable-diffusion/data/per-protein.h5
# """


# def make_batch(image, mask, device):
#     image = np.array(Image.open(image).convert("RGB"))
#     image = image.astype(np.float32)/255.0
#     image = image[None].transpose(0,3,1,2)
#     image = torch.from_numpy(image)

#     mask = np.array(Image.open(mask).convert("L"))
#     mask = mask.astype(np.float32)/255.0
#     mask = mask[None,None]
#     mask[mask < 0.5] = 0
#     mask[mask >= 0.5] = 1
#     mask = torch.from_numpy(mask)

#     masked_image = (1-mask)*image

#     batch = {"image": image, "mask": mask, "masked_image": masked_image}
#     for k in batch:
#         batch[k] = batch[k].to(device=device)
#         batch[k] = batch[k]*2.0-1.0
#     return batch


def main(opt):
    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    split = "train"
    if opt.name:
        name = opt.name
    else:
        name = f"{split}__gd{opt.scale}__fr_{opt.fix_reference}__steps{opt.steps}"
    # nowname = now + "_" + name
    opt.outdir = "/data/xikunz/stable-diffusion/img_gen" if opt.checkpoint is None else f"{os.path.dirname(os.path.dirname(opt.checkpoint))}/{name}"

    config = OmegaConf.load(opt.config)
    # config = yaml.safe_load(data_config_yaml)
    data_config = config['data']
    data_config['params'][split]["params"]["return_info"] = True

    # Load data
    data = instantiate_from_config(data_config)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # each image is:
    # 'image': array(...)
    # 'file_path_': 'data/celeba/data256x256/21508.jpg'
    
    # Load the model checkpoint
    model = instantiate_from_config(config.model)
    if opt.checkpoint is not None:
        model.load_state_dict(torch.load(opt.checkpoint, map_location="cpu")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    image_evaluator = ImageEvaluator(device=device)

    # Create a mapping from protein to all its possible locations
    if opt.fix_reference:
        with open("/data/wei/hpa-webdataset-all-composite/HPACombineDatasetInfo.pickle", 'rb') as fp:
            info_list = pickle.load(fp)
        protcl2locs = defaultdict(set)
        for x in info_list:
            prot, cl = x["gene_names"], x["atlas_name"]
            if str(prot) != "nan":
                protcl2locs[(prot, cl)].update(str(x["locations"]).split(","))

    ref = None
    os.makedirs(opt.outdir, exist_ok=True)
    total_count = 8 if opt.debug else len(data.datasets[split])
    # np.random.seed(123)
    np.random.seed(12)
    idcs = np.random.choice(len(data.datasets[split]), total_count, replace=False)
    ref_images, predicted_images, gt_images = [], [], []
    locations, filenames, conditions = [], [], []
    mse_list, ssim_list, feats_mse_list, samples_loc_probs_list, sc_gt_locations_list = [[] for _ in range(5)]
    with torch.no_grad():
        with model.ema_scope():
            for i in tqdm(idcs):
            # for i, sample in tqdm(enumerate(data.datasets[split]), total=total_count):
            #     if i % 3 != 0:
            #         continue
                sample = data.datasets[split][i]
                sample = {k: torch.from_numpy(np.expand_dims(sample[k], axis=0)).to(device) if isinstance(sample[k], (np.ndarray, np.generic)) else sample[k] for k in sample.keys()}
                name = sample['info']['filename'].split('/')[-1]
                if opt.fix_reference:
                    if ref is None:
                        ref = sample['ref-image']
                    else:
                        sample['ref-image'] = ref
                else:
                    ref = sample['ref-image']
                outpath = os.path.join(opt.outdir, name)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model(sample)
                # uc = {'c_concat': [torch.zeros_like(v) for v in c['c_concat']], 'c_crossattn': [torch.zeros_like(v) for v in c['c_crossattn']]} #
                uc = {'c_concat': c['c_concat'], 'c_crossattn': [torch.zeros_like(v) for v in c['c_crossattn']]} #

                shape = (c['c_concat'][0].shape[1],)+c['c_concat'][0].shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c['c_concat'][0].shape[0],
                                                 shape=shape,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                gt_locations, bbox_coords = sample["matched_location_classes"], sample["bbox_coords"]
                mse, ssim, feats_mse, samples_loc_probs, sc_gt_locations = image_evaluator.calc_metrics(samples=x_samples_ddim, targets=torch.permute(sample['image'], (0, 3, 1, 2)), refs=torch.permute(ref, (0, 3, 1, 2)), gt_locations=gt_locations, bbox_coords=bbox_coords)
                mse_list.append(mse[0])
                ssim_list.append(ssim[0])
                feats_mse_list.append(feats_mse[0])
                samples_loc_probs_list.append(samples_loc_probs[0])
                sc_gt_locations_list.append(sc_gt_locations[0])

                prot_image = torch.clamp((sample['image']+1.0)/2.0,
                                    min=0.0, max=1.0)
                ref_image = torch.clamp((ref+1.0)/2.0,
                                    min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                ref_image = ref_image.cpu().numpy()[0]*255
                ref_image = ref_image.astype(np.uint8)
                # Image.fromarray(ref_image.astype(np.uint8)).save(outpath+sample['info']['locations']+'_reference.png')
                
                predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                predicted_image = predicted_image.astype(np.uint8)
                # Image.fromarray(predicted_image.astype(np.uint8)).save(outpath+sample['info']['locations']+'_prediction.png')
                fig, axes = plt.subplots(1, 2 if opt.fix_reference else 3)
                ax = axes[0]
                ax.axis('off')
                ax.imshow(ref_image.astype(np.uint8))
                ax.set_title("Reference")
                ax = axes[1]
                ax.axis('off')
                ax.imshow(predicted_image.astype(np.uint8))
                ax.set_title("Predicted protein")
                if not opt.fix_reference:
                    prot_image = prot_image.cpu().numpy()[0]*255
                    prot_image = prot_image.astype(np.uint8)
                    # Image.fromarray(prot_image.astype(np.uint8)).save(outpath+'protein.png')
                    ax = axes[2]
                    ax.axis('off')
                    ax.imshow(prot_image.astype(np.uint8))
                    ax.set_title("GT protein")

                prot, cl = sample['condition_caption'].split("/")
                # locs_str = ",".join(protcl2locs[(prot, cl)]) if opt.fix_reference else sample['info']['locations']
                if opt.fix_reference:
                    locs_str = ""
                    for j, loc in enumerate(protcl2locs[(prot, cl)]):
                        if j > 0 and j % 2 == 0:
                            locs_str += "\n"
                        locs_str += f"{loc},"
                else:
                    locs_str = sample['info']['locations']
                fig.suptitle(f"{name}, {sample['condition_caption']}, {locs_str}")
                fig.savefig(outpath)

                ref_images.append(ref_image)
                predicted_images.append(predicted_image)
                gt_images.append(prot_image)
                # locations.append(locs_str)
                locations.append(gt_locations)
                filenames.append(name)
                conditions.append(sample['condition_caption'])

                # if opt.debug and count >= debug_count - 1:
                #     break
                # count += 1

    # plot the first 15 images in a grid
    # plt.figure(figsize=(20,12))
    # set color map to gray
    plt.set_cmap('gray')
    if opt.fix_reference:
        fig, axes = plt.subplots(3, 5, figsize=(20,12))
        axes = axes.flatten()
        n_images_to_plot = min(15, len(predicted_images) + 1)
        for i in range(n_images_to_plot):
            # plt.subplot(3,5,i+1)
            ax = axes[i]
            # Use mean instead of sum, which was the previous practice
            image = ref_image if i == 0 else predicted_images[i - 1].mean(axis=2)
            # print(f"max: {image.max()}, min:{image.min()}")
            # clip the image to 0-1
            image = np.clip(image, 0, 255) / 255.0
            ax.imshow(image)
            ax.axis('off')
            # plot text in each image with locations
            # plt.text(0, 20, locations[i], color='white', fontsize=10)
            ax.set_title("Reference" if i == 0 else locations[i - 1])
    else:
        fig, axes = plt.subplots(8, 3, figsize=(9,24))
        n_images_to_plot = min(8, len(predicted_images))
        for i in range(n_images_to_plot):
            for j in range(3):
                ax = axes[i, j]
                # Use mean instead of sum, which was the previous practice
                if j == 0:
                    image = ref_images[i]
                    # title = f"{filenames[i]}\n{conditions[i]}"
                    title = f"{conditions[i]}"
                elif j == 1:
                    image = predicted_images[i].mean(axis=2)
                    samples_locations = (samples_loc_probs_list[i] > 0.5).astype(int)
                    samples_locations = decode_one_hot_locations(samples_locations, matched_idx_to_location)
                    title = f"MSE:{mse_list[i]:.2g},SSIM:{ssim_list[i]:.2g},featMSE:{feats_mse_list[i]:.2g}\nsc:{samples_locations}"
                else:
                    image = gt_images[i].mean(axis=2)
                    sc_gt_locations = decode_one_hot_locations(sc_gt_locations_list[i], matched_idx_to_location)
                    gt_locations = decode_one_hot_locations(locations[i], matched_idx_to_location)
                    title = f"image:{gt_locations}\nsc:{sc_gt_locations}"
                # print(f"max: {image.max()}, min:{image.min()}")
                # clip the image to 0-1
                image = np.clip(image, 0, 255) / 255.0
                ax.imshow(image)
                ax.axis('off')
                # plot text in each image with locations
                # plt.text(0, 20, locations[i], color='white', fontsize=10)
                ax.set_title(title)
    mse_mean = np.mean(mse_list)
    ssim_mean = np.mean(ssim_list)
    features_mse_mean = np.mean(feats_mse_list)
    loc_mean_avg_precision = average_precision_score(sc_gt_locations_list, samples_loc_probs_list)
    fig.suptitle(f'{split}, guidance={opt.scale}, DDIM steps={opt.steps}, MSE: {mse_mean:.2g}, SSIM: {ssim_mean:.2g}, features MSE: {features_mse_mean:.2g}, location MAP: {loc_mean_avg_precision:.2g}', y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(opt.outdir, f'predicted-image-grid-s{opt.scale}.png'))

    if opt.fix_reference:
        # Save images in a pickle file
        pickle.dump({"prediction": predicted_images, "reference": ref_image, "locations": locations}, open(os.path.join(opt.outdir, 'predictions.pkl'), 'wb'))
        
        # for each predicted image, we assign a random color map from matplotlib
        # then we stack all the color images into one
        # and save it
        # get all the color maps
        cms = list(plt.cm.datad.keys()) # 75 color maps
        
        result_image = np.zeros((predicted_images[0].shape[0], predicted_images[0].shape[1], 3))
        # Get the color map by name:
        for i in range(len(predicted_images)):
            h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
            r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
            image = predicted_images[i].mean(axis=2)
            # clip the image to 0-1
            image = np.clip(image, 0, 255) / 255.0
            colored_image = np.stack([image*r, image*g, image*b], axis=2)
            result_image += colored_image
        result_image = result_image/result_image.max() * 255.0
        Image.fromarray(result_image.astype(np.uint8)).save(os.path.join(opt.outdir, 'super-multiplexed.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict protein images. Example command: python scripts/prot2img.py --config=configs/latent-diffusion/hpa-ldm-vq-4-hybrid-protein-location-augmentation.yaml --checkpoint=logs/2023-04-07T01-25-41_hpa-ldm-vq-4-hybrid-protein-location-augmentation/checkpoints/last.ckpt --scale=2 --outdir=./data/22-fixed --fix-reference")
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        help="the model config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="?",
        help="the model checkpoint",
    )
    parser.add_argument(
        "--fix-reference",
        action="store_true",
        help="fix the reference channel",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="unconditional guidance scale",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    opt = parser.parse_args()

    main(opt)
