import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
import torchmetrics
# from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision

from ldm.data import image_processing
from ldm.data.hpa2 import matched_idx_to_location
from ldm.models.sc_loc_classifier.cls_inception_v3 import load_model


class ImageEvaluator:
    def __init__(self, device):
        # self.fid = FrechetInceptionDistance(normalize=True)
        self.loc_clf = load_model(device)

    def clf_images(self, resized_cropped_images):
        outputs = self.loc_clf({'image': resized_cropped_images})
        logits = outputs['logits']
        probs = torch.sigmoid(logits)
        probs = probs.to('cpu').detach().numpy()
        feats = outputs['feature_vector']
        return probs, feats

    def calc_metrics(self, samples, targets, refs, gt_locations, bbox_coords, masks, bbox_labels):
        '''The value range of the inputs should be between -1 and 1.'''
        bs, resolution = targets.size(0), targets.size(2)
        assert targets.size() == (bs, 3, resolution, resolution)
        assert samples.size() == (bs, 3, resolution, resolution)
        assert image_processing.is_between_minus1_1(targets)

        targets = (targets + 1) / 2 # [0, 1]
        refs = (refs + 1) / 2 # [0, 1]
        samples = (samples + 1) / 2 # [0, 1]
        samples = torch.clip(samples, min=0, max=1)

        # Calculate MSE and SSIM
        mse = F.mse_loss(samples, targets, reduction='none').mean(dim=[1,2,3]).to('cpu').detach().numpy()
        ssim = torchmetrics.functional.image.ssim.ssim(samples, targets, reduction='none').mean(dim=[1,2,3]).to('cpu').detach().numpy()
        # self.fid.update(targets, real=True)
        # self.fid.update(samples, real=False)
        # fid = self.fid.compute().item()

        # Calculate MSE and SSIM inside the bounding box
        if bbox_coords and masks and bbox_labels:
            bbox_y, bbox_x, bbox_height, bbox_width = bbox_coords[:, 0], bbox_coords[:, 1], bbox_coords[:, 2], bbox_coords[:, 3]
            resized_cropped_samples, resized_cropped_targets, resized_cropped_refs = [], [], []
            padding = int(resolution / 512 * 20)
            masks = masks.unsqueeze(1).expand(-1, 3, -1, -1)
            for i in range(bs):
                min_row, min_col = max(bbox_y[i] - padding, 0), max(bbox_x[i] - padding, 0)
                max_row, max_col = min(bbox_y[i] + bbox_height[i] + padding, resolution), min(bbox_x[i] + bbox_width[i] + padding, resolution)

                samples[i][masks[i] != bbox_labels[i]] = 0
                cropped_image = samples[i, :, min_row:max_row, min_col:max_col]
                resized_cropped_image = torchvision.transforms.functional.resize(cropped_image, size=(128, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                resized_cropped_samples.append(resized_cropped_image)

                targets[i][masks[i] != bbox_labels[i]] = 0
                cropped_image = targets[i, :, min_row:max_row, min_col:max_col]
                resized_cropped_image = torchvision.transforms.functional.resize(cropped_image, size=(128, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                resized_cropped_targets.append(resized_cropped_image)

                refs[i][masks[i] != bbox_labels[i]] = 0
                cropped_image = refs[i, :, min_row:max_row, min_col:max_col]
                resized_cropped_image = torchvision.transforms.functional.resize(cropped_image, size=(128, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                resized_cropped_refs.append(resized_cropped_image)
            resized_cropped_samples = torch.stack(resized_cropped_samples, dim=0)
            resized_cropped_targets = torch.stack(resized_cropped_targets, dim=0)
            resized_cropped_refs = torch.stack(resized_cropped_refs, dim=0)
            mse_bbox = F.mse_loss(resized_cropped_samples, resized_cropped_targets, reduction='none').mean(dim=[1,2,3]).to('cpu').detach().numpy()
            ssim_bbox = torchmetrics.functional.image.ssim.ssim(resized_cropped_samples, resized_cropped_targets, reduction='none').mean(dim=[1,2,3]).to('cpu').detach().numpy()
        else:
            mse_bbox = mse
            ssim_bbox = ssim

        # Evaluate using a single-cell location classifier 
        if gt_locations:
            resized_cropped_samples = resized_cropped_samples.mean(dim=1, keepdim=True)
            samples_4_channels = torch.cat([resized_cropped_refs, resized_cropped_samples], dim=1)[:, [0, 3, 2, 1]]
            samples_loc_probs, samples_feats = self.clf_images(samples_4_channels)
            targets_4_channels = torch.cat([resized_cropped_refs, resized_cropped_targets], dim=1)[:, [0, 3, 2, 1]]
            targets_loc_probs, targets_feats = self.clf_images(targets_4_channels)

            feats_mse = F.mse_loss(samples_feats, targets_feats, reduction='none').mean(dim=1).to('cpu').detach().numpy()
            sc_gt_locations = (targets_loc_probs > 0.5) * gt_locations.to('cpu').detach().numpy()
            for idx, loc in matched_idx_to_location.items():
                print(f"{loc} {targets_loc_probs[0, idx]},", end=" ")
        else:
            feats_mse = []
            samples_loc_probs = []
            sc_gt_locations = []
        return mse, ssim, mse_bbox, ssim_bbox, feats_mse, samples_loc_probs, sc_gt_locations


def calc_localization_metrics(samples_loc_probs, sc_gt_locations, feats_mse):
    any_loc = sc_gt_locations[:, :-1].any(axis=1)
    sc_gt_locations = sc_gt_locations[any_loc]
    samples_locations = samples_loc_probs[any_loc] > 0.5
    samples_locations = samples_locations.astype(int)
    feats_mse_mean = feats_mse[any_loc].mean()
    if len(sc_gt_locations) == 0:
        loc_acc = loc_macrof1 = loc_microf1 = float("nan")
    else:
        loc_acc = (samples_locations == sc_gt_locations).all(axis=1).mean()
        loc_macrof1 = sklearn.metrics.f1_score(sc_gt_locations, samples_locations, average='macro')
        loc_microf1 = sklearn.metrics.f1_score(sc_gt_locations, samples_locations, average='micro')
    return loc_acc, loc_macrof1, loc_microf1, feats_mse_mean


if __name__ == "__main__":
    targets = torch.load("/data/xikunz/stable-diffusion/gt_images.pt")
    samples = torch.load("/data/xikunz/stable-diffusion/gen_images.pt")
    mse, ssim = calc_metrics(targets, samples)
    print(mse, ssim)