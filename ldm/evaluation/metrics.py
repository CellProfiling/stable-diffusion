import torch
import torch.nn.functional as F
import torchmetrics
# from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision

from ldm.data.hpa2 import matched_idx_to_location
from ldm.models.sc_loc_classifier.cls_inception_v3 import load_model


class ImageEvaluator:
    def __init__(self, device):
        # self.fid = FrechetInceptionDistance(normalize=True)
        self.loc_clf = load_model(device)

    def clf_images(self, images, bbox_coords):
        resolution = images.size(2)
        bbox_y, bbox_x, bbox_height, bbox_width = bbox_coords[:, 0], bbox_coords[:, 1], bbox_coords[:, 2], bbox_coords[:, 3]
        resized_cropped_images = []
        padding = int(resolution / 512 * 20)
        for i in range(images.size(0)):
            min_row, min_col = max(bbox_y[i] - padding, 0), max(bbox_x[i] - padding, 0)
            max_row, max_col = min(bbox_y[i] + bbox_height[i] + padding, resolution), min(bbox_x[i] + bbox_width[i] + padding, resolution)
            cropped_image = images[i, :, min_row:max_row, min_col:max_col]
            resized_cropped_image = torchvision.transforms.functional.resize(cropped_image, size=(128, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            resized_cropped_images.append(resized_cropped_image)
        resized_cropped_images = torch.stack(resized_cropped_images, dim=0)
        outputs = self.loc_clf({'image': resized_cropped_images})
        logits = outputs['logits']
        probs = torch.sigmoid(logits)
        probs = probs.to('cpu').detach().numpy()
        feats = outputs['feature_vector']
        return probs, feats

    def calc_metrics(self, samples, targets, refs, gt_locations, bbox_coords):
        '''The value range of the inputs should be between -1 and 1.'''
        bs, resolution = targets.size(0), targets.size(2)
        assert targets.size() == (bs, 3, resolution, resolution)
        assert samples.size() == (bs, 3, resolution, resolution)
        assert targets.min() >= -1
        assert targets.min() < 0
        assert targets.max() <= 1

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

        # Evaluate using a single-cell location classifier 
        samples = samples.mean(dim=1, keepdim=True)
        samples_4_channels = torch.cat([refs, samples], dim=1)[:, [0, 3, 2, 1]]
        samples_loc_probs, samples_feats = self.clf_images(samples_4_channels, bbox_coords)
        targets_4_channels = torch.cat([refs, targets], dim=1)[:, [0, 3, 2, 1]]
        targets_loc_probs, targets_feats = self.clf_images(targets_4_channels, bbox_coords)

        feats_mse = F.mse_loss(samples_feats, targets_feats, reduction='none').mean(dim=1).to('cpu').detach().numpy()
        sc_gt_locations = (targets_loc_probs > 0.5) * gt_locations.to('cpu').detach().numpy()
        for idx, loc in matched_idx_to_location.items():
            print(f"{loc} {targets_loc_probs[0, idx]},", end=" ")
        return mse, ssim, feats_mse, samples_loc_probs, sc_gt_locations





if __name__ == "__main__":
    targets = torch.load("/data/xikunz/stable-diffusion/gt_images.pt")
    samples = torch.load("/data/xikunz/stable-diffusion/gen_images.pt")
    mse, ssim = calc_metrics(targets, samples)
    print(mse, ssim)