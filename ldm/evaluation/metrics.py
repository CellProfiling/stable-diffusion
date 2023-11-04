import torch
import torch.nn.functional as F
import torchmetrics
# from torchmetrics.image.fid import FrechetInceptionDistance


# class ImageEvaluator:
#     def __init__(self):
#         self.fid = FrechetInceptionDistance(normalize=True)


def clf_images(images, clf):
    outputs = clf({'image': images})
    logits = outputs['logits']
    probs = torch.sigmoid(logits)
    probs = probs.to('cpu').detach().numpy()
    feats = outputs['feature_vector']
    return probs, feats

def calc_metrics(samples, targets, refs, gt_locations, clf):
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
    samples_loc_probs, samples_feats = clf_images(samples_4_channels, clf)
    targets_4_channels = torch.cat([refs, targets], dim=1)[:, [0, 3, 2, 1]]
    targets_loc_probs, targets_feats = clf_images(targets_4_channels, clf)

    feats_mse = F.mse_loss(samples_feats, targets_feats, reduction='none').mean(dim=1).to('cpu').detach().numpy()
    sc_gt_locations = (targets_loc_probs > 0.5) * gt_locations.to('cpu').detach().numpy()
    return mse, ssim, feats_mse, samples_loc_probs, sc_gt_locations





if __name__ == "__main__":
    targets = torch.load("/data/xikunz/stable-diffusion/gt_images.pt")
    samples = torch.load("/data/xikunz/stable-diffusion/gen_images.pt")
    mse, ssim = calc_metrics(targets, samples)
    print(mse, ssim)