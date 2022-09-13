import numpy as np
import torch
import torch.nn.functional as F

class PSNR(torch.nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, peak=255.):
        self.name = "PSNR"
        self.peak = peak

    def forward(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 10 * torch.log10(self.peak ** 2 / mse)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return
