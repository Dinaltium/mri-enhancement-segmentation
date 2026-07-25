"""
ssim.py

Differentiable SSIM (Structural Similarity) in pure torch, for use as a
training loss. skimage.metrics.structural_similarity is numpy-based and
can't be backpropagated through - this is needed specifically for
`loss = 1 - ssim(...)` inside train.py. skimage's SSIM is still used
separately for the reported validation metric (that part doesn't need
gradients, so the well-tested library version is used there instead -
see train.py).
"""

import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.outer(g)
    return kernel_2d.unsqueeze(0).unsqueeze(0)  # 1x1xHxW


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
          data_range: float = 1.0) -> torch.Tensor:
    """
    img1, img2: BxCxHxW, values in [0, data_range]. Returns a scalar mean SSIM.
    """
    channels = img1.shape[1]
    kernel = _gaussian_kernel(window_size).to(img1.device).repeat(channels, 1, 1, 1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(img1, img2, self.window_size, self.data_range)
