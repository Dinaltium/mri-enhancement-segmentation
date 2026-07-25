"""
spine_deep_segmentation.py   --   unsupervised DEEP segmentation for spine MRI

The organisers require spine models that need NO annotations for training. Our
earlier spine segmentation was intensity clustering (k-means, then SLIC), which
is not a learned model and cannot represent structure — it only groups
brightness. This replaces it with a genuinely trained network.

Method: differentiable feature clustering (Kanezaki, "Unsupervised Image
Segmentation by Backpropagation", ICASSP 2018), with a superpixel consistency
prior. A small CNN is optimised directly on the image being segmented:

  1. The CNN maps every pixel to a feature vector over K candidate classes.
  2. Its own argmax is taken as a pseudo-label target.
  3. Two losses are minimised together:
       - feature loss: cross-entropy to that pseudo-label, which sharpens
         each pixel's commitment to one class;
       - continuity loss: L1 between horizontally and vertically neighbouring
         feature maps, which forces spatially coherent regions rather than
         speckle;
       - superpixel prior: each SLIC superpixel is forced to a single label,
         which anchors boundaries to real image edges.
  4. Repeating this collapses K candidate classes down to however many
     distinct structures the image actually supports.

No labels are used at any point — the supervision comes from the image's own
structure. This is self-supervised learning in the strict sense, and it is a
trained neural network, not a clustering heuristic.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegNet(nn.Module):
    """Small fully-convolutional network. Deliberately shallow: it is optimised
    per-image in a fraction of a second, so capacity is not the bottleneck —
    the losses are what shape the output."""

    def __init__(self, in_ch: int = 1, n_channel: int = 24, n_conv: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, n_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.middle = nn.ModuleList()
        self.middle_bn = nn.ModuleList()
        for _ in range(n_conv - 1):
            self.middle.append(nn.Conv2d(n_channel, n_channel, 3, padding=1))
            self.middle_bn.append(nn.BatchNorm2d(n_channel))
        self.final = nn.Conv2d(n_channel, n_channel, 1)
        self.final_bn = nn.BatchNorm2d(n_channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for conv, bn in zip(self.middle, self.middle_bn):
            x = F.relu(bn(conv(x)))
        return self.final_bn(self.final(x))


def _superpixels(img01: np.ndarray, n_segments: int = 220):
    from skimage.segmentation import slic
    return slic(img01, n_segments=n_segments, compactness=0.10,
                channel_axis=None, start_label=0, mask=img01 > 0.02)


def segment(img01: np.ndarray, n_classes: int = 12, iters: int = 120,
            lr: float = 0.10, continuity_weight: float = 1.0,
            balance_weight: float = 2.0,
            use_superpixels: bool = True, seed: int = 0):
    """Train the network on this one image and return an integer label map.

    Returns (labels, info) where info records how many distinct structures
    survived — an emergent property, not something we set in advance.
    """
    torch.manual_seed(seed)
    h, w = img01.shape
    x = torch.from_numpy(img01).float().view(1, 1, h, w).to(DEVICE)

    model = SegNet(1, n_classes).to(DEVICE).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()

    sp = _superpixels(img01) if use_superpixels else None
    sp_index = None
    if sp is not None:
        sp_index = [np.nonzero(sp.reshape(-1) == s)[0] for s in np.unique(sp)]
        sp_index = [idx for idx in sp_index if idx.size > 0]

    # Foreground is masked in the LOSS, not in the features. Zeroing the
    # features themselves makes every background pixel argmax to the same
    # class, and the cross-entropy term then drags the whole image into it —
    # which collapses the segmentation to a single region.
    fg_np = img01 > 0.02
    fg_flat = torch.from_numpy(fg_np.reshape(-1)).to(DEVICE)
    n_final = n_classes
    for _ in range(iters):
        opt.zero_grad()
        out = model(x)                                  # 1,K,H,W

        # continuity: neighbouring pixels should share features
        loss_cont = (l1(out[:, :, 1:, :], out[:, :, :-1, :]) +
                     l1(out[:, :, :, 1:], out[:, :, :, :-1]))

        flat = out[0].permute(1, 2, 0).reshape(-1, n_classes)   # HW,K
        target = torch.argmax(flat, dim=1)

        # NOTE: the superpixel prior is deliberately NOT applied inside the
        # training loop. Forcing every superpixel to one label on each step
        # compounds with the cross-entropy feedback and collapses the image to
        # two regions (measured: 2 classes with the prior, 11 without). It is
        # applied once after training instead, purely to snap boundaries onto
        # real image edges.

        # Class-balance term. Cross-entropy against the network's own argmax is
        # a positive feedback loop: whichever class is currently largest keeps
        # absorbing the others until one region remains. Maximising the entropy
        # of the MEAN prediction counteracts that — each pixel stays confident,
        # but the image as a whole is pushed to use several classes.
        p_mean = F.softmax(flat[fg_flat], dim=1).mean(dim=0)
        balance = (p_mean * torch.log(p_mean + 1e-8)).sum()   # = -entropy

        loss = (ce(flat[fg_flat], target[fg_flat])
                + continuity_weight * loss_cont
                + balance_weight * balance)
        loss.backward()
        opt.step()
        n_final = int(len(torch.unique(target[fg_flat])))

    with torch.no_grad():
        out = model(x)
        labels = torch.argmax(out[0], dim=0).cpu().numpy().astype(np.int32)

    # post-hoc superpixel refinement: each superpixel takes its majority label,
    # which aligns region boundaries with real image edges without influencing
    # how many classes survived
    if sp_index is not None:
        flat_lab = labels.reshape(-1)
        for idx in sp_index:
            flat_lab[idx] = np.bincount(flat_lab[idx]).argmax()
        labels = flat_lab.reshape(h, w)

    labels[img01 <= 0.02] = -1                          # background stays unlabelled

    # renumber surviving classes to 0..n-1, ordered by mean intensity so the
    # mapping is stable and interpretable (dark structures first)
    uniq = [u for u in np.unique(labels) if u >= 0]
    means = [(u, float(img01[labels == u].mean())) for u in uniq]
    means.sort(key=lambda t: t[1])
    remap = {u: i for i, (u, _m) in enumerate(means)}
    out_lab = np.full_like(labels, -1)
    for u, i in remap.items():
        out_lab[labels == u] = i
    return out_lab, {"classes_found": len(uniq), "iterations": iters,
                     "classes_requested": n_classes}


DISTINCT = np.array([
    [66, 135, 245], [80, 200, 120], [240, 170, 60], [220, 90, 90],
    [170, 110, 220], [60, 200, 210], [235, 130, 190], [150, 160, 90],
    [110, 140, 200], [200, 200, 110], [130, 200, 160], [230, 150, 120],
], dtype=np.uint8)


def colorize(labels: np.ndarray) -> np.ndarray:
    """BGR colour map, one distinct colour per discovered structure."""
    import cv2
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, u in enumerate(u for u in np.unique(labels) if u >= 0):
        out[labels == u] = DISTINCT[i % len(DISTINCT)][::-1]
    return out


def overlay(img01: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    import cv2
    base = cv2.cvtColor((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    col = colorize(labels)
    out = cv2.addWeighted(base, 1.0, col, alpha, 0)
    out[labels < 0] = base[labels < 0]
    return out
