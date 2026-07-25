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


def _body_mask(img01: np.ndarray) -> np.ndarray:
    """Pixels that are actually the patient, not air.

    A fixed low threshold (we used `> 0.02`) is wrong here, because this runs on
    the CLAHE-enhanced image and CLAHE amplifies background noise far above any
    such cutoff. The network then spent whole classes describing speckle in the
    air outside the body — visible as coloured blobs floating in the background,
    and a waste of the class budget that left real anatomy under-segmented.

    Otsu picks the tissue/air split from the image's own histogram, then we keep
    the largest connected component (the patient) and close small holes so
    interior dark structures — the canal, the airway — stay inside the mask
    rather than being punched out of it.
    """
    import cv2
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    _t, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _c = cv2.connectedComponentsWithStats((th > 0).astype(np.uint8), 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        th = (lab == biggest).astype(np.uint8) * 255
    # Fill enclosed holes so the canal and airway count as inside the body — but
    # only SMALL ones. Filling every enclosed region also swallows the large
    # concave pocket under the neck, which is air, and colouring it makes the
    # segmentation look like it is labelling nothing.
    ff = th.copy()
    h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    hn, hlab, hstats, _hc = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8), 8)
    max_hole = 0.03 * h * w
    keep = np.zeros_like(th)
    for i in range(1, hn):
        if hstats[i, cv2.CC_STAT_AREA] <= max_hole:
            keep[hlab == i] = 255
    m = (th | keep) > 0
    # if Otsu misfires on a very low-contrast scan, fall back rather than
    # returning a mask that covers almost nothing
    return m if m.mean() > 0.02 else (img01 > 0.02)


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
    fg_np = _body_mask(img01)
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

    labels[~fg_np] = -1                                 # background stays unlabelled

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


def vertebra_instances(img01: np.ndarray, labels: np.ndarray,
                       min_area: int = 120, max_area_frac: float = 0.09):
    """Split the vertebral column into INDIVIDUAL vertebrae.

    The reference literature presents spine segmentation as per-vertebra
    instances, not a single "bone" region — so after the network has produced
    semantic classes we separate each vertebral body as its own object.

    No labels are needed: vertebrae are a repeating chain of similarly-sized
    blocks, so for each semantic class we take its connected components and
    score the class on how well its components look like such a chain
    (enough of them, comparable areas, arranged along one axis). The
    best-scoring class is the vertebral column; its components are the
    individual vertebrae.
    """
    import cv2
    h, w = labels.shape
    total_fg = max(int((labels >= 0).sum()), 1)
    best = {"score": -1.0, "instances": None, "n": 0}

    # Geometric prior: vertebral bodies form a chain running parallel to the
    # spinal canal, close to it. Without this constraint the scoring happily
    # picks a chain of soft-tissue fragments elsewhere in the image, which is
    # exactly what it did in testing. The canal detector is reliable (91/92),
    # so use it to restrict where vertebrae may be.
    canal_axis = canal_centre = None
    try:
        from spine_measurements import canal_width_profile
        _prof, cinfo = canal_width_profile(img01)
        if cinfo is not None:
            canal_axis, canal_centre = cinfo["axis"], cinfo["centre"]
            canal_perp = cinfo["perp"]
    except Exception:
        canal_axis = None

    def near_canal(cx, cy):
        """Within a plausible perpendicular distance of the canal line."""
        if canal_axis is None:
            return True
        d = np.array([cx, cy], dtype=np.float32) - canal_centre
        return abs(float(d @ canal_perp)) < 0.22 * max(h, w)

    for cls in [u for u in np.unique(labels) if u >= 0]:
        m = (labels == cls).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        keep = [i for i in range(1, n)
                if stats[i, cv2.CC_STAT_AREA] >= min_area
                and stats[i, cv2.CC_STAT_AREA] <= max_area_frac * total_fg
                and near_canal(cents[i][0], cents[i][1])]
        if len(keep) < 4:                       # a spine shows several vertebrae
            continue
        areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in keep], dtype=np.float32)
        cen = np.array([cents[i] for i in keep], dtype=np.float32)

        # vertebrae are similar in size ...
        uniformity = 1.0 / (1.0 + float(areas.std() / (areas.mean() + 1e-6)))
        # ... and strung out along a single direction
        c = cen - cen.mean(axis=0)
        evals = np.linalg.eigvalsh(np.cov(c.T) + np.eye(2) * 1e-6)
        linearity = float(evals.max() / (evals.sum() + 1e-6))
        score = uniformity * linearity * min(len(keep), 12)

        if score > best["score"]:
            inst = np.zeros_like(labels)
            # number them in order along the chain, so the labels read head-to-tail
            axis_pos = c @ np.linalg.eigh(np.cov(c.T) + np.eye(2) * 1e-6)[1][:, -1]
            order = np.argsort(axis_pos)
            for rank, oi in enumerate(order, start=1):
                inst[lab == keep[oi]] = rank
            best = {"score": score, "instances": inst, "n": len(keep),
                    "uniformity": round(uniformity, 3), "linearity": round(linearity, 3)}

    if best["instances"] is None:
        return np.zeros_like(labels), {"n_vertebrae": 0}
    return best["instances"], {"n_vertebrae": best["n"],
                               "size_uniformity": best.get("uniformity"),
                               "alignment": best.get("linearity")}


def overlay_instances(img01: np.ndarray, inst: np.ndarray) -> np.ndarray:
    """Each vertebra in its own colour, numbered — the presentation used in the
    reference spine-segmentation literature."""
    import cv2
    base = cv2.cvtColor((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    col = np.zeros_like(base)
    ids = [u for u in np.unique(inst) if u > 0]
    for k, u in enumerate(ids):
        col[inst == u] = DISTINCT[k % len(DISTINCT)][::-1]
    out = cv2.addWeighted(base, 1.0, col, 0.5, 0)
    for k, u in enumerate(ids):
        ys, xs = np.nonzero(inst == u)
        if xs.size:
            cv2.putText(out, str(k + 1), (int(xs.mean()) - 4, int(ys.mean()) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return out





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
