"""
spine_autoencoder.py   --   Self-supervised anomaly detection for Spine MRI

Problem: the spine dataset has NO lesion labels and no external data is allowed,
so we cannot TRAIN a supervised "find the herniation" detector. But we CAN find
abnormalities WITHOUT labels, self-supervised:

  1. Train a convolutional AUTOENCODER on NORMAL (healthy) spine scans only.
     It learns to reconstruct what a healthy spine looks like.
  2. Show it a PATHOLOGICAL spine. It reconstructs the healthy-looking version,
     but CANNOT reproduce the lesion it never saw during training.
  3. The reconstruction ERROR map = |input - reconstruction| therefore LIGHTS UP
     exactly where the abnormality is (disc herniation, stenosis, lesion).

Key design point: this is a BOTTLENECK autoencoder with NO skip connections —
skip connections would let it copy the input (lesion included) straight through,
killing the anomaly signal. The tight bottleneck forces it to encode only the
"normal spine" manifold.

Usage:
    python spine_autoencoder.py --train --modality T2 --epochs 40
    python spine_autoencoder.py --detect   # anomaly maps on pathological spine
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nifti_utils import IMG_SIZE, load_volume
from enhancement_dataset import extract_training_slices
from offline_dataset import OFFLINE_ROOTS, find_offline_cases, classify_case_files
from ssim import SSIMLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "models/spine_autoencoder.pt"


class ConvAutoencoder(nn.Module):
    """Bottleneck autoencoder, NO skip connections (so anomalies can't leak
    through). 1-channel in/out, sigmoid output."""

    def __init__(self, f: int = 32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True))
        self.enc = nn.Sequential(
            block(1, f), nn.MaxPool2d(2),            # 112
            block(f, f * 2), nn.MaxPool2d(2),        # 56
            block(f * 2, f * 4), nn.MaxPool2d(2),    # 28
            block(f * 4, f * 4), nn.MaxPool2d(2),    # 14  (tight bottleneck)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(f * 4, f * 4, 2, stride=2), block(f * 4, f * 4),   # 28
            nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2), block(f * 2, f * 2),   # 56
            nn.ConvTranspose2d(f * 2, f, 2, stride=2), block(f, f),               # 112
            nn.ConvTranspose2d(f, f, 2, stride=2), block(f, f),                   # 224
            nn.Conv2d(f, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.dec(self.enc(x)))


def collect_spine_slices(group: str, modality: str, cases=None) -> np.ndarray:
    root = OFFLINE_ROOTS[group]
    case_dirs = cases if cases is not None else find_offline_cases(root)
    slices = []
    for cd in case_dirs:
        info = classify_case_files(cd)
        for p in info["buckets"].get(modality, []):
            try:
                slices.extend(extract_training_slices(load_volume(p)))
            except Exception:
                continue
    return np.array(slices, dtype=np.float32) if slices else np.zeros((0, IMG_SIZE, IMG_SIZE), np.float32)


def train(modality: str, epochs: int):
    X = collect_spine_slices("spine_normal", modality)   # NORMAL only
    if len(X) < 10:
        raise ValueError(f"only {len(X)} normal spine {modality} slices found")
    print(f"[AE] training on {len(X)} NORMAL spine {modality} slices")
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1))
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    model = ConvAutoencoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), 5e-4)          # lower LR = stable, sharper
    l1, ssim = nn.L1Loss(), SSIMLoss()
    for ep in range(epochs):                                   # no AMP (stability > speed here)
        model.train(); tot = 0
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = l1(out, xb) + 0.5 * ssim(out, xb)
            loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[AE] epoch {ep+1}/{epochs} loss={tot/len(dl):.4f}", flush=True)
    torch.save({"model_state_dict": model.state_dict(), "modality": modality}, CKPT)
    print(f"[AE] saved -> {CKPT}")


def load_model():
    c = torch.load(CKPT, map_location=DEVICE)
    m = ConvAutoencoder().to(DEVICE).eval()
    m.load_state_dict(c["model_state_dict"])
    return m, c.get("modality", "T2")


def anomaly_map(model, img01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (reconstruction, anomaly_heatmap 0-1). High = abnormal."""
    with torch.no_grad():
        x = torch.from_numpy(img01).float().view(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
        recon = model(x)[0, 0].cpu().numpy()
    err = np.abs(img01 - recon)
    err = cv2.GaussianBlur(err, (9, 9), 0)
    fg = img01 > 0.05
    err = err * fg
    # focus on EXCESS error: subtract the typical (median) foreground error so
    # only unusually-high regions survive, then sharpen contrast
    if fg.sum() > 0:
        base_err = np.median(err[fg])
        err = np.clip(err - base_err, 0, None)
        err = err ** 1.5
    if err.max() > 0:
        err = err / err.max()
    return np.clip(recon, 0, 1), err


def overlay_anomaly(img01: np.ndarray, heat: np.ndarray, pct: float = 90.0) -> np.ndarray:
    """Heatmap over the scan + a box around the single STRONGEST compact
    anomaly (largest connected component above a high percentile)."""
    base = cv2.cvtColor((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    hm = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(base, 0.6, hm, 0.5, 0)
    fg = heat[img01 > 0.05]
    if fg.size == 0:
        return out
    thr = np.percentile(fg, pct)
    mask = ((heat > thr) & (img01 > 0.05)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[biggest]
        if area > 25:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(out, "suspected", (max(0, x - 4), max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def detect(modality: str, out_dir="outputs/spine_anomaly"):
    os.makedirs(out_dir, exist_ok=True)
    model, mod = load_model()
    modality = modality or mod
    cases = find_offline_cases(OFFLINE_ROOTS["spine_pathological"])
    n = 0
    for cd in cases:
        info = classify_case_files(cd)
        for p in info["buckets"].get(modality, [])[:1]:
            sls = extract_training_slices(load_volume(p))
            if not sls:
                continue
            img = sls[len(sls) // 2]
            recon, heat = anomaly_map(model, img)
            ov = overlay_anomaly(img, heat)
            base = os.path.basename(cd)
            cv2.imwrite(os.path.join(out_dir, f"{base}_anomaly.png"),
                        np.concatenate([
                            cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor((recon * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), ov], axis=1))
            score = float(heat.mean())
            print(f"[AE] {base}: anomaly score={score:.3f}")
            n += 1
            break
    print(f"[AE] wrote {n} anomaly panels -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--modality", default="T2")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    if args.train:
        train(args.modality, args.epochs)
    if args.detect:
        detect(args.modality)
    if not (args.train or args.detect):
        train(args.modality, args.epochs); detect(args.modality)


if __name__ == "__main__":
    main()
