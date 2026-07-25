"""
plots.py

Generates the learning-curve / metric figures the problem statement asks for
(training loss, validation loss, convergence) from the *_metrics.json files.
Matplotlib Agg backend (no display needed). Saves PNGs to outputs/plots/.

Usage:
    python plots.py                      # all available metric JSONs
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/plots"
os.makedirs(OUT, exist_ok=True)


def plot_segmentation(path="results/segmentation_metrics.json"):
    if not os.path.exists(path):
        print(f"[plots] {path} not found, skip"); return
    d = json.load(open(path))
    h = d["history"]
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(epochs, h["train_loss"], label="train loss", marker="o", ms=3)
    ax[0].plot(epochs, h["val_loss"], label="val loss", marker="s", ms=3)
    ax[0].axvline(d["best_epoch_convergence"], color="gray", ls="--",
                  label=f"convergence (ep {d['best_epoch_convergence']})")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("CE + Dice loss")
    ax[0].set_title("Brain Segmentation — Loss Curves"); ax[0].legend(); ax[0].grid(alpha=0.3)

    # per-epoch mean tumour dice
    ep = sorted(int(k) for k in h["val_metrics"].keys())
    dice = [h["val_metrics"][str(e)]["mean_tumor_dice"] for e in ep]
    ax[1].plot(ep, dice, color="green", marker="o", ms=3)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mean tumour Dice")
    ax[1].set_title("Brain Segmentation — Validation Dice"); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(0, 1)

    fig.tight_layout()
    p = os.path.join(OUT, "segmentation_curves.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[plots] wrote {p}")


def plot_enhancement(path):
    d = json.load(open(path))
    tag = d.get("group", os.path.basename(path))
    h = d.get("history", {})
    if not h:
        return
    tr = h.get("train_loss", [])
    te = h.get("test_loss", h.get("val_loss", []))
    epochs = range(1, len(tr) + 1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(epochs, tr, label="train loss", marker="o", ms=3)
    if te:
        ax.plot(epochs, te, label="test/val loss", marker="s", ms=3)
    be = d.get("best_epoch")
    if be:
        ax.axvline(be, color="gray", ls="--", label=f"best (ep {be})")
    ax.set_xlabel("epoch"); ax.set_ylabel("L1 + SSIM loss")
    ax.set_title(f"Enhancement — {tag}"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, f"enhancement_{tag}_curves.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[plots] wrote {p}")


def main():
    plot_segmentation()
    for path in sorted(glob.glob("results/enhancement_metrics*.json")):
        try:
            plot_enhancement(path)
        except Exception as e:
            print(f"[plots] failed on {path}: {e}")


if __name__ == "__main__":
    main()
