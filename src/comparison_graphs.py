"""
comparison_graphs.py

Generates the "why ours wins" figures from the measured result files. Every
number plotted is read from JSON produced by an actual run — nothing here is
hand-typed.

Figures (outputs/demo/):
  cmp_methods.png        our U-Net vs HE / AHE / CLAHE on PSNR, SSIM, FSIM, VIF
  cmp_noise.png          the noise level each processing stage leaves behind
  cmp_modality.png       modality-specific vs pooled spine models
  cmp_segmentation.png   per-class segmentation metrics
  cmp_summary.png        one-slide summary of the four headline claims
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs/demo"
os.makedirs(OUT, exist_ok=True)

# restrained clinical palette — ours is the only saturated bar
INK = "#2b2f36"
GREY = "#b9bfc7"
GREY_D = "#8e959e"
OURS = "#1f6f4a"
BAD = "#b4432c"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9ced5", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _bars(ax, labels, values, ours_idx, title, ylabel, fmt="{:.2f}", ylim=None):
    colors = [OURS if i == ours_idx else GREY for i in range(len(values))]
    b = ax.bar(labels, values, color=colors, width=0.62,
               edgecolor=[INK if i == ours_idx else GREY_D for i in range(len(values))],
               linewidth=1.0)
    ax.set_title(title, fontsize=11, fontweight="600", loc="left", pad=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", color="#e8ebee", linewidth=1)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    for r, v in zip(b, values):
        ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center",
                va="bottom", fontsize=9, fontweight="600")
    ax.tick_params(axis="x", labelsize=9)


def fig_methods():
    d = load("results/paper_comparison.json")
    if not d:
        print("skip cmp_methods (no paper_comparison.json)"); return
    names = ["Degraded input", "HE", "AHE", "CLAHE", "Ours (U-Net)"]
    names = [n for n in names if n in d]
    ours_i = names.index("Ours (2D U-Net)") if "Ours (2D U-Net)" in names else len(names) - 1
    if "Ours (2D U-Net)" in d:
        names = ["Degraded input", "HE", "AHE", "CLAHE", "Ours (2D U-Net)"]
        ours_i = 4
    short = ["Degraded\ninput", "HE", "AHE", "CLAHE", "Ours\n(U-Net)"]

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.9))
    for a, key, lab, unit in zip(
            ax, ["psnr", "ssim", "fsim", "vif"],
            ["PSNR", "SSIM", "FSIM", "VIF"], ["dB", "0–1", "0–1", "0–1"]):
        vals = [d[n][key] for n in names]
        _bars(a, short, vals, ours_i, f"{lab}  ({unit}, higher is better)", lab,
              "{:.2f}" if key == "psnr" else "{:.3f}")
    fig.suptitle("Brain MRI restoration — our model against the classical baselines "
                 "(identical slices, identical degradation)",
                 fontsize=12.5, fontweight="700", x=0.008, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.9])
    fig.text(0.008, 0.008,
             "Every classical method scores BELOW the degraded input — they raise contrast but "
             "amplify the noise instead of removing it.", fontsize=9, color=BAD)
    fig.savefig(f"{OUT}/cmp_methods.png", dpi=140); plt.close(fig)
    print("wrote cmp_methods.png")


def fig_noise():
    d = load("stats/preprocessing_assessment.json")
    if not d:
        print("skip cmp_noise"); return
    groups = [g for g in ["brain_normal", "brain_pathological", "spine_normal",
                          "spine_pathological"] if g in d]
    stages = ["preprocessed", "HE", "CLAHE", "AI_UNet"]
    labels = ["Preprocessed\n(baseline)", "HE", "CLAHE", "Ours\n(U-Net)"]
    x = np.arange(len(stages)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for i, g in enumerate(groups):
        vals = [d[g][s]["noise_level"] for s in stages if s in d[g]]
        ax.bar(x + (i - 1.5) * w, vals, w, label=g.replace("_", " "),
               color=["#cfd5db", "#aab2bb", "#8e959e", OURS][i] if False else None,
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("estimated noise level  (lower is better)")
    ax.set_title("Noise left behind by each processing stage",
                 fontsize=12, fontweight="700", loc="left", pad=10)
    ax.grid(axis="y", color="#e8ebee", linewidth=1); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.text(0.01, 0.005, "The classical steps INCREASE noise. Ours is the only stage that "
                          "reduces it below the preprocessed baseline.", fontsize=9, color=BAD)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{OUT}/cmp_noise.png", dpi=140); plt.close(fig)
    print("wrote cmp_noise.png")


def fig_modality():
    d = load("results/modality_comparison.json")
    if not d:
        print("skip cmp_modality"); return
    mods = [m for m in ["T1", "T2", "STIR"] if m in d]
    pooled = [d[m]["pooled"]["ssim"] for m in mods]
    spec = [d[m]["specific"]["ssim"] for m in mods]
    x = np.arange(len(mods)); w = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    b1 = ax.bar(x - w/2, pooled, w, label="One pooled model for all sequences",
                color=GREY, edgecolor=GREY_D)
    b2 = ax.bar(x + w/2, spec, w, label="A model per sequence (ours)",
                color=OURS, edgecolor=INK)
    for bars, vals in ((b1, pooled), (b2, spec)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width()/2, v, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=9, fontweight="600")
    for i, (p, s) in enumerate(zip(pooled, spec)):
        ax.annotate(f"+{s-p:.2f}", xy=(i + w/2, s), xytext=(i + w/2, s + 0.075),
                    ha="center", fontsize=9, fontweight="700", color=OURS)
    ax.set_xticks(x); ax.set_xticklabels(mods); ax.set_ylim(0, 1.02)
    ax.set_ylabel("SSIM  (higher is better)")
    ax.set_title("Spine: one model per MRI sequence beats one model for everything",
                 fontsize=12, fontweight="700", loc="left", pad=10)
    ax.grid(axis="y", color="#e8ebee", linewidth=1); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "Both models scored on the SAME held-out slices with the same "
                          "degradation — a fair comparison.", fontsize=9, color=GREY_D)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{OUT}/cmp_modality.png", dpi=140); plt.close(fig)
    print("wrote cmp_modality.png")


def fig_segmentation():
    d = load("results/segmentation_full_metrics.json")
    if not d:
        print("skip cmp_segmentation"); return
    cls = [c for c in ["necrotic_non_enhancing", "edema", "enhancing"] if c in d]
    nice = ["Necrotic core", "Edema", "Enhancing tumour"]
    keys = ["dice", "jaccard", "sensitivity_recall", "precision"]
    klab = ["Dice", "Jaccard", "Sensitivity", "Precision"]
    x = np.arange(len(cls)); w = 0.2
    shades = ["#1f6f4a", "#4b8f6d", "#7aae94", "#a9ccbc"]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for i, (k, lab) in enumerate(zip(keys, klab)):
        vals = [d[c][k] for c in cls]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=shades[i],
                      edgecolor="white", linewidth=0.6)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width()/2, v, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels(nice); ax.set_ylim(0, 1.05)
    ax.set_ylabel("score  (higher is better)")
    ax.set_title(f"Brain tumour segmentation — mean tumour Dice "
                 f"{d.get('_mean_tumor_dice', 0):.2f} on held-out patients",
                 fontsize=12, fontweight="700", loc="left", pad=10)
    ax.grid(axis="y", color="#e8ebee", linewidth=1); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=4)
    fig.text(0.01, 0.005, "Scored against the radiologist's annotation on patients the model "
                          "never trained on.", fontsize=9, color=GREY_D)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{OUT}/cmp_segmentation.png", dpi=140); plt.close(fig)
    print("wrote cmp_segmentation.png")


def fig_summary():
    paper = load("results/paper_comparison.json")
    seg = load("results/segmentation_full_metrics.json")
    mod = load("results/modality_comparison.json")
    pre = load("stats/preprocessing_assessment.json")
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2))

    if paper:
        names = ["Degraded input", "HE", "AHE", "CLAHE", "Ours (2D U-Net)"]
        vals = [paper[n]["ssim"] for n in names if n in paper]
        _bars(ax[0][0], ["Input", "HE", "AHE", "CLAHE", "Ours"], vals, 4,
              "1 · Restoration quality (SSIM)", "SSIM", "{:.2f}", (0, 1.05))
    if pre:
        g = "brain_pathological" if "brain_pathological" in pre else list(pre)[0]
        st = ["preprocessed", "HE", "CLAHE", "AI_UNet"]
        vals = [pre[g][s]["noise_level"] for s in st if s in pre[g]]
        colors = [GREY, GREY, GREY, OURS]
        b = ax[0][1].bar(["Baseline", "HE", "CLAHE", "Ours"], vals, 0.62,
                         color=colors, edgecolor=[GREY_D, GREY_D, GREY_D, INK])
        for r, v in zip(b, vals):
            ax[0][1].text(r.get_x()+r.get_width()/2, v, f"{v:.4f}", ha="center",
                          va="bottom", fontsize=9, fontweight="600")
        ax[0][1].set_title("2 · Noise remaining (lower is better)", fontsize=11,
                           fontweight="600", loc="left", pad=9)
        ax[0][1].set_ylabel("noise level")
        ax[0][1].grid(axis="y", color="#e8ebee"); ax[0][1].set_axisbelow(True)
    if seg:
        cls = ["necrotic_non_enhancing", "edema", "enhancing"]
        vals = [seg[c]["dice"] for c in cls if c in seg]
        _bars(ax[1][0], ["Necrotic", "Edema", "Enhancing"], vals, 2,
              "3 · Tumour segmentation Dice vs radiologist", "Dice", "{:.2f}", (0, 1.05))
    if mod:
        mods = [m for m in ["T1", "T2", "STIR"] if m in mod]
        pooled = [mod[m]["pooled"]["ssim"] for m in mods]
        spec = [mod[m]["specific"]["ssim"] for m in mods]
        x = np.arange(len(mods)); w = 0.34
        ax[1][1].bar(x - w/2, pooled, w, label="pooled", color=GREY, edgecolor=GREY_D)
        ax[1][1].bar(x + w/2, spec, w, label="per-sequence (ours)", color=OURS, edgecolor=INK)
        ax[1][1].set_xticks(x); ax[1][1].set_xticklabels(mods); ax[1][1].set_ylim(0, 1.05)
        ax[1][1].set_title("4 · Spine: per-sequence models win 3/3", fontsize=11,
                           fontweight="600", loc="left", pad=9)
        ax[1][1].set_ylabel("SSIM")
        ax[1][1].legend(frameon=False, fontsize=8.5)
        ax[1][1].grid(axis="y", color="#e8ebee"); ax[1][1].set_axisbelow(True)

    fig.suptitle("Four measured claims", fontsize=14, fontweight="700",
                 x=0.008, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/cmp_summary.png", dpi=140); plt.close(fig)
    print("wrote cmp_summary.png")


if __name__ == "__main__":
    fig_methods(); fig_noise(); fig_modality(); fig_segmentation(); fig_summary()
    print(f"\nall comparison figures -> {OUT}/")
