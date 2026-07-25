# MRI Enhancement & ROI Segmentation

**MedhaDrishti National AI Hackathon · Yugma TechFest 2.0 · JNNCE Shivamogga**

A complete pipeline that takes a noisy MRI scan, restores it with a deep-learning
model, and delineates the region of interest — for **Brain** (tumour sub-regions,
CSF/grey/white tissue) and **Lumbo-sacral Spine** (disc/vertebra/cord regions plus
self-supervised abnormality localisation).

---

## Headline results

| What | Result |
|---|---|
| Brain tumour segmentation | **mean Dice 0.76**, enhancing tumour **0.84** (vs radiologist, held-out patients) |
| Brain MRI restoration | **PSNR 30.3 dB · SSIM 0.965** (BraTS FLAIR) |
| Restoration vs classical | Ours **SSIM 0.90** vs CLAHE 0.16, HE 0.15, AHE 0.13 — *every classical method scores below the noisy input* |
| Noise actually removed | Ours **0.0068 → 0.0043**; HE and CLAHE *raise* it to 0.0138 / 0.0106 |
| Spine per-sequence models | Beat one pooled model on **3/3** sequences (T1 +0.23, T2 +0.21, STIR +0.17 SSIM) |
| Spine anomaly detector | **Failed validation (AUC 0.27) — reported, not shipped** |
| Speed | **4.2 ms/image · 236 images/sec**, 7.77 M parameters, 31 MB, 390 MB peak GPU |
| Cross-validation | 0.59 ± **0.04** Dice over 3 folds — consistent, not lucky |

Every number above is produced by a script in `src/` and stored as JSON in
`results/`. Nothing is hand-entered.

---

## Quick start

### 1. Install

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

CPU-only works too — drop the `--index-url` line. Python 3.10+.

### 2. Run the live demo

```bash
python src/webapp.py
```

Then open **http://localhost:5000**. (Windows: double-click `run_demo.bat`.)

Upload any `.nii` / `.nii.gz` scan, choose Brain or Spine, and every processing
stage is shown with its own output and measured scores. No internet required.

---

## What the demo does

**Brain — 7 stages:** upload → histogram equalisation → CLAHE → **U-Net restoration**
→ **tumour detection** → CSF/grey/white tissue map → **Grad-CAM** attention.

**Spine — 6 stages:** upload → HE → CLAHE → **U-Net restoration** (the model matching
the detected sequence) → **SLIC region segmentation** → reconstruction-difference view (research only).

Two more views:

- **Validate against expert annotation** — pick any of 127 held-out BraTS cases and
  see the model's tumour mask beside the radiologist's, with the overlap score.
- **Inside the model** (`/model`) — a white-box trace: all 63 operations, tensor
  shapes, parameter counts, the loss functions, and the *actual* feature maps
  captured from a live forward pass.

---

## Folder map

```
.
├── run_demo.bat            Double-click to launch the demo (Windows)
├── requirements.txt        Python dependencies
├── README.md               You are here
│
├── src/                    ALL source code (37 modules)
│   ├── webapp.py             the live demo web application
│   ├── models.py             the U-Net architectures
│   ├── nifti_utils.py        NIfTI loading, normalisation, slicing
│   ├── metrics.py            every evaluation metric in one place
│   │
│   ├── train_*.py            training scripts (segmentation, enhancement)
│   ├── spine_autoencoder.py  self-supervised spine anomaly detection
│   ├── spine_pipeline.py     classical spine enhancement + SLIC ROI
│   ├── tissue_segmentation.py CSF / grey matter / white matter
│   ├── gradcam.py            attention maps
│   │
│   ├── dataset_stats.py      Stage-1 dataset analysis
│   ├── resolution_stats.py   Stage-1 MRI resolution analysis
│   ├── preprocessing_assessment.py  Stage-2 before/after property tables
│   ├── annotation_viz.py     Stage-2 label visualisation
│   ├── benchmark.py          latency / throughput / memory / complexity
│   ├── cross_validation.py   k-fold validation
│   ├── paper_comparison.py   comparison against classical baselines
│   ├── coco_export.py        COCO JSON mask export
│   └── comparison_graphs.py  the result figures
│
├── models/                 Trained weights (.pt) — 12 checkpoints
├── results/                All metrics as JSON + COCO exports
├── stats/                  Dataset analysis tables (CSV + JSON)
├── outputs/                Generated figures, overlays, demo page
│   └── demo/demo_page.html   offline slide-style walkthrough
├── docs/                   Report, knowledge base, glossary, demo script
├── handoff/                Presentation / report / video preparation pack
├── showcase/               Curated unseen scans for live demonstration
└── data/                   Datasets (not in git — see below)
```

---

## Datasets

| Dataset | Role | Annotations | Where |
|---|---|---|---|
| **BraTS2020** (4.47 GB, 369 cases) | standard **training** set | yes — expert tumour masks | Kaggle `awsaf49/brats20-dataset-training-validation` → `data/brats_subset/` |
| **Hackathon offline** (40 cases) | **testing / validation** | none | provided by the organisers |

We extracted and used **126 BraTS cases**. Splits are always at **patient level**,
never slice level — slices from one patient are highly correlated, so a slice-level
split would leak information and inflate every score.

Offline data is split **5 train / 5 test** per group, following the organisers'
instruction. Full per-case enumeration: `stats/splits_report.txt`.

**Format policy:** every volume is read directly from `.nii`/`.nii.gz` with nibabel.
No dataset file is ever converted to another format; PNG and JSON are outputs only.

> Datasets and model weights are excluded from git (~17 GB). See `.gitignore`.

---

## Reproducing the results

Run from the project root, in this order:

```bash
python src/dataset_stats.py                 # Stage 1 — dataset properties
python src/resolution_stats.py              # Stage 1 — MRI resolution
python src/annotation_viz.py                # Stage 2 — label visualisation
python src/preprocessing_assessment.py      # Stage 2 — before/after properties
python src/train_enhancement_brain.py       # Stage 3 — enhancement model
python src/train_segmentation_brain.py      # Stage 4 — segmentation model
python src/full_segmentation_metrics.py     # Stage 4 — complete metric suite
python src/paper_comparison.py              # Stage 3 — vs classical baselines
python src/benchmark.py                     # efficiency study
python src/comparison_graphs.py             # result figures
python src/build_demo_page.py               # offline walkthrough page
```

---

## Method in one paragraph

Both tasks use a **2D U-Net** (encoder → bottleneck → decoder with skip
connections). For **restoration**, the network is trained on clean scans paired
with synthetically degraded copies — degraded with **Rician noise** (the correct
noise model for MRI magnitude images, not Gaussian) plus a smooth multiplicative
**bias field** — using an `L1 + SSIM` loss so both intensity and structure are
preserved. For **segmentation**, all four modalities (T1/T1c/T2/FLAIR) are stacked
as input channels, because each reveals a different part of the tumour, and the
loss is `cross-entropy + soft Dice` — Dice is essential because tumour classes are
under 1 % of pixels and cross-entropy alone would collapse to predicting
background everywhere. Slice-based 2D processing keeps the memory footprint inside
a 6 GB laptop GPU.

**Spine uses no annotations at all**, as required: restoration is self-supervised and
region segmentation is unsupervised SLIC clustering. We also tested autoencoder-based
anomaly detection (train on healthy spines, treat reconstruction error as pathology)
and **validated it rather than assuming it worked — it failed**, see limitations.

---

## Honest limitations

- Quantitative segmentation scores are reported **only on BraTS**, where expert
  annotations exist. On the unlabelled hackathon data we report enhancement metrics
  and qualitative segmentation, and say so plainly rather than inventing numbers.
- **Our spine anomaly detector failed validation and the claim was withdrawn.** Scored
  on held-out cases it gives **AUC 0.27** — worse than chance, with healthy spines
  scoring *higher* than pathological ones — because the reconstruction error tracks
  image texture, not disease (`results/anomaly_validation.json`). The map is retained
  as a visualisation only. An unvalidated detector that fires on healthy patients is
  worse than no detector at all.
- The restoration model corrects noise and intensity artefacts. It does not
  invent anatomy — SSIM above 0.9 against the true scan is the evidence for that.

---

## Documentation

| File | Contents |
|---|---|
| `docs/REPORT.md` | Technical report, structured to the evaluation rubric |
| `docs/KNOWLEDGE_BASE.md` | Every model, training run, loss, metric and term |
| `docs/STAGES.md` | What each of the four stages required and how we met it |
| `docs/GLOSSARY.md` | Plain-language definition of every term used |
| `docs/DEMO_SCRIPT.md` | How to present the demo |
| `handoff/` | Slide-by-slide presentation plan, report draft, video script |
