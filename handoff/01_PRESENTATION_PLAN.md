# Presentation plan — paste into Gamma, slide by slide

**How to use this file:** each block below is one slide. The heading is the slide
title, "SAY" is the speaking note, and "IMAGE" names the exact file to upload from
the `images/` folder next to this document. Numbers are already final — do not
change them, they come from measured runs.

Deck length: **15 slides**, ~8 minutes.
Tone: confident, plain English. The project's strength is that every claim is measured.

---

## Slide 1 — Title

**MRI Enhancement & ROI Segmentation**
*Restoring degraded MRI scans and delineating the region of interest*

Team name · MedhaDrishti National AI Hackathon · Yugma TechFest 2.0 · JNNCE Shivamogga

IMAGE: `images/tumor_vs_gt.png` (use as a faded background if Gamma supports it)

SAY: "We built a system that takes a noisy MRI, cleans it, and finds the region a
doctor cares about — and every claim we make is measured, not asserted."

---

## Slide 2 — The problem

**Fast MRI scans trade quality for speed**

- To scan quickly, MRI machines produce **grainy** images with **uneven brightness**
- That grain can hide exactly what matters — the edge of a tumour, a compressed disc
- Two specific defects: **Rician noise** (the grain) and a **bias field** (uneven lighting)

SAY: "This isn't generic image noise. MRI has two specific, well-understood defects,
and we correct for those two specifically."

IMAGE: `images/enhancement_compare_brain.png`

---

## Slide 3 — Objectives

**Four stages, as set by the problem statement**

1. **Analyse** the dataset — 7 image properties, resolution, sub-modality division
2. **Preprocess** — normalise, resample, classical enhancement, then re-measure
3. **Enhance** — a deep-learning model that restores degraded scans
4. **Segment** — find the tumour / tissue / spine region of interest

SAY: "We completed all four, for both brain and spine."

IMAGE: `images/pipeline_diagram.png`

---

## Slide 4 — System architecture

**One U-Net backbone, two heads**

- **Encoder → bottleneck → decoder** with **skip connections**
- Restoration: 1 channel in → 1 channel out
- Segmentation: **4 channels in** (T1, T1c, T2, FLAIR) → 4 class channels out
- 7.77 M parameters · 31 MB · runs on a 6 GB laptop GPU

SAY: "Skip connections are the key — they carry fine detail straight from the
encoder to the decoder, so the output isn't blurred. In medicine the fine boundary
IS the diagnosis."

IMAGE: `images/pipeline_diagram.png`

---

## Slide 5 — Stage 1: Dataset analysis

**We measured every scan before touching it**

- 7 properties: contrast, complexity, sharpness, edge strength, noise, mean, deviation
- **BraTS2020**: 127 cases, uniform 240×240×155 at 1 mm
- **Hackathon data**: 40 cases, **highly heterogeneous** — voxels 0.25–1.3 mm, slices 3–13 mm
- Spine scans are **~2× more complex** than brain (more structures per image)

SAY: "That heterogeneity is exactly why Stage 2 resamples everything to a common
grid — you cannot feed mixed resolutions to one model."

IMAGE: `images/dataset_properties.png`

---

## Slide 6 — Understanding the annotations

**Only BraTS has expert labels — and they're extremely imbalanced**

| Label | Share of pixels |
|---|---|
| Background | **99.03 %** |
| Edema (swelling) | 0.71 % |
| Enhancing tumour | 0.17 % |
| Necrotic core | 0.10 % |

SAY: "This table decides our loss function. A model trained only with cross-entropy
scores 99% by predicting 'background' everywhere and finding nothing. That's why we
add Dice loss, which measures region overlap."

IMAGE: `images/annotation_labels.png`

---

## Slide 7 — Stage 3: Enhancement result **(required snapshot)**

**Dirty scan in → clean scan out**

Read the panels left to right: clean → degraded → CLAHE → **ours**

| Method | PSNR | SSIM |
|---|---|---|
| Degraded input | 18.05 | 0.196 |
| HE | 8.05 | 0.149 |
| CLAHE | 11.84 | 0.156 |
| **Ours (U-Net)** | **27.08** | **0.903** |

SAY: "Here's the striking part — every classical method scores *below* the noisy
input. They boost contrast but amplify the noise. Ours actually removes it."

IMAGE: `images/cmp_methods.png`

---

## Slide 8 — Why the classical methods fail

**They amplify what they cannot remove**

Measured noise level after each step (brain pathological):

| Stage | Noise |
|---|---|
| Preprocessed baseline | 0.0068 |
| HE | 0.0138 ↑ |
| CLAHE | 0.0106 ↑ |
| **Ours** | **0.0043 ↓** |

SAY: "Ours is the only stage in the entire pipeline that reduces noise."

IMAGE: `images/cmp_noise.png`

---

## Slide 9 — Stage 4: Segmentation result **(required snapshot)**

**AI tumour outline vs the radiologist's**

- Mean tumour **Dice 0.76**, enhancing tumour **0.84**
- Full suite: Jaccard, accuracy, sensitivity, specificity, precision, F1, Hausdorff, ASD, relative volume error
- Scored on patients the model **never trained on**

SAY: "Middle is our AI, right is the radiologist. They're nearly identical — and this
is a patient the model has never seen."

IMAGE: `images/tumor_vs_gt.png`

---

## Slide 10 — Segmentation, in detail

**Per-class performance**

IMAGE: `images/cmp_segmentation.png`

Supporting: `images/seg_confusion_matrix.png` (bright diagonal = classes rarely confused)

SAY: "Enhancing tumour — the active, growing part, the part that matters most
clinically — is our strongest class at 0.84."

---

## Slide 11 — Spine: no annotations allowed

**Two working methods, zero labels**

| Task | Method | Labels |
|---|---|---|
| Enhancement | Self-supervised restoration (one model per sequence) | none |
| Region segmentation | **Self-supervised CNN** (differentiable feature clustering) | none |
| Canal measurement | Geometric — canal width profile | none |

SAY: "The rules forbid annotations for spine, so everything here is unsupervised. The
restoration model trains against a degraded copy of the scan itself. For segmentation
we went further than clustering — this is a small network optimised on each scan, using
the image's own structure as its supervision. Look at the progression: k-means and SLIC
only group brightness and fragment the image; the trained network resolves the cord, the
vertebral chain and the soft tissue as coherent structures."

IMAGE: `images/spine_method_comparison.png`

---

## Slide 12 — The experiment that didn't work *(keep this slide — it wins trust)*

**We tried autoencoder anomaly detection. We tested it. It failed.**

The idea: train an autoencoder on *healthy* spines only; whatever it cannot
reconstruct should be the pathology.

We validated it instead of assuming it:

| | Normal spines | Pathological spines |
|---|---|---|
| Mean difference score | **0.020** | 0.017 |
| Range | 0.014 – 0.031 | 0.010 – 0.023 |

**AUC 0.27** — worse than a coin flip. Normal scans score *higher* than diseased ones.
The score tracks image texture, not disease.

**So we removed the detection claim** and present the map as a visualisation only.

SAY: "This is the slide we're most proud of. The method looked good and produced a
convincing-looking heat map. We measured whether it actually separates sick from
healthy spines — and it does not, it's worse than guessing. An unvalidated detector
that fires on healthy patients is worse than no detector at all, so we pulled the
claim. The numbers are in the repository."

IMAGE: `images/spine_anomaly.png`

---

## Slide 13 — One model per sequence beats one model for all

| Sequence | Pooled model | **Per-sequence (ours)** |
|---|---|---|
| T1 | 0.598 | **0.827** |
| T2 | 0.594 | **0.802** |
| STIR | 0.540 | **0.714** |

SAY: "T1, T2 and STIR are physically different images. We proved that treating them
separately wins on all three — measured on identical test slices, so it's a fair
comparison."

IMAGE: `images/cmp_modality.png`

---

## Slide 14 — Efficiency & validation

| Metric | Value |
|---|---|
| Latency | **4.2 ms / image** |
| Throughput | **236 images / sec** |
| Parameters | 7.77 M (31 MB) |
| Peak GPU memory | 390 MB |
| GPU utilisation | 84 % (enhance) / 98 % (segment) |
| Cross-validation | 0.59 ± **0.04** Dice, 3 folds |
| Convergence | epoch 25, overfitting gap ≈ 0 |

SAY: "The ±0.04 spread across folds is the important number — it means the result is
consistent, not a lucky split."

IMAGE: `images/segmentation_curves.png`

---

## Slide 15 — Summary & honesty

**Four measured claims**

IMAGE: `images/cmp_summary.png`

**What we deliberately do NOT claim:**
- No tumour accuracy numbers on unlabelled data — only on BraTS, where truth exists
- Our spine anomaly detector failed validation (AUC 0.27) — we report that, not a fake result
- The model corrects noise; it does not invent anatomy (SSIM > 0.9 proves it)

SAY: "We'd rather report a smaller honest number than a large invented one. Every
figure in this deck comes from a script in the repository, and can be re-run."

---

# Speaker cheat sheet — likely questions

**What loss function?**
Enhancement: `L1 + SSIM`. Segmentation: `Cross-Entropy + Dice`.
Dice is there because tumour is <1 % of pixels.

**What architecture?** 2D U-Net, encoder–decoder with skip connections, 7.77 M params.

**Why 4 input channels?** Each MRI sequence shows a different part of the tumour —
T1c the active rim, FLAIR/T2 the swelling. Stacking lets the model cross-reference.

**Why 2D not 3D?** 3D BraTS models document a 16 GB+ GPU requirement; we have 6 GB.
2D slice processing is the standard documented workaround.

**Dataset size?** BraTS2020 — **4.47 GB**, 369 cases; we extracted and used **126**.

**Optimiser?** Adam, learning rate 1e-3, with mixed precision (AMP).

**How do you avoid missing a tumour?** Detection runs on the **original** scan, never
the smoothed one, and we verified enhancement barely changes Dice (0.842 → 0.824).

**How do you avoid false positives?** Non-MRI images are rejected before any clinical
claim; a minimum-region-size guard suppresses single-pixel noise.

**Is it overfitting?** No — validation tracks training, gap ≈ 0, converged at epoch 25,
and 3-fold cross-validation agrees to ±0.04.
