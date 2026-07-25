# Presentation plan — paste into Gamma, slide by slide

**How to use this file:** each block below is one slide. The heading is the slide
title, "SAY" is the speaking note, and "IMAGE" names the exact file to upload from
the `images/` folder next to this document. Numbers are already final — do not
change them, they come from measured runs.

Deck length: **17 slides**, ~9 minutes.
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

## Slide 13 — Per-vertebra segmentation, and why we use a pretrained model

**The output the brief asks for is supervised by nature**

Naming *degenerative disc*, *herniation* or *stenosis* requires labelled examples.
We have **20 spine cases, zero annotations, no external training data**.
No model we train can produce that — it is a property of the problem, not a lack of effort.

**We proved it before reaching for help.** Four annotation-free methods, all measured:

| Method | Outcome |
|---|---|
| k-means / SLIC clustering | groups brightness — cannot separate adjacent vertebrae |
| Self-supervised CNN (ours) | resolves structure well, but semantic regions not instances |
| Autoencoder anomaly detection | **validated and failed — AUC 0.27**, withdrawn |
| Periodicity vertebra detection | canal reliable (91/92), vertebra step failed — not shipped |

**So, with the organisers' approval, we use SPINEPS** (European Radiology 2025,
Apache-2.0) for this one output — Dice **0.92** vertebrae, validated on 1,600+ subjects.
We supply it no annotations, we do not train it, and we claim none of its accuracy.

SAY: "We want to be completely transparent about this slide. Naming a herniated disc is
a supervised problem — you need labelled examples, and we have twenty unlabelled cases
and no external data. So we tested four annotation-free methods properly, and reported
that one of them failed outright. For per-vertebra instances specifically we use a
published, peer-reviewed model, with the organisers' approval, and we label it as
pretrained everywhere it appears. Using a model with quantified accuracy is also the
safer clinical choice — an under-constrained model would be confidently wrong."

IMAGE: `images/spine_method_comparison.png`

---

## Slide 14 — We measured the gap instead of asserting it *(new — strongest spine slide)*

**We used the pretrained model as a reference standard and scored our own work against it.**

| Structure | k-means | SLIC | **Ours (self-sup. CNN)** |
|---|---|---|---|
| Vertebral bodies | 0.263 | 0.204 | **0.257 ± 0.018** |
| Intervertebral discs | 0.047 | 0.047 | **0.090 ± 0.017** |
| Spinal canal + cord | 0.304 | 0.302 | **0.380 ± 0.041** |
| Posterior elements | 0.071 | 0.097 | **0.169 ± 0.032** |

**Ours wins on precision in all four structures** — 0.310 vs 0.194 on the canal,
0.116 vs 0.038 on posterior elements. The classical methods have high recall and
terrible precision: they *find* the structure, then bleed across the whole image.

**And we state the gap plainly:** our best is Dice **0.38** against SPINEPS's published
**0.92**, and ours numbers **zero** vertebrae. That is the justification, quantified.

SAY: "Rather than just claiming our method is good, we scored it. We used the pretrained
model as a reference standard and measured how much our annotation-free work recovers
without ever seeing a label. Two things came out. Ours is the best of the label-free
methods — highest precision on all four structures. And ours is still far from the
reference, and numbers zero vertebrae, because numbering needs labels. We report both
halves. The second half is exactly why we use a pretrained model for that one output."

IF ASKED *"isn't that Dice very low?"*: "Yes, and there are two honest reasons.
Unsupervised clusters are anonymous and spill past structure edges — that's the
precision number. And the metric itself is an upper bound: the reference has to pick
which of our unnamed clusters to grade, because we never gave the method a target."

IMAGE: `images/spine_vs_spineps.png`

---

## Slide 15 — One model per sequence beats one model for all

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

## Slide 16 — Efficiency & validation

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

## Slide 17 — Summary & honesty

**Four measured claims**

IMAGE: `images/cmp_summary.png`

**What we deliberately do NOT claim:**
- No credit for SPINEPS's accuracy — it is a pretrained model, labelled as such throughout
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

---

### Spine / pretrained-model questions

**Did you train SPINEPS?** No. We supply it no annotations and do not train it. We run
published weights and label its provenance everywhere it appears.

**What was it trained on?** The public SPIDER dataset plus the German National Cohort —
about 1,600+ annotated subjects. That external labelled data is exactly what we lack and
are not permitted to collect.

**What are its published numbers?** Dice 0.92 vertebrae, 0.967 discs, 0.958 canal — on
**their** test set. We claim none of it.

**What did it produce on our data?** 13 semantic structures and **17 numbered
vertebrae** on case SP11, in 401 seconds.

**What do the numbers 1–17 on the vertebrae mean?** Instance IDs — "this is a separate
bone from the one above it." Not a diagnosis, not a severity score, not an anatomical
name.

**What do the mask values 41–49, 60, 61, 62, 100 mean?** Structure types: **49**
vertebral body, **60** spinal cord, **61** spinal canal, **100** intervertebral disc,
**41–48** the arch and processes behind the body.

**Semantic vs instance — what's the difference?** Semantic says *what kind* of tissue a
pixel is; two neighbouring vertebrae get the same label. Instance says *which one*. The
instance step is the part that needs labels.

**Isn't using a pretrained model against the rules?** The organisers approved a public
pretrained model and asked us to justify it. Our own annotation-free pipeline is shown
beside it, never replaced by it, and we use it for one output only.

**Which parts are yours?** All brain enhancement and segmentation, all spine
enhancement, the self-supervised CNN segmentation, the canal morphometry, and every
validation. Not ours: the SPINEPS masks, labelled as pretrained everywhere.

**Isn't your spine Dice very low?** Yes — 0.38 at best. Two honest reasons: unsupervised
clusters are anonymous and bleed past structure edges (precision 0.19–0.31), and the
metric is an upper bound because the reference has to pick which unnamed cluster to
grade. We report it rather than hide it — the gap is the argument.

**Why is the CNN reported as mean ± sd?** It is stochastic. It is seeded, but cuDNN
picks nondeterministic kernels, so runs differ. One number would not be reproducible, so
we report three runs.

**What would you improve?** Top of the list: annotate 20–30 slices and fine-tune a
supervised head on our existing features — that turns anonymous regions into named
structures. Then 3D instead of 2D (blocked by 6 GB VRAM), and distilling SPINEPS output
into pseudo-labels so instance numbering no longer needs SPINEPS at inference.
