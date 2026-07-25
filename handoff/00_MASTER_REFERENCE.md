# MASTER REFERENCE — everything, in one file

*Single-file consolidation of every document in this project, plus every*
*measured number, plus what every source file does. Built by*
*`src/build_master_reference.py` — re-run it after any change.*

**How to use this under questioning.** Search it (Ctrl+F) rather than reading
it. Part I is the fast path: it holds the twelve numbers to memorise and the
direct answers. Part VI holds every figure we have ever measured, so if you
are asked something specific it is almost certainly there.

**The one rule:** if you cannot find a number, say *"I don't have that
memorised — it's in `results/`, I can pull it up now"*. That reads as rigour.
Inventing a number is the only losing move.

---

## Contents

| Part | Contains |
|---|---|
| **I** | Viva prep — the 12 key numbers, loss functions, inputs, every likely question |
| **II** | Full detail — architectures, every training run, every method, what we ditched |
| **III** | The pretrained-model justification, as given to the organisers |
| **IV** | Glossary — every term, in plain words |
| **V** | What each of the four stages asked for |
| **VI** | Every measured number, generated from `results/*.json` |
| **VII** | What each of the 52 source files does |

---


# PART I — VIVA: EVERY QUESTION AND ITS ANSWER

*Source: `docs/VIVA_PREP.md`*

## VIVA PREP — every question they can ask, with the answer

Read top to bottom once. The numbers here are all measured; do not invent new ones.
If you do not know something, say **"I don't have that number to hand, it's in
`results/<file>.json`"** — that is a strong answer. Guessing is the only losing move.

---

### 0. THE 12 NUMBERS TO MEMORISE

| Thing | Number |
|---|---|
| Enhancement loss | **L1 + SSIM** |
| Segmentation loss | **Cross-Entropy + soft Dice** |
| Tumour Dice (mean / enhancing) | **0.76 / 0.84** |
| Enhancement PSNR / SSIM (BraTS) | **30.3 dB / 0.965** |
| Model size | **7.77 M params, 31 MB** |
| Speed | **4.24 ms/image, 236 img/sec** |
| BraTS dataset | **4.47 GB archive, 369 cases, we used 126** |
| Background share of pixels | **99.03 %** |
| Spine per-sequence win | **3 / 3 sequences** |
| Spine anomaly detector | **AUC 0.266 — FAILED, withdrawn** |
| Our spine seg vs SPINEPS | **best Dice 0.38 vs their published 0.92** |
| SPINEPS vertebrae found | **17 instances, 13 structure types** |

---

### 1. FOUNDATIONS — "what is X?"

**What is AI / ML / DL?**
AI is the broad goal of machines doing tasks that need intelligence. **Machine
learning** is the subset where the machine learns rules from data instead of being
programmed with them. **Deep learning** is the subset of ML using multi-layer neural
networks that learn features themselves. We use deep learning; our classical baselines
(HE, CLAHE) are neither — they are fixed formulas.

**What is OpenCV and where did you use it?**
An open-source computer-vision library. We use it for the *classical, non-learned* parts:
`cv2.equalizeHist` (HE), `cv2.createCLAHE` (CLAHE), resizing, colour conversion, drawing
overlays, and the Laplacian/Sobel operators for the Stage-1 sharpness and edge measures.
**No learning happens in OpenCV** in this project.

**What is PyTorch?** The deep-learning framework we train and run our models in.
Version 2.6.0+cu124, GPU-enabled.

**What is a CNN?**
A convolutional neural network. It slides small learnable filters across the image, so
it detects the same pattern anywhere in the picture, and it learns edges in early layers
building to whole structures in deeper layers.

**What is a U-Net, in plain words?**
An encoder–decoder network shaped like a U. The **encoder** downsamples, growing the
receptive field so the network sees context. The **decoder** upsamples back to full
resolution. The **skip connections** copy feature maps straight across from encoder to
decoder at matching resolutions.

**Why does U-Net suit medical imaging?**
Two reasons. Output is the same size as the input, which is what segmentation needs. And
skip connections preserve fine boundary detail that pure downsampling destroys — **in
medicine the fine boundary IS the diagnosis**.

**What would happen without skip connections?**
Blurred, smeared boundaries. We know because our anomaly autoencoder deliberately has
**no** skips — there, skips would copy the lesion straight to the output and erase the
very signal we were looking for.

**What is a convolution / kernel / stride / padding?**
A convolution slides a small weight matrix (kernel, 3×3 here) over the image computing
weighted sums. Stride is the step size. Padding adds a border so output size is
controlled. We use 3×3 kernels, stride 1, padding 1 in conv blocks; downsampling is by
2×2 max-pooling.

**What is batch normalisation? ReLU?**
BatchNorm normalises activations per mini-batch, which stabilises and speeds training.
ReLU is `max(0, x)` — the non-linearity; without a non-linearity the whole network would
collapse to a single linear function.

**What is an epoch / batch / learning rate?**
An epoch is one full pass over the training set. A batch is the group of samples
processed before one weight update (ours: 8). Learning rate is the step size (ours:
1e-3).

**What is overfitting and did you overfit?**
Overfitting is memorising training data and failing on new data. **We did not**:
validation loss tracks training loss with a gap of ≈ 0, converged around epoch 25, and
3-fold cross-validation agrees to **±0.04**.

**What is transfer learning? Did you use it?**
Reusing a model trained on one task for another. **We did not train with it.** SPINEPS
is a pretrained model we *run as-is* for one output — we do not fine-tune it.

**Supervised vs unsupervised vs self-supervised?**
- **Supervised** — you have labels. Our brain tumour segmentation (BraTS masks).
- **Unsupervised** — no labels, find structure. Our k-means / SLIC / GMM.
- **Self-supervised** — you *create* the labels from the data. Our enhancement: degrade a
  scan, train the model to restore it back to itself. Our spine CNN: the image's own
  structure is the supervision.

---

### 2. LOSS FUNCTIONS — they asked this and you did not know. Know it cold.

**Enhancement loss = L1 + SSIM.**
- **L1** = mean absolute error, `mean(|pred − target|)`. Fixes per-pixel brightness.
  Chosen over L2/MSE because L2 over-penalises large errors and produces blur.
- **SSIM** = Structural Similarity Index. Compares local luminance, contrast and
  structure in a sliding window; 1.0 = identical. We use `1 − SSIM` as the loss term.
- **Why both:** L1 alone gives an image that is numerically close but structurally
  wrong — smooth and washed out. SSIM is the term that forces the anatomy to be right.

**Segmentation loss = Cross-Entropy + soft Dice.**
- **Cross-Entropy** = per-pixel classification loss, `−log(probability of correct class)`.
- **Dice** = overlap measure, `2|A∩B| / (|A|+|B|)`. "Soft" means computed on
  probabilities so it is differentiable and can be trained through.
- **WHY DICE — the answer that matters:** background is **99.03 %** of pixels. A model
  trained on cross-entropy alone reaches 99 % accuracy by predicting "background"
  everywhere and finding **no tumour at all**. Dice measures region overlap, so that
  degenerate solution scores zero. **This is the single most important design decision in
  the project.**

**Why not just accuracy as a metric?** Same reason — 99 % accuracy is achievable by
finding nothing. That is why we report Dice, Jaccard, sensitivity and specificity.

**What optimiser?** Adam, lr 1e-3, with AMP (automatic mixed precision) for speed and
memory. AMP runs parts of the network in 16-bit; GradScaler prevents gradient underflow.

---

### 3. INPUTS AND OUTPUTS — "what did you feed it?"

**Enhancement model:**
- IN: **1 channel**, one greyscale slice, 224×224, values normalised to [0, 1].
- OUT: **1 channel**, same size, **sigmoid** so output stays in [0, 1].

**Segmentation model:**
- IN: **4 channels** — T1, T1c, T2, FLAIR stacked, 224×224.
- OUT: **4 channels** of raw logits (no softmax inside the model — CrossEntropyLoss
  expects logits; adding softmax would apply it twice).

**Why 4 input channels?** Each MRI sequence reveals a different part of the tumour —
**T1c** shows the active enhancing rim, **FLAIR/T2** show the oedema, **T1** gives
anatomy. Stacking lets the network cross-reference them, which measurably improves Dice
over any single sequence.

**What are the output classes?**
`0` background · `1` necrotic/non-enhancing core · `2` oedema (swelling) ·
`3` enhancing tumour.
**Note:** BraTS labels enhancing tumour as **4**, and label 3 does not exist in the raw
data. We remap 4 → 3 so the classes are contiguous, which PyTorch requires.

**How did you make noisy/clean training pairs when MRI has no clean copy?**
Self-supervision. Take the clean scan, **degrade it ourselves**, and train the model to
restore it. The degradation is MRI-correct:
- **Rician noise** — the true noise distribution for MRI magnitude images. Using plain
  Gaussian would be a genuine methodological error.
- **Bias field** — a smooth multiplicative field simulating RF coil inhomogeneity, which
  maps directly onto the brief's "artifact correction".
- **Mild blur.**
σ range was later widened from 0.02 to **0.02–0.20** so the model survives very noisy
real uploads.

---

### 4. DATASETS

**Which dataset and how big?** **BraTS2020** (Kaggle `awsaf49/brats20`) —
**4.47 GB archive**, 9.9 GB uncompressed, **369 cases**, we extracted and used **126**.
Each case: T1, T1c, T2, FLAIR + expert segmentation, 240×240×155, **1 mm isotropic**.
*(If you previously said 7 GB or 42 GB — those were wrong. 4.47 GB.)*

**And the hackathon data?** 10 normal + 10 pathological brain; 10 normal + 10
pathological spine. **No annotations at all.** Voxels 0.25–1.3 mm, slice thickness
3–13 mm — highly heterogeneous, unlike BraTS.

**Train/test/validation counts?** Segmentation 16 train / 4 val; enhancement 32 / 8; plus
3-fold cross-validation over 21 cases. Offline data **5 train / 5 test** in each of the
four groups per the coordinator's rule. Per-case IDs in `stats/splits_report.txt`.

**How did you split, and why does it matter?** **Case-level (by patient), never
slice-level.** Slices from one patient are near-identical; splitting by slice puts nearly
the same image in train and test, which leaks information and inflates every score. This
is a real methodological trap and we avoided it deliberately.

**Did you convert the .nii files?** **No.** Every volume is read directly from
`.nii`/`.nii.gz` with nibabel. PNG and JSON are outputs only, never pipeline inputs.

**Why is the class distribution important?** Background 99.03 %, oedema 0.71 %, enhancing
0.17 %, necrotic 0.10 %. It dictates the loss function (see §2).

---

### 5. RESULTS — what we gained

**Enhancement (BraTS held-out):** PSNR **30.3 dB**, SSIM **0.965**. Under heavy noise
SSIM goes 0.19 → **0.89**.

**Against the classical family, identical degraded slices:**

| Method | PSNR | SSIM |
|---|---|---|
| Degraded input | 18.05 | 0.196 |
| Histogram equalisation | 8.05 | 0.149 |
| Adaptive HE | 6.35 | 0.133 |
| CLAHE | 11.84 | 0.156 |
| **Ours (U-Net)** | **27.08** | **0.903** |

**The line to say:** *"Every classical method scores below the noisy input. They boost
contrast and amplify the noise with it. Ours is the only stage that removes it."*

**Noise measured after each step:** baseline 0.0068 → HE 0.0138 ↑ → CLAHE 0.0106 ↑ →
**ours 0.0043 ↓**.

**Segmentation:** mean tumour Dice **0.76**, enhancing **0.84**, necrotic and oedema
lower. Full suite computed: Jaccard, accuracy, sensitivity, specificity, precision, F1,
Hausdorff distance, average surface distance, relative volume error.

**Efficiency:** 7.77 M params, 31 MB, **4.24 ms/image**, 236 images/sec, peak 385–390 MB
GPU, 84 % GPU utilisation on enhancement / 98 % on segmentation, CPU ~15 %.

**Spine enhancement, per-sequence vs pooled (SSIM, same test slices):**

| Sequence | Pooled | **Per-sequence (ours)** |
|---|---|---|
| T1 | 0.598 | **0.827** |
| T2 | 0.594 | **0.802** |
| STIR | 0.540 | **0.714** |

**3/3 wins.** Why: STIR deliberately suppresses fat, so its intensity statistics are
genuinely unlike T1's. One model averages across contradictory targets.

---

### 6. SPINE — the hard questions

**Does your model find the spine problem / lesion / stenosis location?**
**No, and we say so plainly.** That is the honest limit of our own work. We built an
autoencoder anomaly detector, validated it, and **it failed: AUC 0.266 — worse than
chance**, with normal spines scoring *higher* (0.0199) than pathological (0.0167). We
tested five different scoring statistics (mean 0.304, max 0.500, p99 0.388, p95 0.312,
top-1 % 0.413) — all at or below chance, so it was not a scoring artefact. **We removed
the claim.** An unvalidated detector that fires on healthy patients is more dangerous
than no detector at all.

**So what DO you do for spine pathology?**
We **measure instead of guess**. Spinal stenosis *is* narrowing of the canal, so we
measure canal width: segment the CSF column, find its axis by **PCA** so orientation
does not matter, sample width perpendicular to that axis. Narrowing ratio = narrowest ÷
that same canal's own median width, so patient size and resolution cancel out.
**Canal detected on 91 of 92 slices.** Pathological canals trend narrower
(**0.485 vs 0.557**, AUC 0.69) — the direction stenosis predicts — but with 10 vs 9
patients this is **not significant (p = 0.089)**. We report the measurement and the
trend, and stop short of a diagnosis.

**Why is a single slice's narrowing ratio unstable?** Because it is — one normal scan
ranges 0.29–0.66 across its own slices. The app reports the **median over 5 slices**,
matching the validation protocol.

**How does a model train on spine with no labels at all?**
Differentiable feature clustering (Kanezaki, ICASSP 2018). A small CNN trains **on the
single scan in front of it**, from scratch, ~120 iterations, using three constraints and
no labels: **commit** (each pixel pushed to its own argmax class), **continuity**
(neighbours pushed to agree), **balance** (entropy term stopping collapse into one
class). Structures found are **emergent** — we offer 12 candidates, it settles on 9–10.

**Any bugs in that? (Say yes — it shows you understand it.)**
Two, both measured. (1) Collapse to **1 class**: we first masked background by zeroing
the *features*; every background pixel then argmaxed to the same class and cross-entropy
dragged the whole image into it. Fix: mask the **loss**, not the features. (2) Collapse
to **2 classes**: applying the superpixel prior every iteration compounded with the CE
feedback — measured 2 classes with it in-loop vs 11 without. Fix: apply it once, after
training.

**How good is your spine segmentation, honestly?**
Against SPINEPS as reference standard: our CNN has the **highest precision of all three
annotation-free methods on all four structures** (0.191 bodies / 0.050 discs / 0.310
canal / 0.116 posterior elements) and the best Dice on three of four. Best Dice **0.380**
(canal) against their published **0.920**, and **zero** numbered vertebrae. We report
both halves.

**Why is your Dice so low?** Two real reasons. Unsupervised clusters are anonymous and
bleed past structure edges — that is what the precision numbers show. And the metric is
an **oracle-assisted upper bound**: the reference has to pick which of our unnamed
clusters to grade, because the method was never given a target to fit.

---

### 7. THE PRETRAINED MODEL — expect hostility, answer calmly

**What is SPINEPS?** Möller et al., *European Radiology* **2025**, Apache-2.0, built on
**nnU-Net**. Two phases: **semantic** (13–14 structure types) and **instance**
(individually numbered vertebrae).

**What was it trained on?** The public **SPIDER** dataset + the **German National
Cohort**, about **1,600+ annotated subjects**. That external labelled data is exactly
what we do not have and were not permitted to collect.

**Its accuracy?** Dice **0.92** vertebrae / **0.967** discs / **0.958** canal — **their
numbers on their test set. We claim none of them.**

**Did you train it?** No. We supply it no annotations and do not train it. We run
published weights.

**Isn't that cheating?** The organisers approved a public pretrained model and asked us to
justify it — `docs/PRETRAINED_MODEL_JUSTIFICATION.md` is that justification. We use it
for **one output only**, our own annotation-free pipeline is shown beside it, and its
provenance is labelled everywhere it appears.

**Why couldn't you do it yourselves?** Naming a herniated disc, or numbering a vertebra,
is **supervised by nature** — a model can only output "L4" if it has seen examples
labelled L4. With 20 unlabelled cases and no external data, **no model we train can
produce it.** We proved that with four annotation-free methods, and reported that two
failed.

**What did it give you on your data?** **13 semantic structures** and **17 numbered
vertebrae** on case SP11, in **401 s**.

**What do the numbers on the vertebrae mean?** **Instance IDs** — "this is a separate
bone from the one above it." **Not** a diagnosis, **not** severity, **not** an anatomical
name.

**What do the mask values mean?** Structure types: **49** vertebral body, **60** spinal
cord, **61** spinal canal, **100** intervertebral disc, **41–48** the arch and processes
(arcus, spinous, costal, articular).

**Semantic vs instance?** Semantic says *what kind* of tissue — two neighbouring
vertebrae get the same label. Instance says *which one*. Instance is the part that needs
labels.

**Why does the demo run it on GPU but say the vertebra numbering is precomputed?**
The instance phase forwards its whole working volume at once and needs **~12.4 GiB**;
this card has 6 GB. The semantic phase fits and runs live in under a minute. We say which
is which rather than implying both ran live.

**Did you try to fix that?** Yes, and the first two hypotheses were **wrong**, which is
worth telling. We thought the scan's resolution was to blame — downsampling 3.6× gave a
**byte-identical** 12.44 GiB failure. The log showed why: the instance phase does not read
the input scan at all, it reads the semantic mask SPINEPS just wrote, stored in its own
internal 0.75 mm space regardless of input. We then found SPINEPS has an
`auto_crop_to_spine` setting that **silently disables itself** when slice thickness
exceeds 1.2 mm — ours is 4.4 mm — but forcing it requires downloading a separate 732 MB
locator model. So: instance on CPU.

---

### 8. EXPLAINABILITY AND SAFETY

**How do you know the model looks at the right place?** **Grad-CAM** — gradient-weighted
class activation mapping. We take the gradient of the tumour score w.r.t. the last
convolutional feature maps, weight the channels by those gradients, and get a heat map.
The hot region sits on the lesion.

**Any artefact in that?** Yes, and we fixed it: without masking, normalisation put a
bright ring around the skull edge. We now restrict the score to predicted-tumour pixels
and mask the CAM to the brain.

**Does enhancement erase the tumour?** *(Your biotech friend's concern — verified.)*
**No.** Segmentation Dice on original 0.842 vs on enhanced 0.824 — a change of 0.017.
And critically, **tumour detection always runs on the ORIGINAL slice**, never the
smoothed one, so enhancement cannot remove a finding even in principle.

**How do you avoid false negatives?** Detection runs on the original scan; the threshold
is set sensitivity-first (150 px minimum region) so small tumours are not missed.

**How do you avoid false positives?** Non-MRI images are rejected before any clinical
claim is made, and a minimum-region-size guard suppresses single-pixel noise.

**Is there a quality loop?** Yes — `_enhance_refine`, **capped at 3 passes**. Capped
deliberately: an uncapped loop would keep smoothing until it destroyed anatomy.

---

### 9. "WHAT WOULD YOU IMPROVE?" — never say "nothing"

1. **Annotate 20–30 spine slices and fine-tune a supervised head** on our existing
   features. Highest-value next step — it turns anonymous regions into named structures.
2. **3D instead of 2D.** Vertebrae are 3D objects; through-plane context would improve
   boundaries. Blocked by 6 GB VRAM.
3. **Distil SPINEPS output into pseudo-labels** and train a small model on them, so
   instance numbering no longer needs SPINEPS at inference.
4. **More cases.** The canal-width trend points the right way but p = 0.089; ~40 per
   group would likely settle it.
5. **Ensembling and test-time augmentation** for brain — reliable small Dice gains.
6. **Boundary-aware loss** — our Hausdorff is the weakest metric relative to Dice, which
   points at boundary quality.
7. **Deterministic cuDNN kernels** so results are exactly reproducible rather than
   mean ± sd.

---

### 10. TRAPS — questions designed to catch you

**"Your accuracy is on BraTS, not on our data."** Correct, and deliberate. BraTS is the
only dataset with expert annotation. On your unlabelled data we report enhancement
metrics and qualitative segmentation rather than inventing numbers. Fabricating a Dice
against no ground truth would be the real failure.

**"Why 2D when everyone uses 3D?"** Published 3D BraTS models document a **16 GB+** VRAM
requirement; this is a 6 GB laptop card. 2D slice-wise is a standard, citable workaround
— not corner-cutting.

**"Your spine work doesn't detect anything."** Correct. We measure canal width and we
segment structure. We built a detector, validated it, it failed at AUC 0.266, and we
withdrew it. We would rather show you a negative result than a false positive on a
healthy patient.

**"Isn't your enhancement just making things up?"** No — SSIM above 0.9 against the true
scan is the evidence. A model inventing anatomy would diverge structurally, and SSIM is
precisely the metric that measures structural divergence.

**"Why should we trust one slice?"** You shouldn't, which is why the canal measurement
reports a **median over 5 slices**, the CNN reports **mean ± sd over 3 runs**, and
segmentation reports **3-fold cross-validation ±0.04**.

**"Run it on this file."** `python demos/run_everything.py`, or a single stage:
`python demos/04_brain_tumour_seg.py`. Every script prints its own numbers.

**If you genuinely do not know:** *"I don't have that figure memorised — it's in
`results/…json` and I can pull it up right now."* Then do it. That reads as rigour.
Never invent a number.

---


# PART II — WHAT WE BUILT, IN FULL DETAIL

*Source: `docs/KNOWLEDGE_BASE.md`*

## MASTER KNOWLEDGE BASE — everything we built, in one place

Answer any judge question from here. Organized: Models → Training → Losses →
Metrics → Preprocessing → Algorithms → Datasets → Numbers → Decisions → Q&A.

---

### 1. THE MODELS (architecture)

#### The shared backbone — 2D U-Net (`models.py`)
- Shape: **encoder → bottleneck → decoder, with skip connections** (the "U").
- **conv_block** = Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU.
- **Encoder** (downsampling, MaxPool 2×2 between): channels 32 → 64 → 128 → 256.
- **Bottleneck**: 512 channels.
- **Decoder** (upsampling via ConvTranspose2d 2×2): 256 → 128 → 64 → 32,
  each concatenated with the matching encoder features (**skip connections**).
- `base_filters = 32` (the starting channel count).

#### Model A — EnhancementUNet
- **1 channel in → 1 channel out**, final Conv1×1 + **sigmoid** (output 0–1).
- Job: noisy MRI slice in → clean slice out (denoising / artifact removal).
- ~**7.77 million parameters**, ~31 MB.

#### Model B — SegmentationUNet
- **4 channels in → 4 channels out** (raw logits, no softmax in the model).
- 4 in = the 4 MRI modalities stacked (T1, T1c, T2, FLAIR).
- 4 out = the 4 classes (background, necrotic core, edema, enhancing tumour).
- ~**7.77 million parameters**, ~31 MB.

---

### 2. EVERY TRAINING RUN

#### Run 1 — Brain Segmentation (BraTS2020)
- Data: 20 BraTS cases → **16 train / 4 val** (case-level split, seed 42).
- ~2075 train / 504 val 2D slices.
- **30 epochs**, batch size 8, **Adam**, lr 1e-3, **AMP** (mixed precision).
- Loss: **Cross-Entropy + soft Dice**.
- Result: **mean tumour Dice 0.73** (enhancing 0.796, edema 0.708, necrotic
  0.685), Jaccard 0.66/0.55/0.52, Hausdorff 8.0/23.6/9.4, ASD 0.81/2.0/0.72.
  Convergence epoch 25, overfitting gap −0.08 (no overfitting).
- File: `train_segmentation_brain.py` → `segmentation_model.pt`.

#### Run 2 — Brain Enhancement (BraTS FLAIR)
- Data: 40 BraTS cases, FLAIR only. **18 epochs**, batch 8, Adam lr 1e-3, AMP.
- Loss: **L1 + SSIM**.
- Degradation to make training pairs: **Rician noise + bias field + blur**,
  noise σ range **0.02–0.20** (wide, so it cleans genuinely noisy scans).
- Result: **PSNR 30.3 dB, SSIM 0.965**, FSIM 0.985, UQI 0.988, LPIPS 0.041.
- On heavy noise: restores SSIM 0.19→**0.89**. File: `enhancement_model_brain.pt`.

#### Runs 3-6 — Offline Enhancement (Spine + Brain, 4 groups)
- Groups: spine_normal, spine_pathological, brain_normal, brain_pathological.
- Each: **5 train / 5 test** cases (the coordinators' rule), 25 epochs, L1+SSIM.
- 3-way comparison per group (degraded input vs CLAHE vs our U-Net), all vs clean:
  | Group | input SSIM | CLAHE SSIM | **our U-Net SSIM** |
  |---|---|---|---|
  | spine_normal | 0.40 | 0.28 | **0.82** |
  | spine_pathological | 0.52 | 0.39 | **0.75** |
  | brain_normal | 0.35 | 0.26 | **0.87** |
  | brain_pathological | 0.19 | 0.15 | **0.96** |
- Finding: **CLAHE lowers fidelity** (redistributes contrast, doesn't restore);
  our U-Net wins every group. Files: `enhancement_model_<group>.pt`.

#### Run 7 — Cross-validation (brain segmentation)
- **3-fold cross-validation**, 21 cases, 25 epochs/fold.
- **CV accuracy = 0.59 ± 0.04 mean tumour Dice** (folds 0.62 / 0.62 / 0.54).
- Why lower than the 0.73 headline: each fold trains on only ~14 cases (vs 16
  for the main model) and averages over 3 different splits. **The key number is
  the tiny ±0.04 std — it proves the model is CONSISTENT, not lucky.** That is
  exactly what cross-validation is meant to show. File: `cross_validation.py`.

---

#### Run 8 — Spine Anomaly Autoencoder (self-supervised lesion localisation)
- **Trained on NORMAL spine T2 only — 719 slices**, 80 epochs, Adam **5e-4**,
  loss **L1 + 0.5·SSIM**, no AMP (stability), final loss 0.124.
- Architecture: **bottleneck convolutional autoencoder, NO skip connections**
  (encoder 1→32→64→128→128 with 4 MaxPools → 14×14 bottleneck → mirrored
  decoder with ConvTranspose). Skips are omitted *on purpose*: they would copy
  the input (lesion included) to the output and destroy the anomaly signal.
- **How detection works:** it can only rebuild "healthy spine". On a diseased
  spine the lesion cannot be reconstructed, so |input − reconstruction| peaks
  exactly at the abnormality. Post-processing: blur → background mask →
  subtract median foreground error (keep only *excess*) → ^1.5 → normalise →
  box the largest connected component above the 90th percentile.
- Result: flags a focal "suspected" region on all 9 pathological cases
  (anomaly scores 0.013–0.022). File: `spine_autoencoder.py`.

#### Run 9 — Modality-specific spine enhancement
- One enhancement U-Net per spine modality (**T1, T2, STIR**) on spine_normal,
  25 epochs each, vs the pooled all-modality model.
- Fair comparison in `compare_modality_models.py`: both evaluated on the SAME
  held-out per-modality test slices (each run's own metrics use different test
  sets and must not be compared directly).

---

### 3. EVERY LOSS FUNCTION (what + why)

- **L1 loss (Mean Absolute Error)** — average pixel-wise difference. Used for
  enhancement; keeps output numerically close to the clean scan.
- **SSIM loss** = 1 − SSIM (differentiable, `ssim.py`). Preserves *structure*
  (edges, texture), so the enhanced image looks right, not just numerically close.
- **Cross-Entropy loss** — per-pixel classification loss for segmentation.
- **Soft Dice loss** — 1 − Dice overlap. Directly maximizes region overlap.
  **Why we add it:** the tumour is tiny vs background; plain cross-entropy would
  predict "background everywhere". Dice fixes this **class imbalance**.
- Enhancement total = **L1 + SSIM**. Segmentation total = **Cross-Entropy + Dice**.

**Optimizer:** Adam, learning rate 1e-3. **AMP** (Automatic Mixed Precision) for
speed + less memory. **GradScaler** to keep fp16 training stable.

---

### 4. EVERY METRIC (definition in plain words)

#### Enhancement — full-reference (need clean copy to compare)
- **PSNR** (Peak Signal-to-Noise Ratio) — cleanliness in dB, higher better.
- **SSIM** (Structural Similarity) — structural closeness 0–1, higher better.
- **MSE / RMSE** — average pixel error, lower better.
- **UQI** — universal quality index. **FSIM** — feature/edge similarity.
- **GMSD** — gradient-magnitude similarity deviation, lower better.
- **VIF** — visual information fidelity. **LPIPS** — AI "looks-the-same" score, lower better.

#### Enhancement — no-reference (single image, no clean copy)
- **BRISQUE / NIQE / PIQE** — blind quality scores, lower better.
- **Entropy** — information/detail content.

#### Segmentation
- **Dice (DSC)** — overlap with ground truth, 0–1 (main score).
- **Jaccard (IoU)** — stricter overlap. **Hausdorff Distance** — worst-case
  boundary error (lower better). **ASD** — average surface distance.
- **Sensitivity/Recall, Specificity, Precision, F1, Relative Volume Error.**

---

### 5. EVERY PREPROCESSING STEP (Stage 2 — classical, NO model)

1. **Load** NIfTI (.nii/.nii.gz) with nibabel.
2. **Normalize** to [0,1] using robust percentiles (0.5–99.5), per volume.
3. **Slice** 3D → 2D axial; skip near-empty slices (<2% non-zero).
4. **Resize** to 224×224 (cubic) + **clip to [0,1]** (cubic overshoots at edges).
5. Masks resized **nearest-neighbour** (cubic would invent label values).
6. **Remap BraTS labels** 4 → 3 (label 3 unused → make classes contiguous 0-3).
7. **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) + **HE**.
8. **Augmentation**: horizontal flips (train only).
9. **Synthetic degradation** (for enhancement pairs): Rician + bias field + blur.

---

### 6. EVERY ALGORITHM / TECHNIQUE

- **CLAHE** — local contrast enhancement (clipLimit 2.0, 8×8 tiles).
- **HE** — histogram equalization (global contrast).
- **Rician noise** — the correct MRI magnitude-image noise model (not Gaussian).
- **Bias field** — smooth multiplicative RF-coil intensity artifact.
- **k-means** — unsupervised intensity clustering (spine ROI, tissue).
- **GMM (Gaussian Mixture)** — softer tissue clustering (CSF/GM/WM).
- **SLIC superpixels** — groups nearby similar pixels into ~250 coherent
  superpixels; we cluster those by mean intensity → connected disc/vertebra/
  cord regions (replaced the speckly per-pixel k-means for spine ROI).
- **Autoencoder anomaly detection** — train on healthy only, treat reconstruction
  error as pathology. Self-supervised. **We tested this and it failed on our data
  (AUC 0.27, worse than chance); claim withdrawn.**
- **Otsu multi-threshold** — alternative unsupervised split.
- **Grad-CAM (Seg-Grad-CAM)** — attention heatmap, gradient of tumour score
  w.r.t. decoder features, restricted to the predicted tumour region.
- **Immerkær estimator** — single-image noise-σ. **Michelson contrast**,
  **Laplacian variance** (sharpness), **Sobel** (edge), **Shannon entropy** (complexity).

---

### 7. THE DATASETS

- **BraTS2020** (Kaggle awsaf49): **4.47 GB**, **369 cases**; we extracted
  **126 complete cases**. Each case = T1, T1c(T1ce), T2, FLAIR + seg mask,
  240×240×155, **1 mm isotropic**. Labels 0=bg, 1=necrotic, 2=edema, 4=enhancing.
- **Offline Brain**: 10 Normal (S1–S10, Philips scanner) + 10 Pathological
  (BRP1–BRP10, BraTS-geometry). No ground truth.
- **Offline Spine**: 10 Normal (SP1–SP10) + 10 Pathological (SP11+). No GT.
- **Resolution finding**: BraTS uniform 1 mm; offline highly heterogeneous
  (voxel 0.25–1.3 mm, slice thickness 3–13 mm) → why we resample to 224×224.

---

### 8. HEADLINE NUMBERS (memorize)

- Segmentation: **mean tumour Dice 0.73**, enhancing tumour **0.80**.
- Enhancement (BraTS): **PSNR 30.3, SSIM 0.965**; heavy noise 0.19→0.89.
- Offline enhancement beats CLAHE everywhere (e.g. brain-path SSIM 0.19→0.96).
- Model: **7.77 M params, 31 MB, 4.2 ms/image, 236 images/sec**, peak 390 MB GPU.
- Utilization: Enhancement 84% GPU, Segmentation 98% GPU, CPU ~15%.

---

### 9. DESIGN DECISIONS (with justification)

- **2D slices, not 3D** — 3D BraTS models need 16 GB+ GPU; we have 6 GB. 2D is a
  standard, documented workaround.
- **Case-level split** (never slice-level) — slices from one patient are
  correlated; slice-split leaks info and inflates metrics.
- **Multi-modal (4-channel) segmentation** — each modality shows a different
  tumour part (T1c→enhancing, FLAIR/T2→edema); improves Dice.
- **CE + Dice loss** — Dice counters severe tumour/background imbalance.
- **Synthetic Rician degradation** — the *correct* MRI noise model, so the
  enhancer learns to remove real MRI noise, not generic noise.
- **Spine = unsupervised** — no labels exist, no external data allowed →
  supervised would be dishonest; self/unsupervised is what the brief encourages.
- **Detection runs on the ORIGINAL scan**, not the smoothed one — so enhancement
  can never erase a tumour (verified: Dice 0.842→0.824, negligible).

---

### 10. RAPID Q&A (one-line answers)

- **Loss function?** Enhancement: L1 + SSIM. Segmentation: Cross-Entropy + Dice.
- **Architecture?** 2D U-Net, encoder-decoder + skip connections.
- **Why 4 channels in?** Stack T1/T1c/T2/FLAIR — each shows a different tumour part.
- **Optimizer?** Adam, lr 1e-3, with AMP mixed precision.
- **Dataset + size?** BraTS2020, 4.47 GB, 369 cases; we used 126.
- **Dice score?** Mean tumour 0.73, enhancing 0.80.
- **Enhancement metrics?** PSNR 30.3, SSIM 0.965.
- **Why 2D not 3D?** 3D needs 16 GB+ GPU; 2D fits our 6 GB.
- **Preprocessing model?** None — classical CLAHE + histogram equalization.
- **Spine segmentation?** Unsupervised SLIC-superpixel clustering (no labels allowed),
  plus **spinal-canal delineation with a width profile**.
- **How do you handle stenosis without labels?** Stenosis is by definition canal
  narrowing, so we *measure* canal width instead of predicting a label: segment the
  CSF column, find its axis by PCA, sample width perpendicular to it, and report the
  narrowing ratio (narrowest ÷ typical width of that same canal, so patient size and
  resolution cancel out). Canal detected on **91/92** validation slices. Pathological
  canals trend narrower (0.485 vs 0.557, AUC 0.69) but at 10 vs 9 patients this is
  **not significant** (p = 0.089) — we report the measurement and the trend, not a
  diagnosis.
- **Do your spine models need annotations?** **No — none of them.** Enhancement is
  self-supervised (degrade the scan, restore it to itself), ROI is unsupervised
  clustering, lesion localisation is an autoencoder trained on healthy scans only.
- **Did you convert the .nii files?** No. Every volume is read directly from
  .nii/.nii.gz with nibabel. PNG/JSON are outputs only, never pipeline inputs.
- **How many samples train/test/val?** BraTS 127 cases available — segmentation
  16 train / 4 val, enhancement 32 / 8, plus 3-fold CV over 21. Offline data:
  **20 train / 20 test** (5+5 per each of the 4 groups). Full per-case ID
  enumeration in `stats/splits_report.txt`.
- **What do the labels mean?** BraTS: 1 = necrotic/dead core, 2 = edema
  (swelling), 3 (raw 4) = enhancing/active tumour. Distribution: background
  **99.03 %**, edema 0.71 %, enhancing 0.17 %, necrotic 0.10 % → why Dice loss.
- **Does one model fit all sub-modalities?** No — we proved it. Modality-specific
  spine models beat the pooled model on **3/3** sequences (T1 0.598→0.827,
  T2 0.594→0.802, STIR 0.540→0.714 SSIM, same test slices).
- **How do you find the spine lesion without labels?** We tried exactly that — an
  autoencoder trained on **healthy spines only**, treating reconstruction error as
  pathology. **We validated it and it failed: AUC 0.27**, worse than chance, healthy
  spines scoring higher than diseased ones. We withdrew the claim and report the
  negative result. Spine deliverables are restoration + unsupervised region
  segmentation.
- **Why no skip connections in that autoencoder?** Skips would copy the input
  (lesion included) straight to the output and erase the anomaly signal.
- **Does it diagnose the spine condition?** No, and it does not even flag one — the
  detector failed validation, so the map is shown purely as a visualisation.
- **How avoid erasing tumour?** Detect on the original scan; verified enhancement
  barely changes Dice.
- **How prevent false positives?** Non-MRI images rejected (dark-corner check);
  minimum tumour-size guard.
- **Overfitting?** No — validation tracks training, gap ≈ 0, converged ~epoch 25.
- **Metrics for segmentation?** Dice, Jaccard, Hausdorff, ASD, sensitivity,
  specificity, precision, F1, Relative Volume Error.

---

### 11. THE SPINE TRACK — the complete story

The brain track had ground truth (BraTS) and so it is a conventional supervised
story. **The spine track is the interesting one**, because it had no labels at
all, and most of the engineering effort went into working out what is and is not
possible in that situation. Judges will probe here — this is the full account.

#### 11.1 What the spine data actually is

- **20 cases total**: 10 normal (SP1–SP10), 10 pathological (SP11+).
- Sagittal MRI, sequences **T1 / T2 / STIR**.
- **No annotations of any kind.** No masks, no boxes, no labels.
- Highly heterogeneous geometry (voxel 0.25–1.3 mm, slice thickness 3–13 mm).
- Split 5 train / 5 test within each group, per the coordinator's instruction.

#### 11.2 The seven things we built for spine, in order

| # | Method | Needs labels? | Outcome |
|---|---|---|---|
| 1 | CLAHE enhancement (classical) | no | baseline, works |
| 2 | **Self-supervised U-Net enhancement (ours)** | no | **beats CLAHE on 3/3 sequences** |
| 3 | k-means / SLIC ROI clustering | no | works, but groups brightness only |
| 4 | **Self-supervised CNN segmentation (ours)** | no | **best annotation-free method, measured** |
| 5 | Autoencoder anomaly detection | no | **FAILED validation — withdrawn** |
| 6 | Canal-width morphometry (ours) | no | works, 91/92 slices |
| 7 | SPINEPS per-vertebra instances | pretrained | 17 vertebrae, 13 structures |

#### 11.3 The self-supervised CNN — how something with no labels trains at all

This is the question most likely to be asked, because it sounds impossible.

It is **differentiable feature clustering** (Kanezaki, ICASSP 2018). A small CNN
is trained **on the single scan in front of it**, from scratch, for ~120
iterations. Its supervision comes from three constraints, none of which need a
human label:

1. **Commit** — each pixel's feature vector is pushed toward its own argmax
   class. The network trains on its own current best guess, which sharpens fuzzy
   assignments into definite ones.
2. **Continuity** — neighbouring pixels are pushed to agree, so regions come out
   spatially coherent instead of speckled.
3. **Balance** — an entropy term on the mean class distribution stops every
   pixel collapsing into one class. Without it the whole image becomes one
   region (we hit this).

The number of structures it finds is **emergent, not set by us**: we offer 12
candidate classes and it settled on **9–10** on the test slice.

**Two real bugs, both measured, both worth telling:**

- *Collapse to 1 class.* We first masked the background by zeroing the
  **features**. That made every background pixel argmax to the same class, and
  the cross-entropy term then dragged the entire image into it. Fix: mask the
  **loss**, not the features.
- *Collapse to 2 classes.* Applying the superpixel prior every training
  iteration compounded with the cross-entropy feedback. **Measured: 2 classes
  with the prior in-loop, 11 without.** Fix: apply the prior once, after
  training.

#### 11.4 What we ditched, and why (be ready for this)

| Ditched | Why |
|---|---|
| **Supervised spine segmentation** | No labels exist and no external data was permitted. Any "trained" claim would be fabricated. |
| **Autoencoder anomaly detection** | Validated at **AUC 0.266 — worse than chance**; normal spines scored *higher* (0.0199) than pathological (0.0167). We tested five alternative scoring statistics (mean 0.304, max 0.500, p99 0.388, p95 0.312, top-1% 0.413) — all at or below chance, so it was not a scoring artefact. The verdict box was removed; the map remains only as a labelled visualisation. |
| **Vertebra instance segmentation, our own** | **Four** attempts: component uniformity/linearity scoring; canal-proximity prior; periodicity + autocorrelation; tissue-constrained band. All four locked onto soft tissue rather than bone. Not shipped. |
| **3D volumetric models** | Need 16 GB+ VRAM; we have 6 GB. 2D slice-wise is the documented workaround. |
| **Diagnosis / severity claims** | We measure canal width; we do not output "stenosis". The statistics do not support a diagnostic claim (p = 0.089). |

The pattern is deliberate: **we validated before shipping, and we withdrew two
things that failed.** That is a stronger position than claiming four working
features, because a judge can break a false claim in one question.

---

### 12. SPINEPS — the pretrained model (what, why, trained on what)

#### 12.1 What it is

**SPINEPS** — Möller et al., *"SPINEPS: automatic whole spine segmentation of
T2-weighted MR images using a two-phase approach to multi-class semantic and
instance segmentation"*, **European Radiology (2025)**. Apache-2.0 licence.
Built on **nnU-Net**. Runs in its own Python 3.11 conda environment.

**Two phases:**
1. **Semantic** — labels 14 spinal structure *types* (body, disc, canal, cord,
   arch, processes). Answers "what kind of tissue is this pixel?"
2. **Instance** — converts that into *individually numbered* vertebrae.
   Answers "which bone is this?"

#### 12.2 What it was trained on

- The public **SPIDER** dataset (annotated lumbar spine MRI).
- The **German National Cohort (NAKO)** — a large population imaging study.
- Roughly **1,600+ subjects**, with expert annotations.

That external annotated data is exactly what we do not have and are not
permitted to collect — which is the entire argument for using it.

#### 12.3 Its published accuracy (theirs, not ours)

| Structure | Dice |
|---|---|
| Vertebrae | **0.920** |
| Intervertebral discs | **0.967** |
| Spinal canal | **0.958** |

**Say clearly: these are their numbers on their test set. We claim none of them.**

#### 12.4 What it produced on OUR data (case SP11, sagittal T2w 512×512×12)

| Phase | Result |
|---|---|
| Semantic | **13 structures** — labels 41–49, 60, 61, 62, 100 |
| Instance | **17 individually numbered vertebrae** |
| Runtime | **401 s** total, instance phase on CPU (~25 s per vertebra) |

#### 12.5 What the numbers on the spine actually MEAN

Two different numbering systems appear, and confusing them is an easy trap:

**A. Instance IDs (1, 2, 3 … 17)** — the numbers drawn on each coloured
vertebra. They mean *"this is a distinct bone, separate from its neighbour."*
They are an **ordering down the spine**, not a diagnosis, not a severity score,
and not an anatomical name. ID 5 is simply the fifth vertebra the model
separated.

**B. Semantic label values** — the mask's pixel values, which are structure
*types* from the TPTBox convention:

| Value | Structure |
|---|---|
| 41 | Arcus vertebrae (vertebral arch) |
| 42 | Spinous process |
| 43 / 44 | Costal process left / right |
| 45 / 46 | Superior articular process left / right |
| 47 / 48 | Inferior articular process left / right |
| **49** | **Vertebral body (corpus)** |
| **60** | **Spinal cord** |
| **61** | **Spinal canal** |
| 62 | Endplate |
| **100** | **Intervertebral disc** |

So a pixel valued 100 is disc tissue; a vertebra tagged instance 7 is the
seventh bone down. Different questions, different numbers.

#### 12.6 Why using it is defensible

1. **The required output is supervised by nature.** Naming a herniated disc
   requires having seen examples labelled "herniated disc". With 20 unlabelled
   cases and no external data, *no model we train can produce it.* That is a
   property of the problem, not a failure of effort.
2. **We proved the alternatives fall short** — four annotation-free methods,
   measured, one reported as an outright failure (11.4, 13).
3. **We supply it no annotations and do not train it**, which matches the
   brief's "no annotations for model training".
4. **It is auditable** — open source, peer-reviewed, published weights and
   accuracy. Anyone can re-run it.
5. **Provenance is labelled everywhere it appears** — demo, figures, report.

#### 12.7 The two engineering problems it caused (good story, tells well)

- **CUDA OOM on the instance phase.** Failed at **12.44 GiB on a 6 GiB card**.
  Our first hypothesis — that the scan's 0.44 mm in-plane resolution was to
  blame — was **wrong**: downsampling 3.6x produced a byte-identical failure.
  The log revealed why: the instance phase does not read the input scan, it
  reads the **semantic mask SPINEPS just wrote**, which is stored in SPINEPS's
  own internal 0.75 mm space (512x512x53) regardless of input. Fixed by running
  that phase on CPU.
- **Misaligned overlay.** SPINEPS **resamples and reorients** to a canonical
  axis order, so a mask cannot be matched to the scan by array index. Fixed with
  `spineps_runner.mask_in_scan_space()`, which composes the two affines and
  resamples nearest-neighbour (interpolating integer labels would invent classes
  that do not exist).

---

### 13. OURS vs SPINEPS — the measured comparison

`src/spine_vs_spineps.py` produces `results/spine_vs_spineps.json` and
`outputs/demo/spine_vs_spineps.png`.

**The idea:** SPINEPS can name structures we cannot, so use it as a *reference
standard* and measure how much our annotation-free methods recover without ever
seeing a label. This converts "we used a pretrained model" into "here is exactly
what our own work achieves, and exactly what it cannot."

**Dice (case SP11, slice 5; CNN = mean ± sd over 3 runs):**

| Structure | k-means | SLIC | **Self-sup. CNN (ours)** |
|---|---|---|---|
| Vertebral bodies | 0.263 | 0.204 | **0.257 ± 0.018** |
| Intervertebral discs | 0.047 | 0.047 | **0.090 ± 0.017** |
| Spinal canal + cord | 0.304 | 0.302 | **0.380 ± 0.041** |
| Posterior elements | 0.071 | 0.097 | **0.169 ± 0.032** |

**Precision** (does the region stop at the structure's edge?):

| Structure | k-means | SLIC | **Ours** |
|---|---|---|---|
| Vertebral bodies | 0.157 | 0.121 | **0.191** |
| Discs | 0.024 | 0.024 | **0.050** |
| Canal + cord | 0.194 | 0.190 | **0.310** |
| Posterior elements | 0.038 | 0.052 | **0.116** |

**The three claims this supports:**
1. Our CNN has the **highest precision on all four structures**, and the best
   overlap on three of four (vertebral bodies is a tie within uncertainty).
2. Classical clustering shows **high recall, very low precision** (k-means
   recall 0.81 on bodies but precision 0.16) — it *finds* structures but its
   clusters flood across the image, exactly the "groups brightness" failure.
3. Even our best is **Dice 0.38** against SPINEPS's published **0.92**, and ours
   numbers **zero** vertebrae. That gap is the justification, quantified.

#### 13.1 Three methodological honesty points (judges may probe these)

- **The Dice values are oracle-assisted upper bounds.** Unsupervised methods
  return anonymous cluster indices; to score them at all, the reference must
  pick which cluster to compare. So the number answers *"was this structure
  carved out as a distinct region?"* — **not** *"can the method name it?"*
  Naming is precisely the supervised step our data cannot provide.
- **We checked whether the metric was unfair** to methods with more clusters
  (ours makes 9–10, k-means 4, so no single cluster can cover a whole
  structure). We added a greedy **best-union** metric to test exactly that. It
  moved the result by 0.002 (0.226 to 0.228) — so the concern was **measured and
  rejected**, not assumed away.
- **The CNN is stochastic.** It is seeded, but cuDNN chooses nondeterministic
  kernels, so runs differ (measured: posterior-element Dice 0.215 vs 0.143 on
  two runs of identical code). Reporting one number would not be reproducible,
  so all CNN figures are **mean ± sd over 3 runs**.

---

### 14. WHAT COULD BE IMPROVED (asked in almost every viva)

Have a real answer here — "nothing" reads as not understanding the work.

**Spine, biggest wins first:**
1. **Label a small subset and fine-tune.** Even 20–30 annotated slices would
   allow a supervised head on top of our features, turning anonymous regions
   into named structures. This is the single highest-value next step.
2. **3D instead of 2D.** We segment slice by slice; vertebrae are 3D objects and
   through-plane context would improve boundaries. Blocked by 6 GB VRAM.
3. **Distil SPINEPS output into pseudo-labels** and train a small model on them
   — gets instance numbering without shipping SPINEPS at inference time.
4. **More cases.** 20 patients is small; the canal-width trend (0.485 vs 0.557,
   AUC 0.69) points the right way but is **not significant (p = 0.089)**. Around
   40 per group would likely settle it.

**Brain:**
5. **3D or 2.5D segmentation** — adjacent-slice context typically adds several
   Dice points on BraTS.
6. **Ensembling** across seeds/folds — a reliable small gain we did not spend
   time on.
7. **Test-time augmentation** (flips) — cheap, usually worth about 1 Dice point.
8. **Boundary-aware loss** (add a surface term) — our Hausdorff is the weakest
   metric relative to Dice, which points at boundary quality.

**Engineering:**
9. **Determinism** — enable deterministic cuDNN kernels so results are exactly
   reproducible rather than mean ± sd.
10. **SPINEPS on GPU** would need more than 6 GB, or patched sliding-window
    inference in its instance phase.

---

### 15. RAPID Q&A — spine and SPINEPS

- **Did you train SPINEPS?** No. We give it no annotations and do not train it.
  We run published weights and label its provenance everywhere.
- **What was it trained on?** SPIDER + the German National Cohort, about 1,600+
  annotated subjects. That external labelled data is what we lack.
- **Isn't that against the rules?** The organisers approved a public pretrained
  model and asked for justification — that document is
  `PRETRAINED_MODEL_JUSTIFICATION.md`. Our own annotation-free pipeline is shown
  *beside* it, never replaced by it.
- **What do the numbers 1–17 on the vertebrae mean?** Instance IDs — "this is a
  separate bone from the one above." Not a diagnosis, not severity, not an
  anatomical name.
- **What do 41–49, 60, 61, 62, 100 mean?** Structure *types*: 49 vertebral body,
  60 cord, 61 canal, 100 disc, 41–48 arch and processes.
- **How good is your own spine segmentation, honestly?** Best Dice **0.38**
  (canal) against SPINEPS as reference, and the highest precision of the three
  annotation-free methods on all four structures — but zero numbered vertebrae,
  because numbering needs labels.
- **Why is your Dice so low?** Two reasons, both real: unsupervised clusters are
  anonymous and bleed past structure edges (precision 0.19–0.31), and the metric
  is an oracle-assisted upper bound on a method that was never given a target to
  fit.
- **Why did downsampling not fix the OOM?** Because the instance phase reads the
  saved semantic mask in SPINEPS's own 0.75 mm space, not our input — so the
  working volume never changed. We measured that, then moved to CPU.
- **Why not run SPINEPS live in the demo?** 401 s per scan on CPU. It is shown
  as a precomputed reference on a named case, explicitly labelled as not having
  been run on the uploaded scan.
- **Which parts are yours and which are not?** Ours: all brain enhancement and
  segmentation, all spine enhancement, the self-supervised CNN segmentation, the
  canal morphometry, and every validation. Not ours: SPINEPS instance/semantic
  masks, used for one output and labelled as pretrained everywhere.

---


# PART III — WHY WE USED A PRETRAINED MODEL

*Source: `docs/PRETRAINED_MODEL_JUSTIFICATION.md`*

## Why we use a pretrained model for spine ROI segmentation

*(Requested by the organisers alongside approval to use a publicly available
pretrained model. This states what we use, why the alternative was insufficient,
and exactly what is and is not our own work.)*

---

### 1. What we use

**SPINEPS** — Möller et al., *"SPINEPS: automatic whole spine segmentation of
T2-weighted MR images using a two-phase approach to multi-class semantic and
instance segmentation"*, **European Radiology (2025)**, Apache-2.0 licence.

It performs semantic segmentation of 14 spinal structures and, in a second
phase, converts that into an **instance** mask separating individual vertebrae
and intervertebral discs. Published accuracy: **Dice 0.92 vertebrae, 0.967
intervertebral discs, 0.958 spinal canal.** Its weights were trained on the
public SPIDER dataset and the German National Cohort.

We use it **only for spine ROI instance segmentation**. Every other result in
this project — all brain enhancement and segmentation, and all spine
enhancement — is our own model trained by us.

#### Exactly what "pretrained" means here — please read this line

**We did not obtain, download, view or train on any dataset other than the one
supplied to us.** What we use is the published **weights file** — the learned
numbers. Its authors trained those weights on their own data (SPIDER and the
German National Cohort), in their own published work, *before this competition
existed*. We load that file and run inference.

So when this document says the weights "encode external data", it means exactly
that and nothing more: the numbers in the file carry patterns their authors
learned. It does **not** mean we brought an outside dataset into our pipeline.
The distinction matters, because "we used a pretrained model" and "we used
external data for training" are different claims, and only the first is true of
us.

---

### 2. Why a pretrained model is necessary here

The task asks us to delineate **degenerative disc, disc herniation and spinal
stenosis** as regions of interest. These are *named clinical entities*, and
naming them is a supervised problem: a model can only learn to output the label
"herniated disc" if it has seen examples labelled that way.

The competition constraints make that impossible to learn from the supplied
data:

| Constraint | Consequence |
|---|---|
| The spine dataset carries **no annotations** | There is no target for a supervised model to fit. |
| **No external data** may be used for training | We cannot supply that target from elsewhere. |
| Only **20 spine cases** in total | Even with labels, this is far below what per-vertebra segmentation needs. |

So the required output cannot be produced by any model *we* train. That is not
a limitation of effort — it is a property of the problem as specified.

---

### 3. What we tried first, and what it showed

We did not reach for a pretrained model as a shortcut. We implemented and
**measured** four annotation-free approaches first, and the evidence for each
is in `results/`:

**a) Intensity clustering (k-means, then SLIC superpixels).** Works, and it is
what we shipped initially — but it groups *brightness*, not structure. It
cannot separate one vertebra from the next because adjacent vertebrae have the
same intensity.

**b) Self-supervised CNN segmentation** (differentiable feature clustering,
Kanezaki 2018). A genuine trained network requiring no annotations, optimised
on each scan using the image's own structure. This is a clear improvement — it
resolves the cord, the vertebral chain and soft tissue as coherent regions
(see `outputs/demo/spine_method_comparison.png`). But it produces *semantic*
regions, not numbered per-vertebra *instances*.

**c) Autoencoder anomaly detection**, trained on healthy spines only. We
validated it rather than assuming it worked, and **it failed**: AUC 0.27 on
held-out cases — worse than chance, with normal spines scoring higher than
pathological ones (`results/anomaly_validation.json`). We withdrew the claim.

**d) Periodicity-based vertebra detection**, using the fact that vertebrae
repeat at regular spacing along the canal. The canal detector itself is
reliable (91/92 slices), but the vertebra step kept locking onto non-spinal
tissue. Not shipped.

The conclusion is evidence-based, not assumed: **annotation-free methods can
recover spinal structure, but not per-vertebra instances.**

---

### 4. Why this is scientifically appropriate, not a shortcut

Using a validated pretrained model for a task that cannot be learned from the
available data is standard practice in medical imaging, for three reasons:

1. **It is the clinically responsible choice.** A per-vertebra segmentation
   invented by an under-constrained model would be confidently wrong. A model
   validated on 1,600+ subjects and published in a peer-reviewed radiology
   journal has known, quantified accuracy.
2. **It requires no annotation effort from us**, which is precisely what the
   revised brief asks for — *"AI models which do not require any annotations
   for model training"*. We supply SPINEPS no labels and do not train it.
3. **It is reproducible and auditable.** Open source, Apache-2.0, published
   weights, published accuracy. Anyone can re-run our result.

---

### 5. What it actually produced on our data

Run on spine case **SP11** (sagittal T2w, 512x512x12):

| Phase | Result |
|---|---|
| Semantic | **13 structures** — vertebral subregions (labels 41–49), intervertebral discs, spinal canal, spinal cord (60–62, 100) |
| Instance | **17 individually numbered vertebrae** |
| Runtime | 401 s total (instance phase on CPU — see `SPINEPS_SETUP.md`) |

Both masks are returned on SPINEPS's own resampled, reoriented grid, so they
are mapped back onto the original scan through the image affine
(`spineps_runner.mask_in_scan_space`) before display. Matching by array index
instead produces a visibly offset overlay — a rendering error rather than a
segmentation error, but one worth naming since the figures depend on it.

The comparison figure `outputs/demo/spine_method_comparison.png` shows all of
this on **one slice**: intensity clustering groups brightness and cannot
separate adjacent vertebrae; our self-supervised CNN resolves the cord,
vertebral chain and soft tissue as distinct structures without annotations;
SPINEPS adds the numbered per-vertebra instances that neither can reach.

---

### 6. How we report it — the provenance is never hidden

- SPINEPS output is **always labelled as a pretrained model with external
  training data**, in the demo, the figures and the report.
- Our own annotation-free methods are shown **beside** it in
  `spine_method_comparison.png`, so a reader sees exactly how much of the
  achievable result our own work recovers under the constraint.
- We claim **no credit for SPINEPS's accuracy**. The 0.92 vertebra Dice is
  their published figure, not our contribution.
- Our contributions on spine remain: modality-specific self-supervised
  enhancement (measured to beat a pooled model on 3/3 sequences), the
  self-supervised CNN segmentation, the canal-width morphometry, and the
  validation work that determined which methods actually function.

---

### 7. Summary

> We use SPINEPS for per-vertebra spine instance segmentation because that
> output is supervised by nature and cannot be learned from an unlabelled
> 20-case dataset with no external data permitted. We established this by
> implementing and measuring four annotation-free alternatives first, and
> reporting honestly that they do not reach per-vertebra instances. SPINEPS is
> open-source, peer-reviewed and validated at Dice 0.92; we supply it no
> annotations, we do not train it, and we label its provenance everywhere it
> appears.

---


# PART IV — GLOSSARY: EVERY TERM

*Source: `docs/GLOSSARY.md`*

## Glossary — every term on the website & in the demo, in plain words

Keep this open on your phone during judging. Format: **TERM (full form)** —
what it means simply · *(technical note if a judge digs deeper)*.

---

### Scan types (the "sub-modalities")
- **MRI (Magnetic Resonance Imaging)** — a scan that uses a strong magnet + radio
  waves (no radiation) to image soft tissue.
- **NIfTI (.nii / .nii.gz)** — the standard file format for brain/spine MRI
  volumes. `.gz` just means zipped. *(Hospitals and research both use it.)*
- **T1 (T1-weighted)** — anatomy looks "natural"; fat is bright, fluid is dark.
  Good for structure.
- **T1c / T1ce (T1 contrast-enhanced)** — a T1 scan after injecting a contrast
  dye (gadolinium); makes active tumour light up bright.
- **T2 (T2-weighted)** — fluid is bright; good for spotting swelling/lesions.
- **FLAIR (FLuid-Attenuated Inversion Recovery)** — a T2 scan with the bright
  fluid "switched off", so lesions next to fluid stand out. Brain only.
- **STIR (Short-TI Inversion Recovery)** — fat-suppressed scan; makes swelling
  and bone-marrow problems obvious. Used for spine.

### The two "dirt" types we remove
- **Rician noise** — the specific grainy noise pattern that MRI magnitude images
  have (not ordinary "Gaussian" camera noise). We simulate the *correct* kind.
- **Bias field (RF-coil inhomogeneity)** — smooth uneven brightness across the
  image (bright centre, dark edges) from the scanner's antenna.

### Enhancement quality scores (higher = better unless noted)
- **PSNR (Peak Signal-to-Noise Ratio)** — how close to the clean image, in
  decibels (dB). Higher = cleaner. ~30 dB is very good.
- **SSIM (Structural Similarity Index)** — how similar the *structure* is,
  0 to 1. 1 = identical. **This is our headline number** (0.9+ = excellent).
- **MSE / RMSE (Mean / Root-Mean-Square Error)** — average pixel error.
  **Lower = better.**
- **UQI (Universal Quality Index)** — an older combined similarity score, 0–1.
- **FSIM (Feature Similarity)** — similarity focused on edges/features, 0–1.
- **GMSD (Gradient Magnitude Similarity Deviation)** — edge-based difference.
  **Lower = better.**
- **VIF (Visual Information Fidelity)** — how much visual information survived.
- **LPIPS (Learned Perceptual Image Patch Similarity)** — an AI-based
  "does it *look* the same to a human" score. **Lower = better** (0 = identical).
- **BRISQUE / NIQE / PIQE** — "no-reference" quality scores: they rate a single
  image's quality *without* needing a clean copy to compare to. **Lower = better.**
  *(We use these on the real hospital scans, where there's no clean original.)*
- **Entropy** — how much detail/information is in the image (in bits).

### Segmentation quality scores (Stage 4)
- **Dice (Dice Similarity Coefficient, DSC)** — how much the AI's region
  overlaps the doctor's, 0–1 (or %). **The main segmentation score.** 0.8 = great.
  *(On the demo we show it as "overlap with doctor".)*
- **Jaccard (IoU, Intersection-over-Union)** — same idea as Dice, slightly
  stricter. 0–1.
- **Hausdorff Distance (HD)** — the worst-case boundary error (how far off the
  edge is at its worst). **Lower = better.**
- **ASD (Average Surface Distance)** — average boundary error. **Lower = better.**
- **Sensitivity / Recall** — of the real tumour, how much did we catch.
- **Specificity** — of the healthy tissue, how much did we correctly leave out.
- **Precision** — of what we flagged, how much was truly tumour.

### What the tumour colours mean
- **Green = Edema** — swelling around the tumour.
- **Red = Enhancing tumour** — the active, growing part.
- **Blue = Necrotic / non-enhancing core** — the dead centre.

### Methods & model
- **CLAHE (Contrast-Limited Adaptive Histogram Equalization)** — the classic
  textbook contrast-boost method. Our baseline — we beat it.
- **Histogram Equalization (HE)** — spreads out brightness levels to boost
  contrast (CLAHE is a smarter, local version).
- **U-Net** — the AI network shape we use: an encoder that compresses the image
  and a decoder that rebuilds it, with "skip connections" that preserve fine
  detail. Standard for medical imaging.
- **Skip connections** — shortcuts that carry sharp detail from early layers to
  late layers so the output isn't blurry.
- **CNN (Convolutional Neural Network)** — the family of AI models U-Net belongs
  to; learns visual patterns from data.
- **Supervised vs Unsupervised** — supervised = trained with expert labels
  (our brain tumour model); unsupervised = no labels, finds structure itself
  (our spine ROI, because no labels exist and none are allowed).

### Data & training words
- **BraTS (Brain Tumour Segmentation challenge 2020)** — the standard,
  expert-labelled brain-tumour dataset we had to use for brain.
- **Ground truth** — the expert/doctor's correct answer we compare against.
- **Unseen / held-out** — data the model never trained on (proves it
  generalises, not memorises).
- **Epoch** — one full pass through the training data.
- **Loss** — the error the model minimises while learning; lower = learning.
- **Convergence** — the point where training stops improving.
- **Overfitting** — memorising training data instead of learning general rules;
  we check the "overfitting gap" (ours is ~0 — healthy).
- **Augmentation** — making extra training examples by flipping/altering scans.

### Output & performance words
- **ROI (Region of Interest)** — the area we care about (tumour, disc, etc.).
- **COCO JSON** — a standard file format for storing segmentation masks, so
  other software can read our results.
- **Latency** — time to process one image (ours ~4 ms).
- **Throughput** — images processed per second (ours ~236/sec).
- **GPU** — the graphics chip that runs the AI fast. Ours is a 6 GB laptop GPU.
- **AMP (Automatic Mixed Precision)** — a trick to run the AI faster and use
  less memory without losing accuracy.
- **Parameters** — the AI's internal "knobs" (ours ~7.8 million; small = light).

### Anatomy words (healthy-tissue ROI)
- **CSF (Cerebrospinal Fluid)** — the fluid around the brain/spinal cord.
- **Grey matter / White matter** — the two main brain tissue types.
- **Intervertebral disc** — the cushion between spine bones (where herniation /
  degeneration happens).
- **Spinal stenosis** — narrowing of the spinal canal.

---

#### 3 lines that make you sound fluent
- "SSIM 0.9 means the cleaned scan is 90%+ structurally identical to the true one."
- "Dice is just the overlap between our outline and the doctor's — 0.8 is strong."
- "CLAHE is the textbook method; it boosts contrast but amplifies noise — we beat it."

---

### Spine + pretrained-model words (added with SPINEPS)

- **Semantic segmentation** — labels each pixel by *what kind* of thing it is
  ("this is disc tissue"). Two neighbouring vertebrae get the same label.
- **Instance segmentation** — separates *individual objects* ("this is vertebra
  5, that is vertebra 6"). Harder, and the thing that needs labels.
- **Instance ID** — the number drawn on each vertebra in our SPINEPS figure. It
  means "a separate bone from the one above", **not** a diagnosis or a severity
  score.
- **Pretrained model** — a model someone else already trained on their own
  labelled data; you run it as-is. We use one (SPINEPS) for exactly one output
  and label it everywhere.
- **SPINEPS** — published spine model (Möller et al., *European Radiology* 2025,
  Apache-2.0), trained on the SPIDER dataset + the German National Cohort,
  ~1,600+ annotated subjects. Published Dice 0.92 vertebrae / 0.967 discs /
  0.958 canal. Those are **their** numbers, not ours.
- **nnU-Net** — the self-configuring U-Net framework SPINEPS is built on; it
  picks its own preprocessing and patch sizes per dataset.
- **Reference standard** — a trusted output you measure yourself against when
  you have no ground truth. We use SPINEPS this way.
- **Self-supervised** — the model makes its own training target from the data
  (e.g. degrade a scan, learn to restore it). No human labels.
- **Differentiable feature clustering** — the annotation-free segmentation
  method we use (Kanezaki 2018): a small CNN trains on one image using
  commit + continuity + balance constraints instead of labels.
- **Precision vs recall** (for a segmented region) — *recall* = how much of the
  real structure you covered; *precision* = how much of what you marked was
  actually the structure. Clustering scores **high recall, low precision**: it
  finds the structure but spills far past its edges.
- **Oracle-assisted metric** — a score where the answer key picks which of your
  anonymous clusters to grade. It is an **upper bound**, and we say so.
- **Affine (in NIfTI)** — the matrix mapping voxel indices to real-world
  millimetres. Two images line up only through their affines, never by array
  index — which is why our first SPINEPS overlay was rotated.
- **Nearest-neighbour resampling** — resizing a label map by copying the closest
  label. Required for integer labels; smooth interpolation would invent classes
  that do not exist.
- **Vertebral body (corpus)** — the big drum-shaped front part of a vertebra
  (SPINEPS label 49).
- **Posterior elements** — the bony arch and processes behind the body (labels
  41–48): arch, spinous process, costal and articular processes.
- **Spinal canal vs spinal cord** — the canal (label 61) is the bony tunnel; the
  cord (label 60) is the nerve tissue inside it. Stenosis narrows the canal.
- **Endplate** — the cartilage surface between a vertebral body and its disc
  (label 62).

#### 3 more lines that make you sound fluent
- "Semantic says *what*, instance says *which one* — instance is the part that
  needs labels, which is why we use a pretrained model for it."
- "Our clustering has high recall but low precision: it finds the structure and
  then bleeds past its edge."
- "Masks only line up through the affine — SPINEPS reorients, so index-matching
  gives you a rotated overlay."

---


# PART V — WHAT EACH STAGE ASKED FOR

*Source: `docs/STAGES.md`*

## The 4 Stages — what each is, what WE did, how to explain it

The hackathon workflow has 4 stages. Phase-1 judging covers **Stage 1 + 2**
(and half of Stage 3). Here's the task for each, what we built, and a
one-line simple explanation you can say out loud.

---

### STAGE 1 — Dataset Exploration, Analysis & Preparation  ✅ DONE

**The official task:** Understand the MRI sub-modalities (T1, T2, FLAIR,
STIR…), analyse the dataset, measure image properties, divide patients by
sub-modality, and lay out the train/test/validation split.

**What we did:**
- Profiled **every** scan on 7 properties: Contrast, Complexity, Sharpness,
  Edge strength, Noise level, Mean, Deviation — for BraTS2020 **and** all four
  offline groups (`dataset_stats.py` → `stats/dataset_stats.csv`).
- Built a smart classifier that reads the messy scanner filenames and sorts
  each file into T1 / T1c / T2 / FLAIR / STIR, throwing out the out-of-scope
  sequences (DWI, ADC, SWI…). Every decision is logged (`stats/modality_audit.txt`).
- Enumerated the splits: BraTS 80/20; offline 5-train / 5-test per group (the
  coordinators' rule).

**Say it simply:**
> "First we studied the scans — measured how noisy, how sharp, how detailed
> each type is. That told us brain and spine scans are very different, so we
> handle them differently instead of one-size-fits-all."

---

### STAGE 2 — Pre-processing  ✅ DONE

**The official task:** Denoise, rescale, correct artifacts, de-emphasise
unimportant regions, apply basic enhancement (Histogram Equalization / CLAHE),
then re-measure quality with the same properties + IQA metrics (PSNR, SSIM…).

**What we did:**
- **Normalize** every scan to a common range (robust percentiles), **slice**
  the 3D volume into 2D images, **resize** to 224×224 (with clipping to kill
  interpolation ringing), skip empty background slices.
- **CLAHE** contrast enhancement as the classical baseline.
- Built **synthetic degradation pairs** (Rician noise + bias-field + blur) so
  we have clean-vs-dirty examples to train the enhancer on — no annotations
  needed.
- Augmentation (flips) and curated train/test sets.

**Say it simply:**
> "Then we cleaned and standardised every scan — same size, same brightness
> scale — and prepared matched clean/dirty pairs so the AI has something to
> learn from."

---

### STAGE 3 — MR Image Quality Enhancement  🔵 THIS IS THE MAIN AI (done + demoing)

**The official task:** Use a deep-learning / ensemble / transformer /
self-supervised model to further enhance quality; compare against published
methods with the full IQA metric suite; report training loss, validation loss,
learning curve, convergence epoch, overfitting gap; and a systematic
comparison on latency, throughput, memory, model complexity.

**What we did:**
- A **2D U-Net** that learns to reverse MRI degradation (noise + bias field).
  Loss = L1 + SSIM (keeps both brightness and structure faithful).
- Trained for brain (BraTS FLAIR) and all offline groups.
- Full IQA suite (PSNR, SSIM, FSIM, GMSD, VIF, LPIPS, BRISQUE, NIQE, PIQE,
  Entropy, MSE, RMSE, UQI) + a **3-way comparison** proving our AI beats the
  classical CLAHE method (e.g. brain SSIM 0.22 → **0.92**; CLAHE makes it
  *worse*).
- Loss curves, convergence epoch, overfitting gap, and a benchmark
  (7.77M params, 4 ms/image, 236 images/sec on a laptop GPU).

**Say it simply (this is the live demo):**
> "This is the core: give the AI a noisy scan, it gives back a clean one —
> without changing the anatomy. The classical method just brightens the noise;
> ours actually removes it. Watch — [run the website]."

---

### STAGE 4 — Region-of-Interest Segmentation  ✅ DONE (bonus for Phase 1)

**The official task:** On the enhanced scans, segment the region of interest —
tumour / edema / lesion in diseased brains, disc / herniation / stenosis in
spine; report Dice, Jaccard, Hausdorff, ASD, sensitivity/specificity,
Grad-CAM / attention.

**What we did:**
- **Brain (supervised, BraTS):** a 4-input U-Net (T1+T1c+T2+FLAIR) that finds
  the tumour sub-regions. Real metrics: **mean tumour Dice 0.73**, enhancing
  tumour **0.80**. On the demo it matches the doctor's label **~87%** on
  unseen cases.
- **Spine (unsupervised):** no ground-truth labels exist and no external data
  is allowed, so we use intensity clustering (k-means) — framed honestly as
  *exploratory* ROI detection, exactly what the problem statement suggests for
  a no-label situation.

**Say it simply:**
> "Finally, the cleaned scan feeds tumour detection — and here's the proof: our
> AI's tumour outline sits right on top of the doctor's, 87% overlap, on a scan
> it had never seen."

---

### The one-paragraph story (ties all four together)
> "A fast MRI is noisy and unevenly lit. **Stage 1** we measured exactly how bad.
> **Stage 2** we standardised and prepared the data. **Stage 3** — the heart —
> an AI cleans the scan without altering anatomy, beating the textbook method.
> **Stage 4** the clean scan feeds automatic tumour detection that matches the
> radiologist. All in under a second, on a laptop, using the files hospitals
> already have."

---


# PART VI — EVERY MEASURED NUMBER WE HOLD

Generated directly from `results/*.json`. If a judge asks for a figure
not in Parts I–V, it is almost certainly here. Nothing in this section
was retyped by hand, so it cannot disagree with the code.

## `anomaly_validation.json`

| Field | Value |
|---|---|
| `modality` | T2 |
| `normal.n` | 28 |
| `normal.mean` | 0.0199 |
| `normal.min` | 0.0137 |
| `normal.max` | 0.0314 |
| `pathological.n` | 27 |
| `pathological.mean` | 0.0167 |
| `pathological.min` | 0.0101 |
| `pathological.max` | 0.0234 |
| `auc` | 0.266 |
| `threshold_95th_pct_of_normal` | 0.025 |
| `pathological_detected_at_that_threshold` | 0.0 |
| `verdict` | NOT A VALID DETECTOR — the anomaly score does not separate pathological from ... |
| `alternative_scoring_statistics.why` | A focal lesion barely shifts a whole-image mean, so the mean might have been ... |
| `alternative_scoring_statistics.auc_by_statistic.mean` | 0.304 |
| `alternative_scoring_statistics.auc_by_statistic.max` | 0.5 |
| `alternative_scoring_statistics.auc_by_statistic.p99` | 0.388 |
| `alternative_scoring_statistics.auc_by_statistic.p95` | 0.312 |
| `alternative_scoring_statistics.auc_by_statistic.top_1_percent_mean` | 0.413 |
| `alternative_scoring_statistics.auc_by_statistic.fraction_above_0.5` | 0.407 |
| `alternative_scoring_statistics.conclusion` | Every statistic is at or below chance (0.5). The failure is not an artefact o... |

## `benchmark_results.json`

| Field | Value |
|---|---|
| `cuda.device_name` | NVIDIA GeForce RTX 4050 Laptop GPU |
| `cuda.EnhancementUNet.params` | 7765409 |
| `cuda.EnhancementUNet.model_size_mb` | 31.06 |
| `cuda.EnhancementUNet.batch_size` | 8 |
| `cuda.EnhancementUNet.latency_ms_per_image_mean` | 4.2353 |
| `cuda.EnhancementUNet.latency_ms_per_image_std` | 0.0488 |
| `cuda.EnhancementUNet.throughput_images_per_sec` | 236.1126 |
| `cuda.EnhancementUNet.peak_gpu_mem_mb` | 384.9 |
| `cuda.EnhancementUNet.runs` | 30 |
| `cuda.SegmentationUNet.params` | 7766372 |
| `cuda.SegmentationUNet.model_size_mb` | 31.07 |
| `cuda.SegmentationUNet.batch_size` | 8 |
| `cuda.SegmentationUNet.latency_ms_per_image_mean` | 4.2439 |
| `cuda.SegmentationUNet.latency_ms_per_image_std` | 0.0492 |
| `cuda.SegmentationUNet.throughput_images_per_sec` | 235.6304 |
| `cuda.SegmentationUNet.peak_gpu_mem_mb` | 389.8 |
| `cuda.SegmentationUNet.runs` | 30 |
| `cpu.device_name` | CPU |
| `cpu.EnhancementUNet.params` | 7765409 |
| `cpu.EnhancementUNet.model_size_mb` | 31.06 |
| `cpu.EnhancementUNet.batch_size` | 8 |
| `cpu.EnhancementUNet.latency_ms_per_image_mean` | 49.2512 |
| `cpu.EnhancementUNet.latency_ms_per_image_std` | 0.7806 |
| `cpu.EnhancementUNet.throughput_images_per_sec` | 20.3041 |
| `cpu.EnhancementUNet.peak_gpu_mem_mb` | None |
| `cpu.EnhancementUNet.runs` | 30 |
| `cpu.SegmentationUNet.params` | 7766372 |
| `cpu.SegmentationUNet.model_size_mb` | 31.07 |
| `cpu.SegmentationUNet.batch_size` | 8 |
| `cpu.SegmentationUNet.latency_ms_per_image_mean` | 54.2342 |
| `cpu.SegmentationUNet.latency_ms_per_image_std` | 5.7564 |
| `cpu.SegmentationUNet.throughput_images_per_sec` | 18.4386 |
| `cpu.SegmentationUNet.peak_gpu_mem_mb` | None |
| `cpu.SegmentationUNet.runs` | 30 |

## `benchmark_utilization.json`

| Field | Value |
|---|---|
| `device` | cuda |
| `EnhancementUNet.gpu_util_pct_mean` | 84.1 |
| `EnhancementUNet.gpu_util_pct_max` | 99 |
| `EnhancementUNet.gpu_mem_util_pct_mean` | 71.6 |
| `EnhancementUNet.cpu_util_pct_mean` | 16.4 |
| `EnhancementUNet.samples` | 27 |
| `SegmentationUNet.gpu_util_pct_mean` | 97.6 |
| `SegmentationUNet.gpu_util_pct_max` | 100 |
| `SegmentationUNet.gpu_mem_util_pct_mean` | 84.3 |
| `SegmentationUNet.cpu_util_pct_mean` | 14.2 |
| `SegmentationUNet.samples` | 27 |

## `brain_offline_coco.json`

| Field | Value |
|---|---|
| `info.description` | brain_offline ROI segmentation (MedhaDrishti hackathon) |
| `info.kind` | brain_seg |
| `images[0].id` | 1 |
| `images[0].file_name` | BRP10_z50.png |
| `images[0].width` | 224 |
| `images[0].height` | 224 |
| `images[1].id` | 2 |
| `images[1].file_name` | BRP10_z51.png |
| `images[1].width` | 224 |
| `images[1].height` | 224 |
| `images[2].id` | 3 |
| `images[2].file_name` | BRP10_z52.png |
| `images[2].width` | 224 |
| `images[2].height` | 224 |
| `images[3].id` | 4 |
| `images[3].file_name` | BRP1_z59.png |
| `images[3].width` | 224 |
| `images[3].height` | 224 |
| `annotations[0].id` | 1 |
| `annotations[0].image_id` | 1 |
| `annotations[0].category_id` | 1 |
| `annotations[0].segmentation.size` | [2 values] |
| `annotations[0].segmentation.counts` | olf01j6312O010Tl0KQQg0 |
| `annotations[0].area` | 24.0 |
| `annotations[0].bbox` | [4 values] |
| `annotations[0].iscrowd` | 0 |
| `annotations[1].id` | 2 |
| `annotations[1].image_id` | 1 |
| `annotations[1].category_id` | 2 |
| `annotations[1].segmentation.size` | [2 values] |
| `annotations[1].segmentation.counts` | gnb02m62L301N3N1N2N100O10O1O2N`IF\69cII]66cIL\63eIK]65cIJ^66eIIY66iIIW66jIJV6... |
| `annotations[1].area` | 497.0 |
| `annotations[1].bbox` | [4 values] |
| `annotations[1].iscrowd` | 0 |
| `annotations[2].id` | 3 |
| `annotations[2].image_id` | 1 |
| `annotations[2].category_id` | 3 |
| `annotations[2].segmentation.size` | [2 values] |
| `annotations[2].segmentation.counts` | i^f0241c63[ILc65]ILa60aIOM1a60dI2\6LfI3e6O1OCO>000000010ZI1Z6OcI2f6OSQg0 |
| `annotations[2].area` | 63.0 |
| `annotations[2].bbox` | [4 values] |
| `annotations[2].iscrowd` | 0 |
| `annotations[3].id` | 4 |
| `annotations[3].image_id` | 2 |
| `annotations[3].category_id` | 1 |
| `annotations[3].segmentation.size` | [2 values] |
| `annotations[3].segmentation.counts` | kSg02l64M101N100001O004L1NT_g0 |
| `annotations[3].area` | 72.0 |
| `annotations[3].bbox` | [4 values] |
| `annotations[3].iscrowd` | 0 |
| `categories[0].id` | 1 |
| `categories[0].name` | necrotic_non_enhancing_core |
| `categories[1].id` | 2 |
| `categories[1].name` | edema |
| `categories[2].id` | 3 |
| `categories[2].name` | enhancing_tumor |

## `cross_validation.json`

| Field | Value |
|---|---|
| `method` | 3-fold cross-validation, brain segmentation (21 cases, 25 epochs/fold) |
| `per_fold_mean_tumor_dice` | [3 values] |
| `cv_accuracy_mean_dice` | 0.5895 |
| `cv_accuracy_std_dice` | 0.0384 |
| `per_fold_per_class[0].necrotic_non_enhancing` | 0.4093 |
| `per_fold_per_class[0].edema` | 0.6617 |
| `per_fold_per_class[0].enhancing` | 0.7756 |
| `per_fold_per_class[1].necrotic_non_enhancing` | 0.4559 |
| `per_fold_per_class[1].edema` | 0.6824 |
| `per_fold_per_class[1].enhancing` | 0.7149 |
| `per_fold_per_class[2].necrotic_non_enhancing` | 0.2419 |
| `per_fold_per_class[2].edema` | 0.7355 |
| `per_fold_per_class[2].enhancing` | 0.628 |

## `enhancement_metrics.json`

| Field | Value |
|---|---|
| `mse` | 0.0023 |
| `rmse` | 0.0449 |
| `uqi` | 0.9778 |
| `psnr` | 27.5906 |
| `ssim` | 0.9421 |
| `fsim` | 0.9759 |
| `gmsd` | 0.0552 |
| `vif` | 0.3605 |
| `lpips` | 0.066 |

## `enhancement_metrics_brain_normal.json`

| Field | Value |
|---|---|
| `group` | brain_normal |
| `best_epoch` | 9 |
| `best_test_loss` | -0.0575 |
| `overfitting_gap` | 0.0041 |
| `history.train_loss` | [25 values] |
| `history.test_loss` | [25 values] |
| `three_way_iqa.input.mse` | 0.0063 |
| `three_way_iqa.input.rmse` | 0.0763 |
| `three_way_iqa.input.uqi` | 0.9152 |
| `three_way_iqa.input.psnr` | 22.7604 |
| `three_way_iqa.input.ssim` | 0.3486 |
| `three_way_iqa.input.fsim` | 0.6643 |
| `three_way_iqa.input.gmsd` | 0.1239 |
| `three_way_iqa.input.vif` | 0.4244 |
| `three_way_iqa.input.lpips` | 0.3872 |
| `three_way_iqa.clahe.mse` | 0.033 |
| `three_way_iqa.clahe.rmse` | 0.1769 |
| `three_way_iqa.clahe.uqi` | 0.7146 |
| `three_way_iqa.clahe.psnr` | 15.2926 |
| `three_way_iqa.clahe.ssim` | 0.2627 |
| `three_way_iqa.clahe.fsim` | 0.4773 |
| `three_way_iqa.clahe.gmsd` | 0.2418 |
| `three_way_iqa.clahe.vif` | 0.3326 |
| `three_way_iqa.clahe.lpips` | 0.6391 |
| `three_way_iqa.model.mse` | 0.0031 |
| `three_way_iqa.model.rmse` | 0.0531 |
| `three_way_iqa.model.uqi` | 0.968 |
| `three_way_iqa.model.psnr` | 25.9692 |
| `three_way_iqa.model.ssim` | 0.8652 |
| `three_way_iqa.model.fsim` | 0.9316 |
| `three_way_iqa.model.gmsd` | 0.081 |
| `three_way_iqa.model.vif` | 0.3878 |
| `three_way_iqa.model.lpips` | 0.0922 |
| `note` | input=degraded (no enhancement), clahe=classical baseline, model=trained U-Net |

## `enhancement_metrics_brain_pathological.json`

| Field | Value |
|---|---|
| `group` | brain_pathological |
| `best_epoch` | 23 |
| `best_test_loss` | -0.1907 |
| `overfitting_gap` | -0.0144 |
| `history.train_loss` | [25 values] |
| `history.test_loss` | [25 values] |
| `three_way_iqa.input.mse` | 0.006 |
| `three_way_iqa.input.rmse` | 0.0749 |
| `three_way_iqa.input.uqi` | 0.8848 |
| `three_way_iqa.input.psnr` | 22.8802 |
| `three_way_iqa.input.ssim` | 0.192 |
| `three_way_iqa.input.fsim` | 0.5242 |
| `three_way_iqa.input.gmsd` | 0.1224 |
| `three_way_iqa.input.vif` | 0.4599 |
| `three_way_iqa.input.lpips` | 0.4353 |
| `three_way_iqa.clahe.mse` | 0.0309 |
| `three_way_iqa.clahe.rmse` | 0.1698 |
| `three_way_iqa.clahe.uqi` | 0.6618 |
| `three_way_iqa.clahe.psnr` | 15.7244 |
| `three_way_iqa.clahe.ssim` | 0.1495 |
| `three_way_iqa.clahe.fsim` | 0.308 |
| `three_way_iqa.clahe.gmsd` | 0.2393 |
| `three_way_iqa.clahe.vif` | 0.3476 |
| `three_way_iqa.clahe.lpips` | 0.7577 |
| `three_way_iqa.model.mse` | 0.0019 |
| `three_way_iqa.model.rmse` | 0.0406 |
| `three_way_iqa.model.uqi` | 0.9814 |
| `three_way_iqa.model.psnr` | 28.3651 |
| `three_way_iqa.model.ssim` | 0.9625 |
| `three_way_iqa.model.fsim` | 0.9812 |
| `three_way_iqa.model.gmsd` | 0.047 |
| `three_way_iqa.model.vif` | 0.4471 |
| `three_way_iqa.model.lpips` | 0.0536 |
| `note` | input=degraded (no enhancement), clahe=classical baseline, model=trained U-Net |

## `enhancement_metrics_spine_normal.json`

| Field | Value |
|---|---|
| `group` | spine_normal |
| `best_epoch` | 5 |
| `best_test_loss` | 0.078 |
| `overfitting_gap` | -0.1505 |
| `history.train_loss` | [25 values] |
| `history.test_loss` | [25 values] |
| `three_way_iqa.input.mse` | 0.0065 |
| `three_way_iqa.input.rmse` | 0.0777 |
| `three_way_iqa.input.uqi` | 0.9056 |
| `three_way_iqa.input.psnr` | 22.5348 |
| `three_way_iqa.input.ssim` | 0.3998 |
| `three_way_iqa.input.fsim` | 0.7269 |
| `three_way_iqa.input.gmsd` | 0.1156 |
| `three_way_iqa.input.vif` | 0.4244 |
| `three_way_iqa.input.lpips` | 0.4335 |
| `three_way_iqa.clahe.mse` | 0.0371 |
| `three_way_iqa.clahe.rmse` | 0.1883 |
| `three_way_iqa.clahe.uqi` | 0.6762 |
| `three_way_iqa.clahe.psnr` | 14.7089 |
| `three_way_iqa.clahe.ssim` | 0.2804 |
| `three_way_iqa.clahe.fsim` | 0.5438 |
| `three_way_iqa.clahe.gmsd` | 0.2285 |
| `three_way_iqa.clahe.vif` | 0.3373 |
| `three_way_iqa.clahe.lpips` | 0.6563 |
| `three_way_iqa.model.mse` | 0.0065 |
| `three_way_iqa.model.rmse` | 0.078 |
| `three_way_iqa.model.uqi` | 0.9344 |
| `three_way_iqa.model.psnr` | 22.4035 |
| `three_way_iqa.model.ssim` | 0.8157 |
| `three_way_iqa.model.fsim` | 0.884 |
| `three_way_iqa.model.gmsd` | 0.1199 |
| `three_way_iqa.model.vif` | 0.3284 |
| `three_way_iqa.model.lpips` | 0.1529 |
| `note` | input=degraded (no enhancement), clahe=classical baseline, model=trained U-Net |

## `enhancement_metrics_spine_normal_STIR.json`

| Field | Value |
|---|---|
| `group` | spine_normal_STIR |
| `best_epoch` | 25 |
| `best_test_loss` | 0.1406 |
| `overfitting_gap` | -0.1892 |
| `history.train_loss` | [25 values] |
| `history.test_loss` | [25 values] |
| `three_way_iqa.input.mse` | 0.0248 |
| `three_way_iqa.input.rmse` | 0.1467 |
| `three_way_iqa.input.uqi` | 0.688 |
| `three_way_iqa.input.psnr` | 17.4735 |
| `three_way_iqa.input.ssim` | 0.2334 |
| `three_way_iqa.input.fsim` | 0.6047 |
| `three_way_iqa.input.gmsd` | 0.2125 |
| `three_way_iqa.input.vif` | 0.2233 |
| `three_way_iqa.input.lpips` | 0.7638 |
| `three_way_iqa.clahe.mse` | 0.1034 |
| `three_way_iqa.clahe.rmse` | 0.3107 |
| `three_way_iqa.clahe.uqi` | 0.3907 |
| `three_way_iqa.clahe.psnr` | 10.5108 |
| `three_way_iqa.clahe.ssim` | 0.155 |
| `three_way_iqa.clahe.fsim` | 0.4673 |
| `three_way_iqa.clahe.gmsd` | 0.2988 |
| `three_way_iqa.clahe.vif` | 0.189 |
| `three_way_iqa.clahe.lpips` | 0.9395 |
| `three_way_iqa.model.mse` | 0.0048 |
| `three_way_iqa.model.rmse` | 0.0675 |
| `three_way_iqa.model.uqi` | 0.9269 |
| `three_way_iqa.model.psnr` | 23.6271 |
| `three_way_iqa.model.ssim` | 0.7221 |
| `three_way_iqa.model.fsim` | 0.8459 |
| `three_way_iqa.model.gmsd` | 0.1359 |
| `three_way_iqa.model.vif` | 0.2322 |
| `three_way_iqa.model.lpips` | 0.2491 |
| `note` | input=degraded (no enhancement), clahe=classical baseline, model=trained U-Net |

## `enhancement_metrics_spine_normal_T1.json`

| Field | Value |
|---|---|
| `group` | spine_normal_T1 |
| `best_epoch` | 10 |
| `best_test_loss` | 0.0804 |
| `overfitting_gap` | -0.1191 |
| `history.train_loss` | [25 values] |
| `history.test_loss` | [25 values] |
| `three_way_iqa.input.mse` | 0.0228 |
| `three_way_iqa.input.rmse` | 0.1399 |
| `three_way_iqa.input.uqi` | 0.7746 |
| `three_way_iqa.input.psnr` | 17.9575 |
| `three_way_iqa.input.ssim` | 0.279 |
| `three_way_iqa.input.fsim` | 0.6075 |
| `three_way_iqa.input.gmsd` | 0.1948 |
| `three_way_iqa.input.vif` | 0.2626 |
| `three_way_iqa.input.lpips` | 0.7302 |
| `three_way_iqa.clahe.mse` | 0.0796 |
| `three_way_iqa.clahe.rmse` | 0.269 |
| `three_way_iqa.clahe.uqi` | 0.5342 |
| `three_way_iqa.clahe.psnr` | 11.8814 |
| `three_way_iqa.clahe.ssim` | 0.2005 |
| `three_way_iqa.clahe.fsim` | 0.4807 |
| `three_way_iqa.clahe.gmsd` | 0.2777 |
| `three_way_iqa.clahe.vif` | 0.2208 |
| `three_way_iqa.clahe.lpips` | 0.8748 |
| `three_way_iqa.model.mse` | 0.0116 |
| `three_way_iqa.model.rmse` | 0.1035 |
| `three_way_iqa.model.uqi` | 0.8521 |
| `three_way_iqa.model.psnr` | 20.0376 |
| `three_way_iqa.model.ssim` | 0.4915 |
| `three_way_iqa.model.fsim` | 0.815 |
| `three_way_iqa.model.gmsd` | 0.1587 |
| `three_way_iqa.model.vif` | 0.2454 |
| `three_way_iqa.model.lpips` | 0.3156 |
| `note` | input=degraded (no enhancement), clahe=classical baseline, model=trained U-Net |

## `enhancement_metrics_spine_normal_T2.json`

| Field | Value |
|---|---|
| `group` | spine_normal_T2 |
| `best_epoch` | 4 |
| `best_test_loss` | 0.2084 |
| `overfitting_gap` | -0.1545 |
| `history.train_loss` | [25 values] |
| `history.test_loss` | [25 values] |
| `three_way_iqa.input.mse` | 0.0233 |
| `three_way_iqa.input.rmse` | 0.1398 |
| `three_way_iqa.input.uqi` | 0.7538 |
| `three_way_iqa.input.psnr` | 18.1522 |
| `three_way_iqa.input.ssim` | 0.3157 |
| `three_way_iqa.input.fsim` | 0.6227 |
| `three_way_iqa.input.gmsd` | 0.1925 |
| `three_way_iqa.input.vif` | 0.3111 |
| `three_way_iqa.input.lpips` | 0.615 |
| `three_way_iqa.clahe.mse` | 0.0878 |
| `three_way_iqa.clahe.rmse` | 0.2794 |
| `three_way_iqa.clahe.uqi` | 0.5084 |
| `three_way_iqa.clahe.psnr` | 11.6866 |
| `three_way_iqa.clahe.ssim` | 0.2402 |
| `three_way_iqa.clahe.fsim` | 0.4968 |
| `three_way_iqa.clahe.gmsd` | 0.2822 |
| `three_way_iqa.clahe.vif` | 0.2538 |
| `three_way_iqa.clahe.lpips` | 0.786 |
| `three_way_iqa.model.mse` | 0.0275 |
| `three_way_iqa.model.rmse` | 0.1589 |
| `three_way_iqa.model.uqi` | 0.7212 |
| `three_way_iqa.model.psnr` | 16.3691 |
| `three_way_iqa.model.ssim` | 0.3654 |
| `three_way_iqa.model.fsim` | 0.7028 |
| `three_way_iqa.model.gmsd` | 0.2229 |
| `three_way_iqa.model.vif` | 0.2673 |
| `three_way_iqa.model.lpips` | 0.3945 |
| `note` | input=degraded (no enhancement), clahe=classical baseline, model=trained U-Net |

## `enhancement_metrics_spine_pathological.json`

| Field | Value |
|---|---|
| `group` | spine_pathological |
| `best_epoch` | 6 |
| `best_test_loss` | 0.0904 |
| `overfitting_gap` | -0.0813 |
| `history.train_loss` | [25 values] |
| `history.test_loss` | [25 values] |
| `three_way_iqa.input.mse` | 0.0066 |
| `three_way_iqa.input.rmse` | 0.0787 |
| `three_way_iqa.input.uqi` | 0.9467 |
| `three_way_iqa.input.psnr` | 22.408 |
| `three_way_iqa.input.ssim` | 0.5219 |
| `three_way_iqa.input.fsim` | 0.8299 |
| `three_way_iqa.input.gmsd` | 0.1033 |
| `three_way_iqa.input.vif` | 0.4138 |
| `three_way_iqa.input.lpips` | 0.3561 |
| `three_way_iqa.clahe.mse` | 0.0284 |
| `three_way_iqa.clahe.rmse` | 0.1657 |
| `three_way_iqa.clahe.uqi` | 0.8229 |
| `three_way_iqa.clahe.psnr` | 15.7737 |
| `three_way_iqa.clahe.ssim` | 0.3939 |
| `three_way_iqa.clahe.fsim` | 0.701 |
| `three_way_iqa.clahe.gmsd` | 0.2031 |
| `three_way_iqa.clahe.vif` | 0.3373 |
| `three_way_iqa.clahe.lpips` | 0.5173 |
| `three_way_iqa.model.mse` | 0.0092 |
| `three_way_iqa.model.rmse` | 0.0941 |
| `three_way_iqa.model.uqi` | 0.9387 |
| `three_way_iqa.model.psnr` | 20.6853 |
| `three_way_iqa.model.ssim` | 0.7465 |
| `three_way_iqa.model.fsim` | 0.8454 |
| `three_way_iqa.model.gmsd` | 0.1453 |
| `three_way_iqa.model.vif` | 0.2953 |
| `three_way_iqa.model.lpips` | 0.2421 |
| `note` | input=degraded (no enhancement), clahe=classical baseline, model=trained U-Net |

## `modality_comparison.json`

| Field | Value |
|---|---|
| `T1.input.psnr` | 18.11 |
| `T1.input.ssim` | 0.299 |
| `T1.clahe.psnr` | 12.05 |
| `T1.clahe.ssim` | 0.22 |
| `T1.pooled.psnr` | 21.03 |
| `T1.pooled.ssim` | 0.598 |
| `T1.specific.psnr` | 24.27 |
| `T1.specific.ssim` | 0.827 |
| `T1.n_slices` | 60 |
| `T2.input.psnr` | 18.04 |
| `T2.input.ssim` | 0.28 |
| `T2.clahe.psnr` | 11.44 |
| `T2.clahe.ssim` | 0.212 |
| `T2.pooled.psnr` | 21.02 |
| `T2.pooled.ssim` | 0.594 |
| `T2.specific.psnr` | 23.69 |
| `T2.specific.ssim` | 0.802 |
| `T2.n_slices` | 60 |
| `STIR.input.psnr` | 18.84 |
| `STIR.input.ssim` | 0.279 |
| `STIR.clahe.psnr` | 11.27 |
| `STIR.clahe.ssim` | 0.174 |
| `STIR.pooled.psnr` | 21.6 |
| `STIR.pooled.ssim` | 0.54 |
| `STIR.specific.psnr` | 24.26 |
| `STIR.specific.ssim` | 0.714 |
| `STIR.n_slices` | 60 |
| `_verdict` | modality-specific wins on 3/3 modalities (SSIM, same test slices) |
| `_note` | Both models evaluated on the SAME held-out per-modality test slices with iden... |

## `paper_comparison.json`

| Field | Value |
|---|---|
| `Degraded input.psnr` | 18.0538 |
| `Degraded input.ssim` | 0.1964 |
| `Degraded input.fsim` | 0.4303 |
| `Degraded input.vif` | 0.302 |
| `HE.psnr` | 8.0541 |
| `HE.ssim` | 0.1492 |
| `HE.fsim` | 0.2107 |
| `HE.vif` | 0.1923 |
| `AHE.psnr` | 6.3529 |
| `AHE.ssim` | 0.1333 |
| `AHE.fsim` | 0.1874 |
| `AHE.vif` | 0.1985 |
| `CLAHE.psnr` | 11.8362 |
| `CLAHE.ssim` | 0.1556 |
| `CLAHE.fsim` | 0.28 |
| `CLAHE.vif` | 0.2478 |
| `Ours (2D U-Net).psnr` | 27.0794 |
| `Ours (2D U-Net).ssim` | 0.9033 |
| `Ours (2D U-Net).fsim` | 0.9594 |
| `Ours (2D U-Net).vif` | 0.3606 |
| `_improvement_over_degraded.psnr_db_gain` | 9.03 |
| `_improvement_over_degraded.ssim_gain` | 0.707 |
| `_methodology_note` | The reference papers evaluate on different datasets, so their absolute PSNR/S... |
| `_reference_papers[0]` | Ravi Kumar & A.K. Bhandari, Computers in Biology and Medicine 146 (2022) 1056... |
| `_reference_papers[1]` | Huayu Fan, Xiangyang Cao et al., Current Medical Imaging, 2024 (spine SPAIR). |

## `segmentation_full_metrics.json`

| Field | Value |
|---|---|
| `necrotic_non_enhancing.dice` | 0.6717 |
| `necrotic_non_enhancing.jaccard` | 0.5057 |
| `necrotic_non_enhancing.accuracy` | 0.9996 |
| `necrotic_non_enhancing.sensitivity_recall` | 0.7436 |
| `necrotic_non_enhancing.specificity` | 0.9998 |
| `necrotic_non_enhancing.precision` | 0.6125 |
| `necrotic_non_enhancing.f1_score` | 0.6717 |
| `necrotic_non_enhancing.hausdorff_distance_mean` | 8.592 |
| `necrotic_non_enhancing.average_surface_distance_mean` | 1.68 |
| `necrotic_non_enhancing.relative_volume_error` | 0.214 |
| `edema.dice` | 0.768 |
| `edema.jaccard` | 0.6234 |
| `edema.accuracy` | 0.9966 |
| `edema.sensitivity_recall` | 0.6541 |
| `edema.specificity` | 0.9996 |
| `edema.precision` | 0.93 |
| `edema.f1_score` | 0.768 |
| `edema.hausdorff_distance_mean` | 19.706 |
| `edema.average_surface_distance_mean` | 1.589 |
| `edema.relative_volume_error` | 0.2967 |
| `enhancing.dice` | 0.8395 |
| `enhancing.jaccard` | 0.7234 |
| `enhancing.accuracy` | 0.9995 |
| `enhancing.sensitivity_recall` | 0.7831 |
| `enhancing.specificity` | 0.9998 |
| `enhancing.precision` | 0.9046 |
| `enhancing.f1_score` | 0.8395 |
| `enhancing.hausdorff_distance_mean` | 6.935 |
| `enhancing.average_surface_distance_mean` | 0.86 |
| `enhancing.relative_volume_error` | 0.1342 |
| `_mean_tumor_dice` | 0.7597 |
| `_protocol` | dataset-level accumulation of TP/TN/FP/FN over all held-out validation slices... |

## `segmentation_metrics.json`

| Field | Value |
|---|---|
| `best_epoch_convergence` | 25 |
| `best_val_loss` | 0.2393 |
| `overfitting_gap` | -0.0845 |
| `loss_type` | ce_dice |
| `final_val_metrics.necrotic_non_enhancing.dice` | 0.6854 |
| `final_val_metrics.necrotic_non_enhancing.jaccard` | 0.5213 |
| `final_val_metrics.necrotic_non_enhancing.hausdorff_distance_mean` | 9.4284 |
| `final_val_metrics.necrotic_non_enhancing.average_surface_distance_mean` | 0.7167 |
| `final_val_metrics.necrotic_non_enhancing.val_slices_with_class` | 122 |
| `final_val_metrics.edema.dice` | 0.7076 |
| `final_val_metrics.edema.jaccard` | 0.5475 |
| `final_val_metrics.edema.hausdorff_distance_mean` | 23.5769 |
| `final_val_metrics.edema.average_surface_distance_mean` | 1.9953 |
| `final_val_metrics.edema.val_slices_with_class` | 237 |
| `final_val_metrics.enhancing.dice` | 0.7958 |
| `final_val_metrics.enhancing.jaccard` | 0.6608 |
| `final_val_metrics.enhancing.hausdorff_distance_mean` | 8.009 |
| `final_val_metrics.enhancing.average_surface_distance_mean` | 0.8055 |
| `final_val_metrics.enhancing.val_slices_with_class` | 145 |
| `final_val_metrics.mean_tumor_dice` | 0.7296 |
| `history.train_loss` | [30 values] |
| `history.val_loss` | [30 values] |
| `history.val_metrics.1.necrotic_non_enhancing.dice` | 0.1334 |
| `history.val_metrics.1.necrotic_non_enhancing.jaccard` | 0.0715 |
| `history.val_metrics.1.necrotic_non_enhancing.hausdorff_distance_mean` | 31.6739 |
| `history.val_metrics.1.necrotic_non_enhancing.average_surface_distance_mean` | 10.6857 |
| `history.val_metrics.1.necrotic_non_enhancing.val_slices_with_class` | 122 |
| `history.val_metrics.1.edema.dice` | 0.6341 |
| `history.val_metrics.1.edema.jaccard` | 0.4642 |
| `history.val_metrics.1.edema.hausdorff_distance_mean` | 29.9968 |
| `history.val_metrics.1.edema.average_surface_distance_mean` | 5.7683 |
| `history.val_metrics.1.edema.val_slices_with_class` | 237 |
| `history.val_metrics.1.enhancing.dice` | 0.6719 |
| `history.val_metrics.1.enhancing.jaccard` | 0.5059 |
| `history.val_metrics.1.enhancing.hausdorff_distance_mean` | 34.5079 |
| `history.val_metrics.1.enhancing.average_surface_distance_mean` | 7.1297 |
| `history.val_metrics.1.enhancing.val_slices_with_class` | 145 |
| `history.val_metrics.1.mean_tumor_dice` | 0.4798 |
| `history.val_metrics.2.necrotic_non_enhancing.dice` | 0.3082 |
| `history.val_metrics.2.necrotic_non_enhancing.jaccard` | 0.1822 |
| `history.val_metrics.2.necrotic_non_enhancing.hausdorff_distance_mean` | 13.9989 |
| `history.val_metrics.2.necrotic_non_enhancing.average_surface_distance_mean` | 4.932 |
| `history.val_metrics.2.necrotic_non_enhancing.val_slices_with_class` | 122 |
| `history.val_metrics.2.edema.dice` | 0.479 |
| `history.val_metrics.2.edema.jaccard` | 0.3149 |
| `history.val_metrics.2.edema.hausdorff_distance_mean` | 25.9542 |
| `history.val_metrics.2.edema.average_surface_distance_mean` | 2.3861 |
| `history.val_metrics.2.edema.val_slices_with_class` | 237 |
| `history.val_metrics.2.enhancing.dice` | 0.8124 |
| `history.val_metrics.2.enhancing.jaccard` | 0.6841 |
| `history.val_metrics.2.enhancing.hausdorff_distance_mean` | 7.8679 |
| `history.val_metrics.2.enhancing.average_surface_distance_mean` | 0.9622 |
| `history.val_metrics.2.enhancing.val_slices_with_class` | 145 |
| `history.val_metrics.2.mean_tumor_dice` | 0.5332 |
| `history.val_metrics.3.necrotic_non_enhancing.dice` | 0.2944 |
| `history.val_metrics.3.necrotic_non_enhancing.jaccard` | 0.1726 |
| `history.val_metrics.3.necrotic_non_enhancing.hausdorff_distance_mean` | 14.2772 |
| `history.val_metrics.3.necrotic_non_enhancing.average_surface_distance_mean` | 5.0664 |
| `history.val_metrics.3.necrotic_non_enhancing.val_slices_with_class` | 122 |
| `history.val_metrics.3.edema.dice` | 0.5267 |
| `history.val_metrics.3.edema.jaccard` | 0.3575 |
| `history.val_metrics.3.edema.hausdorff_distance_mean` | 27.9153 |
| `history.val_metrics.3.edema.average_surface_distance_mean` | 5.2807 |
| `history.val_metrics.3.edema.val_slices_with_class` | 237 |
| `history.val_metrics.3.enhancing.dice` | 0.764 |
| `history.val_metrics.3.enhancing.jaccard` | 0.6182 |
| `history.val_metrics.3.enhancing.hausdorff_distance_mean` | 13.1691 |
| `history.val_metrics.3.enhancing.average_surface_distance_mean` | 2.5091 |
| `history.val_metrics.3.enhancing.val_slices_with_class` | 145 |
| `history.val_metrics.3.mean_tumor_dice` | 0.5284 |
| … | *432 more fields in the file* |

## `spine_level_analysis.json`

| Field | Value |
|---|---|
| `n` | 14 |
| `cases[0].slice_index` | 5 |
| `cases[0].axis` | horizontal |
| `cases[0].position_px` | 74 |
| `cases[0].narrowest_width_px` | 10.0 |
| `cases[0].median_width_px` | 17.2 |
| `cases[0].narrowing_ratio` | 0.5814 |
| `cases[0].nearest_disc_from_top` | 4 |
| `cases[0].profile_start_px` | 0 |
| `cases[0].vertical` | False |
| `cases[0].case` | SPINE_SP11_PATH_T2.nii.gz |
| `cases[0].group` | pathological |
| `cases[1].slice_index` | 6 |
| `cases[1].axis` | horizontal |
| `cases[1].position_px` | 316 |
| `cases[1].narrowest_width_px` | 17.2 |
| `cases[1].median_width_px` | 19.4 |
| `cases[1].narrowing_ratio` | 0.8866 |
| `cases[1].nearest_disc_from_top` | 12 |
| `cases[1].profile_start_px` | 0 |
| `cases[1].vertical` | False |
| `cases[1].case` | SPINE_SP12_PATH_T2.nii.gz |
| `cases[1].group` | pathological |
| `cases[2].slice_index` | 8 |
| `cases[2].axis` | horizontal |
| `cases[2].position_px` | 345 |
| `cases[2].narrowest_width_px` | 6.2 |
| `cases[2].median_width_px` | 12.0 |
| `cases[2].narrowing_ratio` | 0.5167 |
| `cases[2].nearest_disc_from_top` | 13 |
| `cases[2].profile_start_px` | 0 |
| `cases[2].vertical` | False |
| `cases[2].case` | SPINE_SP15_PATH_T2.nii.gz |
| `cases[2].group` | pathological |
| `cases[3].slice_index` | 7 |
| `cases[3].axis` | horizontal |
| `cases[3].position_px` | 352 |
| `cases[3].narrowest_width_px` | 13.2 |
| `cases[3].median_width_px` | 18.6 |
| `cases[3].narrowing_ratio` | 0.7097 |
| `cases[3].nearest_disc_from_top` | 15 |
| `cases[3].profile_start_px` | 0 |
| `cases[3].vertical` | False |
| `cases[3].case` | SPINE_SP17_PATH_T2.nii.gz |
| `cases[3].group` | pathological |

## `spine_measurement_validation.json`

| Field | Value |
|---|---|
| `unit_of_analysis` | patient (median over sampled slices) — not slice, because slices from one pat... |
| `n_normal_cases` | 10 |
| `n_pathological_cases` | 9 |
| `canal_detection_rate.normal` | 46/47 |
| `canal_detection_rate.pathological` | 45/45 |
| `metrics.narrowing_ratio.normal_mean` | 0.5575 |
| `metrics.narrowing_ratio.pathological_mean` | 0.4854 |
| `metrics.narrowing_ratio.auc_oriented` | 0.689 |
| `metrics.min_width_px.normal_mean` | 2.4985 |
| `metrics.min_width_px.pathological_mean` | 2.2522 |
| `metrics.min_width_px.auc_oriented` | 0.606 |
| `metrics.mean_width_px.normal_mean` | 5.0835 |
| `metrics.mean_width_px.pathological_mean` | 5.1078 |
| `metrics.mean_width_px.auc_oriented` | 0.522 |
| `metrics.variability_cv.normal_mean` | 0.3474 |
| `metrics.variability_cv.pathological_mean` | 0.318 |
| `metrics.variability_cv.auc_oriented` | 0.428 |
| `prespecified_test.metric` | narrowing_ratio |
| `prespecified_test.hypothesis` | one-sided: normal > pathological (stenosis narrows the canal) |
| `prespecified_test.mann_whitney_p` | 0.089 |
| `prespecified_test.significant_at_0.05` | False |
| `verdict` | Canal detection is reliable and the narrowing ratio moves in the direction st... |

## `spine_normal_coco.json`

| Field | Value |
|---|---|
| `info.description` | spine_normal ROI segmentation (MedhaDrishti hackathon) |
| `info.kind` | spine_roi |
| `images[0].id` | 1 |
| `images[0].file_name` | SP10_STIR_clahe.png |
| `images[0].width` | 224 |
| `images[0].height` | 224 |
| `images[1].id` | 2 |
| `images[1].file_name` | SP10_T1_clahe.png |
| `images[1].width` | 224 |
| `images[1].height` | 224 |
| `images[2].id` | 3 |
| `images[2].file_name` | SP10_T2_clahe.png |
| `images[2].width` | 224 |
| `images[2].height` | 224 |
| `images[3].id` | 4 |
| `images[3].file_name` | SP1_T1_clahe.png |
| `images[3].width` | 224 |
| `images[3].height` | 224 |
| `annotations[0].id` | 1 |
| `annotations[0].image_id` | 1 |
| `annotations[0].category_id` | 1 |
| `annotations[0].segmentation.size` | [2 values] |
| `annotations[0].segmentation.counts` | h03;4L7JG0:0I3MO0Nj01RO23Md03]OOO0OW11QO4HY11OO_OOWO15R1OSOc1Ko0O`MP1e5ROQJd0... |
| `annotations[0].area` | 19962.0 |
| `annotations[0].bbox` | [4 values] |
| `annotations[0].iscrowd` | 0 |
| `annotations[1].id` | 2 |
| `annotations[1].image_id` | 1 |
| `annotations[1].category_id` | 2 |
| `annotations[1].segmentation.size` | [2 values] |
| `annotations[1].segmentation.counts` | cY212S10nNb5b1\NbNaLLO`0B@1589]35]L@3MJ6\10W2i0aL[O[1MgN1[2:bM20Ca1J[N24:T27Z... |
| `annotations[1].area` | 13940.0 |
| `annotations[1].bbox` | [4 values] |
| `annotations[1].iscrowd` | 0 |
| `annotations[2].id` | 3 |
| `annotations[2].image_id` | 1 |
| `annotations[2].category_id` | 3 |
| `annotations[2].segmentation.size` | [2 values] |
| `annotations[2].segmentation.counts` | Qh222158d5S1E3bNdNRMa1oN[Ne3b0QM\1BUNd2NSMk0?V1E`NR2CkMl0:n1f1aNYN`1FSMh1_1aN... |
| `annotations[2].area` | 12055.0 |
| `annotations[2].bbox` | [4 values] |
| `annotations[2].iscrowd` | 0 |
| `annotations[3].id` | 4 |
| `annotations[3].image_id` | 2 |
| `annotations[3].category_id` | 1 |
| `annotations[3].segmentation.size` | [2 values] |
| `annotations[3].segmentation.counts` | 91_30kL3<MC2a08WOFh30cL2K5MJ0OO0=0;8WOI1Ob20]M0:12^1HbNk0a0a1@lM0HN1b1KcNOKk0... |
| `annotations[3].area` | 19474.0 |
| `annotations[3].bbox` | [4 values] |
| `annotations[3].iscrowd` | 0 |
| `categories[0].id` | 1 |
| `categories[0].name` | roi_cluster_1_unsupervised |
| `categories[1].id` | 2 |
| `categories[1].name` | roi_cluster_2_unsupervised |
| `categories[2].id` | 3 |
| `categories[2].name` | roi_cluster_3_unsupervised |

## `spine_pathological_coco.json`

| Field | Value |
|---|---|
| `info.description` | spine_pathological ROI segmentation (MedhaDrishti hackathon) |
| `info.kind` | spine_roi |
| `images[0].id` | 1 |
| `images[0].file_name` | SP11_STIR_clahe.png |
| `images[0].width` | 224 |
| `images[0].height` | 224 |
| `images[1].id` | 2 |
| `images[1].file_name` | SP11_T1_clahe.png |
| `images[1].width` | 224 |
| `images[1].height` | 224 |
| `images[2].id` | 3 |
| `images[2].file_name` | SP11_T2_clahe.png |
| `images[2].width` | 224 |
| `images[2].height` | 224 |
| `images[3].id` | 4 |
| `images[3].file_name` | SP12_STIR_clahe.png |
| `images[3].width` | 224 |
| `images[3].height` | 224 |
| `annotations[0].id` | 1 |
| `annotations[0].image_id` | 1 |
| `annotations[0].category_id` | 1 |
| `annotations[0].segmentation.size` | [2 values] |
| `annotations[0].segmentation.counts` | j1b04@740NI3Oo0a3\OmLEILL1NS1`3ZOoLEGLNOI02W1a3WORMFLNBV1`3VOSMEJ0BV1a3UOSMFI... |
| `annotations[0].area` | 9359.0 |
| `annotations[0].bbox` | [4 values] |
| `annotations[0].iscrowd` | 0 |
| `annotations[1].id` | 2 |
| `annotations[1].image_id` | 1 |
| `annotations[1].category_id` | 2 |
| `annotations[1].segmentation.size` | [2 values] |
| `annotations[1].segmentation.counts` | \242O4O09NI3OU55lJIO9LL1NX53oJJK9LNOI02Z54SKJGa0NBY51UKLC`00BY51UKMC?OCX51XKM... |
| `annotations[1].area` | 11703.0 |
| `annotations[1].bbox` | [4 values] |
| `annotations[1].iscrowd` | 0 |
| `annotations[2].id` | 3 |
| `annotations[2].image_id` | 1 |
| `annotations[2].category_id` | 3 |
| `annotations[2].segmentation.size` | [2 values] |
| `annotations[2].segmentation.counts` | e26i62M3M3M3N1O2N2N100O10000O1000O1O001BjI0X6N>OXJLg41[K0d4O^K0j50O0000000QO2... |
| `annotations[2].area` | 2604.0 |
| `annotations[2].bbox` | [4 values] |
| `annotations[2].iscrowd` | 0 |
| `annotations[3].id` | 4 |
| `annotations[3].image_id` | 2 |
| `annotations[3].category_id` | 1 |
| `annotations[3].segmentation.size` | [2 values] |
| `annotations[3].segmentation.counts` | i12d0;I4b0@_O0Q32SMN@:<5Q1AP22RMOB4OL0627\1Bn13SMNC01<L1`1Bm12SM0Gm0Z1QOl12SM... |
| `annotations[3].area` | 10120.0 |
| `annotations[3].bbox` | [4 values] |
| `annotations[3].iscrowd` | 0 |
| `categories[0].id` | 1 |
| `categories[0].name` | roi_cluster_1_unsupervised |
| `categories[1].id` | 2 |
| `categories[1].name` | roi_cluster_2_unsupervised |
| `categories[2].id` | 3 |
| `categories[2].name` | roi_cluster_3_unsupervised |

## `spine_stenosis_test.json`

| Field | Value |
|---|---|
| `n_normal` | 6 |
| `n_pathological` | 8 |
| `cases[0].case` | SPINE_SP11_PATH_T2.nii.gz |
| `cases[0].group` | pathological |
| `cases[0].narrowing_ratio` | 0.561 |
| `cases[1].case` | SPINE_SP12_PATH_T2.nii.gz |
| `cases[1].group` | pathological |
| `cases[1].narrowing_ratio` | 0.6585 |
| `cases[2].case` | SPINE_SP15_PATH_T2.nii.gz |
| `cases[2].group` | pathological |
| `cases[2].narrowing_ratio` | 0.4837 |
| `cases[3].case` | SPINE_SP17_PATH_T2.nii.gz |
| `cases[3].group` | pathological |
| `cases[3].narrowing_ratio` | 0.7048 |
| `mean_normal` | 0.5759 |
| `mean_pathological` | 0.508 |
| `p_value` | 0.331 |
| `auc` | 0.5833 |
| `significant` | False |

## `spine_vs_spineps.json`

| Field | Value |
|---|---|
| `scan` | sub-SP11_T2w.nii.gz |
| `slice_index` | 5 |
| `note` | Dice values are oracle-assisted upper bounds: the reference selects which uns... |
| `spineps_reference.semantic_labels_on_slice` | [9 values] |
| `spineps_reference.n_semantic_structures_volume` | 13 |
| `spineps_reference.n_vertebra_instances` | 17 |
| `methods.k-means (intensity).n_clusters` | 4 |
| `methods.k-means (intensity).regions.Vertebral bodies.dice` | 0.263 |
| `methods.k-means (intensity).regions.Vertebral bodies.dice_sd` | 0.0 |
| `methods.k-means (intensity).regions.Vertebral bodies.recall` | 0.8074 |
| `methods.k-means (intensity).regions.Vertebral bodies.precision` | 0.1571 |
| `methods.k-means (intensity).regions.Vertebral bodies.dice_union` | 0.263 |
| `methods.k-means (intensity).regions.Vertebral bodies.clusters_merged` | 1 |
| `methods.k-means (intensity).regions.Vertebral bodies.reference_px` | 7044 |
| `methods.k-means (intensity).regions.Vertebral bodies.n_runs` | 1 |
| `methods.k-means (intensity).regions.Intervertebral discs.dice` | 0.0467 |
| `methods.k-means (intensity).regions.Intervertebral discs.dice_sd` | 0.0 |
| `methods.k-means (intensity).regions.Intervertebral discs.recall` | 0.7502 |
| `methods.k-means (intensity).regions.Intervertebral discs.precision` | 0.0241 |
| `methods.k-means (intensity).regions.Intervertebral discs.dice_union` | 0.0467 |
| `methods.k-means (intensity).regions.Intervertebral discs.clusters_merged` | 1 |
| `methods.k-means (intensity).regions.Intervertebral discs.reference_px` | 1441 |
| `methods.k-means (intensity).regions.Intervertebral discs.n_runs` | 1 |
| `methods.k-means (intensity).regions.Spinal canal + cord.dice` | 0.3041 |
| `methods.k-means (intensity).regions.Spinal canal + cord.dice_sd` | 0.0 |
| `methods.k-means (intensity).regions.Spinal canal + cord.recall` | 0.7047 |
| `methods.k-means (intensity).regions.Spinal canal + cord.precision` | 0.1939 |
| `methods.k-means (intensity).regions.Spinal canal + cord.dice_union` | 0.3041 |
| `methods.k-means (intensity).regions.Spinal canal + cord.clusters_merged` | 1 |
| `methods.k-means (intensity).regions.Spinal canal + cord.reference_px` | 6563 |
| `methods.k-means (intensity).regions.Spinal canal + cord.n_runs` | 1 |
| `methods.k-means (intensity).regions.Posterior elements.dice` | 0.0706 |
| `methods.k-means (intensity).regions.Posterior elements.dice_sd` | 0.0 |
| `methods.k-means (intensity).regions.Posterior elements.recall` | 0.5296 |
| `methods.k-means (intensity).regions.Posterior elements.precision` | 0.0378 |
| `methods.k-means (intensity).regions.Posterior elements.dice_union` | 0.0706 |
| `methods.k-means (intensity).regions.Posterior elements.clusters_merged` | 1 |
| `methods.k-means (intensity).regions.Posterior elements.reference_px` | 2583 |
| `methods.k-means (intensity).regions.Posterior elements.n_runs` | 1 |
| `methods.SLIC superpixels.n_clusters` | 4 |
| `methods.SLIC superpixels.regions.Vertebral bodies.dice` | 0.2041 |
| `methods.SLIC superpixels.regions.Vertebral bodies.dice_sd` | 0.0 |
| `methods.SLIC superpixels.regions.Vertebral bodies.recall` | 0.659 |
| `methods.SLIC superpixels.regions.Vertebral bodies.precision` | 0.1208 |
| `methods.SLIC superpixels.regions.Vertebral bodies.dice_union` | 0.2041 |
| `methods.SLIC superpixels.regions.Vertebral bodies.clusters_merged` | 1 |
| `methods.SLIC superpixels.regions.Vertebral bodies.reference_px` | 7044 |
| `methods.SLIC superpixels.regions.Vertebral bodies.n_runs` | 1 |
| `methods.SLIC superpixels.regions.Intervertebral discs.dice` | 0.0466 |
| `methods.SLIC superpixels.regions.Intervertebral discs.dice_sd` | 0.0 |
| `methods.SLIC superpixels.regions.Intervertebral discs.recall` | 0.6787 |
| `methods.SLIC superpixels.regions.Intervertebral discs.precision` | 0.0241 |
| `methods.SLIC superpixels.regions.Intervertebral discs.dice_union` | 0.0466 |
| `methods.SLIC superpixels.regions.Intervertebral discs.clusters_merged` | 1 |
| `methods.SLIC superpixels.regions.Intervertebral discs.reference_px` | 1441 |
| `methods.SLIC superpixels.regions.Intervertebral discs.n_runs` | 1 |
| `methods.SLIC superpixels.regions.Spinal canal + cord.dice` | 0.3019 |
| `methods.SLIC superpixels.regions.Spinal canal + cord.dice_sd` | 0.0 |
| `methods.SLIC superpixels.regions.Spinal canal + cord.recall` | 0.73 |
| `methods.SLIC superpixels.regions.Spinal canal + cord.precision` | 0.1903 |
| `methods.SLIC superpixels.regions.Spinal canal + cord.dice_union` | 0.3019 |
| `methods.SLIC superpixels.regions.Spinal canal + cord.clusters_merged` | 1 |
| `methods.SLIC superpixels.regions.Spinal canal + cord.reference_px` | 6563 |
| `methods.SLIC superpixels.regions.Spinal canal + cord.n_runs` | 1 |
| `methods.SLIC superpixels.regions.Posterior elements.dice` | 0.097 |
| `methods.SLIC superpixels.regions.Posterior elements.dice_sd` | 0.0 |
| `methods.SLIC superpixels.regions.Posterior elements.recall` | 0.77 |
| `methods.SLIC superpixels.regions.Posterior elements.precision` | 0.0517 |
| `methods.SLIC superpixels.regions.Posterior elements.dice_union` | 0.097 |
| `methods.SLIC superpixels.regions.Posterior elements.clusters_merged` | 1 |
| … | *36 more fields in the file* |


# PART VII — EVERY SOURCE FILE, AND WHAT IT DOES

One line each, taken from the file's own docstring.

| File | Purpose |
|---|---|
| `demos\01_brain_he.py` | Histogram Equalisation (HE) on a brain MRI -- CLASSICAL baseline, no AI. |
| `demos\02_brain_clahe.py` | CLAHE on a brain MRI -- CLASSICAL baseline, no AI. |
| `demos\03_brain_unet_enhance.py` | OUR enhancement model on a brain MRI -- 2D U-Net, self-supervised. |
| `demos\04_brain_tumour_seg.py` | OUR tumour segmentation model -- 2D U-Net, SUPERVISED on BraTS2020. |
| `demos\05_brain_gradcam.py` | Grad-CAM -- explainability. Proves WHERE the network looked. |
| `demos\06_brain_tissue.py` | Healthy-tissue segmentation -- CSF / grey matter / white matter. |
| `demos\10_spine_clahe.py` | CLAHE on a SPINE MRI -- classical baseline for the spine track. |
| `demos\11_spine_unet_enhance.py` | OUR spine enhancement -- one U-Net PER SEQUENCE, self-supervised. |
| `demos\12_spine_selfsup_seg.py` | OUR spine ROI segmentation -- self-supervised CNN, ZERO annotations. |
| `demos\13_spine_canal.py` | Spinal canal width -- MEASUREMENT, not prediction. |
| `demos\14_spine_spineps.py` | SPINEPS -- the PRETRAINED model. Per-vertebra instances. |
| `demos\_common.py` | Shared helpers for the demo scripts. |
| `demos\run_all_brain.py` | EVERY brain stage, one command. Run this when a judge says "show me the brain". |
| `demos\run_all_spine.py` | EVERY spine stage, one command. Run this when a judge says "show me the spine". |
| `demos\run_everything.py` | THE WHOLE PROJECT, one command. Brain then spine, every stage. |
| `src\annotation_viz.py` | Organiser ask: "Annotations and labels need to be understood", and the Stage-2 |
| `src\assemble_demo_samples.py` | Bundles a small folder of images the models have NEVER trained on, for live |
| `src\benchmark.py` | The problem statement asks (twice - once for enhancement, once for |
| `src\brain_dataset.py` | Builds two PyTorch datasets from a BraTS-structured root folder: |
| `src\build_demo_page.py` | Bakes ALL demo figures + plain-language explanations into ONE self-contained |
| `src\build_master_reference.py` | The project's knowledge is spread across several documents, which is right for |
| `src\build_showcase.py` | Builds ONE folder (showcase/) with everything for the live demo: |
| `src\coco_export.py` | Converts predicted ROI masks into COCO-format JSON (a required hackathon |
| `src\compare_modality_models.py` | Fair head-to-head: does a MODALITY-SPECIFIC spine enhancement model beat the |
| `src\comparison_graphs.py` | Generates the "why ours wins" figures from the measured result files. Every |
| `src\cross_validation.py` | The problem statement asks for "Cross validation accuracy" of the DL model. |
| `src\dataset_splits_report.py` | Answers the organisers' explicit asks: |
| `src\dataset_stats.py` | Computes the image-property assessment the problem statement mandates, for |
| `src\demo.py` | Produces the "dirty MRI in -> clean MRI out" visuals and the dataset-analysis |
| `src\enhancement_dataset.py` | A modality-agnostic enhancement dataset for the OFFLINE hackathon data |
| `src\extract_brats_prefix.py` | Stream-extract complete BraTS case files from a PARTIALLY-downloaded ZIP64 |
| `src\full_segmentation_metrics.py` | Stage-4 gap-fill: the problem statement requires the FULL metric list for ROI |
| `src\generate_demo_assets.py` | Generates the full set of figures for the comprehensive demo page: |
| `src\gradcam.py` | The problem statement lists Grad-CAM / attention maps as a Stage-4 evaluation |
| `src\inference_report.py` | Runs the trained models on the ACTUAL offline hackathon dataset (not BraTS, |
| `src\metrics.py` | Every metric the problem statement asks for, in one place: |
| `src\model_inspect.py` | Opens the model up completely: every layer, the tensor shape flowing through |
| `src\models.py` | One U-Net backbone, two heads: |
| `src\mri_degradation.py` | Synthesizes MRI-realistic degradation to build enhancement training pairs |
| `src\nifti_utils.py` | Shared utilities for loading NIfTI (.nii/.nii.gz) volumes and extracting |
| `src\offline_dataset.py` | Discovery + sub-modality classification for the OFFLINE hackathon datasets |
| `src\paper_comparison.py` | Stage-3 gap-fill. The problem statement asks that enhancement performance be |
| `src\plots.py` | Generates the learning-curve / metric figures the problem statement asks for |
| `src\precompute_spineps.py` | The live button runs the semantic phase in about a minute, which is fine to |
| `src\preprocessing_assessment.py` | Stage-2 gap-fill. The problem statement says explicitly: |
| `src\resolution_stats.py` | The problem statement's Stage-1 analysis asks for "MRI resolution, contrast |
| `src\spine_autoencoder.py` | Problem: the spine dataset has NO lesion labels and no external data is allowed, |
| `src\spine_deep_segmentation.py` | The organisers require spine models that need NO annotations for training. Our |
| `src\spine_level_analysis.py` | WHY THIS EXISTS, AND WHAT IT IS NOT |
| `src\spine_measurements.py` | Why this exists: the spine data has no labels, and our autoencoder anomaly |
| `src\spine_method_comparison.py` | Side-by-side of every spine segmentation approach we implemented, on the same |
| `src\spine_pipeline.py` | Per the strategic decision in CLAUDE.md: the offline Spine dataset has NO |
| `src\spine_stenosis_test.py` | normal one? |
| `src\spine_vs_spineps.py` | SPINEPS is trained on external annotated data, so it can name structures we |
| `src\spineps_gpu.py` | WHY THIS EXISTS |
| `src\spineps_runner.py` | SPINEPS (Möller et al., European Radiology 2025, Apache-2.0) is the first |
| `src\ssim.py` | Differentiable SSIM (Structural Similarity) in pure torch, for use as a |
| `src\synthetic_brats.py` | Generates fake volumes that match BraTS2020's exact folder/file naming |
| `src\tissue_segmentation.py` | The problem statement asks, for *healthy* subjects, to segment the region of |
| `src\train_enhancement_brain.py` | Trains the enhancement U-Net on BraTS FLAIR slices (degraded -> clean), |
| `src\train_enhancement_offline.py` | Trains the enhancement U-Net on an OFFLINE hackathon dataset (Spine or |
| `src\train_segmentation_brain.py` | Trains the multi-class segmentation U-Net on BraTS. Input = 4 stacked |
| `src\utilization_bench.py` | The problem statement's systematic-comparison deliverable asks for GPU/CPU |
| `src\validate_anomaly_detector.py` | Honest validation of the spine anomaly autoencoder. |
| `src\validate_spine_measurements.py` | Tests whether the canal-width measurements (spine_measurements.py) actually |
| `src\vertebra_detection.py` | The reference spine literature presents segmentation as INDIVIDUAL vertebrae, |
| `src\webapp.py` | A tiny local website for live judging: upload an MRI (.nii/.nii.gz) or an |
