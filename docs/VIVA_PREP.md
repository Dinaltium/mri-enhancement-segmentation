# VIVA PREP — every question they can ask, with the answer

Read top to bottom once. The numbers here are all measured; do not invent new ones.
If you do not know something, say **"I don't have that number to hand, it's in
`results/<file>.json`"** — that is a strong answer. Guessing is the only losing move.

---

## 0. THE 12 NUMBERS TO MEMORISE

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

## 1. FOUNDATIONS — "what is X?"

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

## 2. LOSS FUNCTIONS — they asked this and you did not know. Know it cold.

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

## 3. INPUTS AND OUTPUTS — "what did you feed it?"

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

## 4. DATASETS

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

## 5. RESULTS — what we gained

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

## 6. SPINE — the hard questions

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

## 7. THE PRETRAINED MODEL — expect hostility, answer calmly

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

## 8. EXPLAINABILITY AND SAFETY

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

## 9. "WHAT WOULD YOU IMPROVE?" — never say "nothing"

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

## 10. TRAPS — questions designed to catch you

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
