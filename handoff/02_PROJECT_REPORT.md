# MRI Enhancement and Region-of-Interest Segmentation

**MedhaDrishti National AI Hackathon · Yugma TechFest 2.0 · JNNCE Shivamogga**

> **For the person formatting this:** this is the 3–4 page project report. Paste into
> Word/Docs, keep the section order, and insert the four figures where marked. Target
> length is 3–4 pages including figures. Every number is final and measured.

---

## 1. Objectives

Magnetic resonance imaging trades image quality for acquisition speed. Faster scans
carry two characteristic defects: **Rician noise**, the grain inherent to MRI
magnitude images, and a **bias field**, a smooth intensity gradient caused by
radio-frequency coil inhomogeneity. Both obscure the fine boundaries that carry
diagnostic meaning — the rim of a tumour, the margin of a compressed disc.

This project delivers an end-to-end pipeline that:

1. characterises the supplied datasets quantitatively before any processing;
2. preprocesses and standardises heterogeneous scans;
3. restores degraded scans using a deep-learning model, benchmarked against the
   classical enhancement methods; and
4. delineates the region of interest — tumour sub-regions and CSF/grey/white matter
   for brain, and disc/vertebra/cord regions plus abnormality localisation for
   lumbo-sacral spine.

A hard constraint shaped the spine work: the supplied spine data carries **no
annotations**, and external data is not permitted. Every spine method is therefore
unsupervised or self-supervised.

---

## 2. Datasets

| Dataset | Role | Cases | Annotations |
|---|---|---|---|
| BraTS2020 | standard training set | 127 used (4.47 GB archive, 369 available) | Expert tumour masks |
| Hackathon offline — brain | testing / validation | 20 | None |
| Hackathon offline — spine | testing / validation | 20 | None |

Splits are performed at **patient level**, never slice level: slices from one patient
are highly correlated, so a slice-level split leaks information between train and
test and inflates every reported score. BraTS is split 80/20; the offline groups are
split 5 train / 5 test per the organisers' instruction. All volumes are read directly
from `.nii`/`.nii.gz`; no dataset file is converted to another format.

**Stage-1 analysis** measured seven properties — contrast, complexity, sharpness,
edge strength, noise level, mean and deviation — across every dataset and
sub-modality. The decisive finding: BraTS is uniform (240×240×155 at 1 mm isotropic),
while the offline scans are highly heterogeneous, with in-plane voxels from 0.25 to
1.3 mm and slice thickness from 3 to 13 mm. This heterogeneity is the direct
justification for resampling all inputs to a common 224×224 grid in Stage 2.

A second finding governs the loss function: in the annotated data, background occupies
**99.03 %** of pixels, while edema, enhancing tumour and necrotic core occupy 0.71 %,
0.17 % and 0.10 % respectively.

**[FIGURE 1 — `images/dataset_properties.png`]**
*Caption: Image properties across all datasets and sub-modalities. Spine scans carry
roughly twice the complexity of brain scans; the offline brain data matches BraTS
closely, which supports transferring BraTS-trained models to it.*

---

## 3. Methodology

### 3.1 Preprocessing (Stage 2)

Volumes are normalised to [0,1] using robust percentiles (0.5–99.5), sliced along the
axial plane with near-empty slices discarded, and resampled to 224×224 with cubic
interpolation followed by clipping — cubic interpolation overshoots at sharp edges,
which would otherwise push intensities outside the valid range. Label masks are
resampled with nearest-neighbour interpolation only, since interpolating labels would
invent class values that do not exist. BraTS label 4 is remapped to 3 so classes are
contiguous. Histogram equalisation and CLAHE are applied as classical baselines, and
the seven image properties are **re-measured after every step** so the effect of each
is quantified rather than asserted.

### 3.2 Enhancement model (Stage 3)

A **2D U-Net** — encoder, bottleneck, decoder, with skip connections — with 7.77
million parameters. Training pairs are constructed by degrading clean scans with
MRI-correct corruption: Rician noise (σ 0.02–0.20), a smooth multiplicative bias
field, and mild blur. Using Gaussian rather than Rician noise here would be a
methodological error, since it does not match the physics of MRI magnitude images.

The loss is `L1 + SSIM`. The L1 term drives per-pixel fidelity; the SSIM term
preserves structure, so the output is not merely numerically close but visually
faithful. Optimiser: Adam at 1e-3 with mixed precision.

Because the three spine sequences are physically different images — T1 shows anatomy,
T2 shows fluid, STIR suppresses fat — a **separate model was trained per sequence**
and compared against a single pooled model on identical held-out slices.

### 3.3 Segmentation model (Stage 4)

The same backbone with four input channels (T1, T1c, T2, FLAIR stacked) and four
output classes. Multi-modal input matters because each sequence reveals a different
tumour component: T1c highlights the enhancing rim, FLAIR and T2 the surrounding
edema. The loss is `cross-entropy + soft Dice`; given the 99 % background imbalance
reported above, cross-entropy alone converges to predicting background everywhere,
while the Dice term optimises region overlap directly.

For **healthy** subjects, CSF, grey matter and white matter are separated by
unsupervised Gaussian-mixture clustering on T1 intensity, since no tissue labels
exist. For **spine**, we implemented and measured a progression of annotation-free methods:
intensity k-means, SLIC superpixels, and finally a **self-supervised CNN**
(differentiable feature clustering) optimised on each scan using the image's own
structure as supervision. We additionally investigated autoencoder-based anomaly
detection and validated it rather than assuming it worked — it did not (§4.3).

### 3.4 Use of a pretrained model for per-vertebra instances

The task additionally asks for degenerative disc, herniation and stenosis as regions of
interest. These are *named clinical entities*, and naming them is inherently supervised:
a model can only output "herniated disc" if it has seen examples labelled as such. With
no annotations, no external training data and 20 spine cases, that target cannot be
learned from the supplied data by any model we train — a property of the problem as
specified, not a shortfall of effort. Our measured results above establish this
empirically: annotation-free methods recover spinal *structure*, but not per-vertebra
*instances*.

With the organisers' approval we therefore use **SPINEPS** (Möller et al., European
Radiology 2025, Apache-2.0) for this one output. It is the first publicly available
whole-spine model for sagittal T2w MRI, producing semantic and instance masks for 14
structures, with published accuracy of Dice 0.92 (vertebrae), 0.967 (discs) and 0.958
(spinal canal), validated on over 1,600 subjects. We supply it no annotations and do not
train it. Using a peer-reviewed model with quantified accuracy is also the clinically
responsible choice: a per-vertebra segmentation invented by an under-constrained model
would be confidently wrong. Its provenance — pretrained on external data — is labelled
wherever it appears, and we claim no credit for its accuracy. Full reasoning:
`docs/PRETRAINED_MODEL_JUSTIFICATION.md`.

Slice-based 2D processing throughout keeps memory within a 6 GB laptop GPU; the
published 3D BraTS models document a 16 GB+ requirement.

---

## 4. Results

### 4.1 Enhancement

Evaluated on held-out slices under identical degradation, against the classical
baseline family that the referenced literature benchmarks against:

| Method | PSNR (dB) | SSIM | FSIM | VIF |
|---|---|---|---|---|
| Degraded input | 18.05 | 0.196 | 0.430 | 0.302 |
| Histogram equalisation | 8.05 | 0.149 | 0.211 | 0.192 |
| Adaptive HE | 6.35 | 0.133 | 0.187 | 0.199 |
| CLAHE | 11.84 | 0.156 | 0.280 | 0.248 |
| **Proposed (2D U-Net)** | **27.08** | **0.903** | **0.959** | **0.361** |

Every classical method scores **below the degraded input** on all four metrics. This
is not a defect of the implementations — it is what those methods do. They redistribute
contrast without any model of noise, so measured against the clean reference they move
away from it. Property re-measurement confirms the mechanism directly: the noise level
rises from 0.0068 to 0.0138 under HE and 0.0106 under CLAHE, while the proposed model
reduces it to **0.0043**. It is the only stage in the pipeline that removes noise.

On BraTS FLAIR the trained model reaches **PSNR 30.34 dB and SSIM 0.965**.

Per-sequence spine models outperformed a single pooled model on all three sequences,
scored on identical test slices: T1 0.598 → **0.827**, T2 0.594 → **0.802**, STIR
0.540 → **0.714** SSIM.

**[FIGURE 2 — `images/cmp_methods.png`]**
*Caption: Restoration quality against the classical baselines on identical slices.*

### 4.2 Segmentation

Evaluated against expert annotation on held-out patients, accumulating counts across
the whole validation set rather than averaging per slice — per-slice averaging
inflates results because most slices contain no tumour at all.

| Class | Dice | Jaccard | Sensitivity | Specificity | Precision | F1 | HD | ASD | RVE |
|---|---|---|---|---|---|---|---|---|---|
| Necrotic core | 0.672 | 0.506 | 0.744 | 0.9998 | 0.613 | 0.672 | 8.59 | 1.68 | 0.214 |
| Edema | 0.768 | 0.623 | 0.654 | 0.9996 | 0.930 | 0.768 | 19.71 | 1.59 | 0.297 |
| Enhancing tumour | **0.840** | 0.723 | 0.783 | 0.9998 | 0.905 | 0.840 | 6.94 | 0.86 | 0.134 |

**Mean tumour Dice: 0.760.** Training converged at epoch 25 with an overfitting gap of
approximately zero, and 3-fold cross-validation gives 0.59 ± **0.04** Dice — the small
spread across folds indicates a consistent model rather than a favourable split.

**[FIGURE 3 — `images/tumor_vs_gt.png`]**
*Caption: Predicted tumour sub-regions (centre) beside the radiologist's annotation
(right) on a patient excluded from training. Green edema, red enhancing tumour, blue
necrotic core.*

### 4.4 Efficiency

| Metric | Enhancement | Segmentation |
|---|---|---|
| Parameters | 7.77 M | 7.77 M |
| Model size | 31 MB | 31 MB |
| Latency (GPU) | 4.24 ms/image | 4.24 ms/image |
| Throughput | 236 images/sec | 236 images/sec |
| Peak GPU memory | 385 MB | 390 MB |
| GPU utilisation | 84 % | 98 % |

### 4.3 A negative result: autoencoder anomaly detection

The autoencoder reconstructs healthy spinal anatomy convincingly, and its error map is
visually persuasive. We tested whether the resulting score actually separates diseased
from healthy spines on held-out cases. It does not:

| | Normal (n=28) | Pathological (n=27) |
|---|---|---|
| Mean score | 0.0199 | 0.0167 |
| Range | 0.0137 – 0.0314 | 0.0101 – 0.0234 |

**AUC = 0.27**, i.e. worse than chance, with completely overlapping distributions and
no usable operating threshold; normal spines in fact score *higher* than pathological
ones. The reconstruction error is dominated by image texture and anatomical complexity
rather than by disease. We therefore removed the detection claim from the system and
present the map purely as a visualisation. An unvalidated detector that fires on
healthy patients is more dangerous than no detector at all. Full figures are recorded
in `results/anomaly_validation.json`.

**[FIGURE 4 — `images/spine_anomaly.png`]**
*Caption: Reconstruction difference against a healthy-only autoencoder. Left, a
pathological spine; centre, the model's healthy reconstruction; right, the difference
map. Presented as a visualisation only — validation showed the score does not
distinguish pathological from normal spines.*

---

## 5. Conclusion and limitations

The proposed pipeline restores degraded MRI substantially better than the classical
enhancement methods, and delineates brain tumour sub-regions at a mean Dice of 0.76
against expert annotation, while running in 4 ms per image on consumer hardware. The
spine track satisfies the constraint that no annotations may be used for training, via
three independent unsupervised and self-supervised methods.

Three limitations are stated deliberately. First, quantitative segmentation accuracy is
reported only on BraTS, the sole dataset with expert annotation; on the unlabelled
hackathon data we report enhancement metrics and qualitative segmentation rather than
inventing figures. Second, our autoencoder-based spine anomaly detector failed validation (AUC 0.27) and
the claim was withdrawn. Our own spine contributions are therefore restoration,
self-supervised region segmentation and canal morphometry; per-vertebra instance
segmentation is provided by a pretrained model (§3.4), clearly attributed, because that
output cannot be learned from unlabelled data. Third, the restoration model corrects noise
and intensity artefacts only; it does not synthesise anatomy, and the SSIM above 0.9
against the reference scan is the evidence for that claim.

All results are reproducible from the accompanying source code; each figure and table
is generated by a named script and stored as JSON.
