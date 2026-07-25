# Why we use a pretrained model for spine ROI segmentation

*(Requested by the organisers alongside approval to use a publicly available
pretrained model. This states what we use, why the alternative was insufficient,
and exactly what is and is not our own work.)*

---

## 1. What we use

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

---

## 2. Why a pretrained model is necessary here

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

## 3. What we tried first, and what it showed

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

## 4. Why this is scientifically appropriate, not a shortcut

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

## 5. What it actually produced on our data

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

## 6. How we report it — the provenance is never hidden

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

## 7. Summary

> We use SPINEPS for per-vertebra spine instance segmentation because that
> output is supervised by nature and cannot be learned from an unlabelled
> 20-case dataset with no external data permitted. We established this by
> implementing and measuring four annotation-free alternatives first, and
> reporting honestly that they do not reach per-vertebra instances. SPINEPS is
> open-source, peer-reviewed and validated at Dice 0.92; we supply it no
> annotations, we do not train it, and we label its provenance everywhere it
> appears.
