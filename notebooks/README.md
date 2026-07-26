# Notebooks

## `brats_3d_unet_colab.ipynb` — 3D U-Net on BraTS2020, resumable

Answers the one open question the hackathon hardware could not: **how much does
3D segmentation actually buy over our 2D models?**

Our 2D pipeline segments slice by slice, because published 3D BraTS models
document a 16 GB+ VRAM requirement and the development laptop has 6 GB. That is
a hardware limit rather than a method limit, so the claim "3D would be better"
stayed a hypothesis. A Colab T4 has ~15 GB, which is enough for patch-based 3D
training, so this notebook turns it into a measurement.

**Baseline to beat:** mean tumour Dice **0.76**, enhancing tumour **0.84**.

### It survives disconnects

Colab sessions end — on idle, on timeout, or at random. Every epoch writes a
checkpoint to Google Drive, and on start the notebook resumes from the newest
one, restoring weights, optimiser state, epoch counter and metric history. Close
the tab, come back the next day, run all cells: it continues where it stopped.

Nothing needs to finish in one sitting.

### Running it

1. Open in Colab, set *Runtime → Change runtime type → T4 GPU*.
2. Run cells in order. Cell 3 pulls BraTS **directly from Kaggle into Colab** —
   do not upload 14 GB from your own machine; you will need a `kaggle.json` API
   token, which the cell prompts for.
3. Later sessions: run all cells again. Cell 7 detects the checkpoint and
   resumes.

### Reading the result honestly

Cell 9 plots 3D against the 2D baseline. One caveat is stated in the notebook
and repeated here: the 2D number is computed over whole validation slices while
this validates on centre patches, so the evaluations are close but not
identical. Treat a fraction of a point as noise — a real win should be several
points.

**If 3D does not clearly win, that is a legitimate result and worth reporting.**
