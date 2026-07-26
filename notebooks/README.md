# Notebooks

## `brats_3d_unet.ipynb` — 3D U-Net on BraTS2020

Answers the one question the hackathon hardware could not: **how much does 3D
segmentation actually buy over our 2D models?**

Our 2D pipeline segments slice by slice, because published 3D BraTS models
document a 16 GB+ VRAM requirement and the development laptop has 6 GB. That is
a hardware limit rather than a method limit, so "3D would be better" stayed a
hypothesis. Free GPUs on Kaggle and Colab have ~15 GB, which is enough for
patch-based 3D — so this turns it into a number.

**Baseline to beat:** mean tumour Dice **0.76**, enhancing tumour **0.84**.

## One notebook, both platforms

Cell 1 detects where it is running and adapts the data paths and checkpoint
location. Nothing else in the notebook is platform-specific.

| | Kaggle | Colab |
|---|---|---|
| BraTS data | **already hosted** — mounts instantly via *Add Data* | re-downloads ~14 GB per session |
| Background running | **up to 12 h after closing the tab** (*Save & Run All*) | free tier stops when idle (~90 min) |
| Interactive idle timeout | ~20 min | ~90 min |
| Checkpoints persist in | `/kaggle/working` (only on *Save Version*) | Google Drive |

**Use Kaggle for long overnight runs; Colab for shorter interactive ones.**

## Handing a run between them

Cell 10 pushes `best.pt` and `history.json` to a Kaggle Dataset; cell 3 pulls
them back. Both platforms can read it, so you can train overnight on Kaggle,
continue on Colab in the morning, and the run picks up at the same epoch.

Every checkpoint carries optimiser state, epoch counter and the full metric
history, so nothing is lost in the handover.

## What survives a disconnect

Checkpoints are written **every epoch** to the persistent location, never to the
VM's own disk. Losing a session costs the current epoch, never the run — restart
and cell 7 resumes automatically.

On Kaggle the one thing to remember is that `/kaggle/working` persists **only if
you Save a Version**, which is why long runs should use *Save & Run All (Commit)*
rather than the interactive editor.

## Reading the result honestly

Cell 9 plots 3D against the 2D baseline. The caveat, stated in the notebook and
repeated here: our 2D number is computed over whole validation slices while this
validates on centre patches, so the evaluations are close but not identical.
Treat a fraction of a point as noise — a real win should be several points.

**If 3D does not clearly win, that is a legitimate result and worth reporting.**
