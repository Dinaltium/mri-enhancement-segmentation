# Notebooks

Two notebooks, one experiment: **how much does 3D segmentation actually buy over
our 2D models?**

Our 2D pipeline segments slice by slice, because published 3D BraTS models
document a 16 GB+ VRAM requirement and the development laptop has 6 GB. That is
a hardware limit rather than a method limit, so "3D would be better" stayed a
hypothesis. Free GPUs on Kaggle and Colab have ~15 GB, which is enough for
patch-based 3D — so these turn it into a number.

**Baseline to beat:** mean tumour Dice **0.76**, enhancing tumour **0.84**.

| Notebook | Host |
|---|---|
| `brats_3d_unet.ipynb` | **Kaggle** |
| `brats_3d_unet_colab.ipynb` | **Colab** |

Same model, same loss, same splits, same config. Only the environment differs.

## Why two files rather than one that detects its platform

An earlier single notebook auto-detected its host. **On Colab that detection was
wrong** — the Colab image provides a `/kaggle` path, so the notebook believed it
was on Kaggle and pointed checkpoints at `/kaggle/working`, which does not
persist on a Colab VM. Every checkpoint would have been lost on disconnect.

Two explicit notebooks remove the guess. Paths are hardcoded per host, and the
Colab one asserts its checkpoint directory is on Drive before training starts.

## Which to use when

|  | Kaggle | Colab |
|---|---|---|
| BraTS data | **already hosted** — mounts instantly via *Add Data* | re-downloads ~14 GB per session |
| Unattended running | **up to 12 h after closing the tab** (*Save & Run All*) | stops on idle (~90 min) |
| Checkpoints persist in | `/kaggle/working` (only on *Save Version*) | Google Drive |

**Kaggle for long overnight runs. Colab for shorter interactive ones.**

## Moving a run between them

The last cell of each notebook pushes `best.pt` and `history.json` to a Kaggle
Dataset; the early optional cell in the other pulls it back. So a run started
overnight on Kaggle can be continued on Colab the next morning at the same
epoch.

Every checkpoint carries optimiser state, epoch counter and the full metric
history, and records which platform produced each epoch — so `history.json`
shows exactly where the run has been.

## What survives a disconnect

Checkpoints are written **every epoch** to the persistent location, never to the
VM's own disk. Losing a session costs the current epoch, never the run — restart
and the resume cell continues automatically.

On Kaggle, remember `/kaggle/working` persists **only if you Save a Version**,
which is why long runs should use *Save & Run All (Commit)* rather than the
interactive editor.

## Reading the result honestly

The comparison cell plots 3D against the 2D baseline. The caveat, stated in both
notebooks and repeated here: our 2D number is computed over whole validation
slices while these validate on centre patches, so the evaluations are close but
not identical. Treat a fraction of a point as noise — a real win should be
several points.

**If 3D does not clearly win, that is a legitimate result and worth reporting.**
