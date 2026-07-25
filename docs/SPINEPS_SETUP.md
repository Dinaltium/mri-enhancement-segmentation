# SPINEPS setup (per-vertebra spine segmentation)

SPINEPS is used **only** for per-vertebra instance segmentation on spine MRI.
Everything else in this project is our own model. The reasoning for using it is
in [PRETRAINED_MODEL_JUSTIFICATION.md](PRETRAINED_MODEL_JUSTIFICATION.md).

It runs in its **own conda environment**, separate from the project's `tfenv`,
for two reasons: it requires Python 3.11 (the project runs on 3.10), and it
pins dependency versions that would otherwise disturb the main environment.

---

## Install

```bash
conda create -n spineps python=3.11 -y
conda activate spineps
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install spineps
```

### Important: re-pin CUDA torch after installing spineps

`pip install spineps` pulls `nnunetv2`, which **replaces the CUDA build of
torch with the CPU wheel from PyPI** and leaves `torchvision` at a mismatched
version. The symptoms are:

```
torch: 2.13.0+cpu | CUDA available: False
AttributeError: partially initialized module 'torchvision' has no attribute
'extension' (most likely due to a circular import)
```

Fix by reinstalling the matched CUDA pair afterwards, without dependencies so
pip does not resolve them away again:

```bash
pip install --force-reinstall --no-deps \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch, spineps; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.6.0+cu124 True
```

---

## Model weights

Weights download automatically on first run, from the project's GitHub
releases. On a slow connection this is the longest step — start it early. A
custom location can be set with the `SPINEPS_SEGMENTOR_MODELS` environment
variable.

---

## Running it

Direct CLI, on a sagittal T2-weighted NIfTI:

```bash
spineps sample -i path/to/scan_T2w.nii.gz -model_semantic t2w -model_instance instance
```

From this project (wraps the CLI, loads the masks back, renders the overlay):

```bash
python src/spineps_runner.py "path/to/scan_T2w.nii.gz"
```

`src/spineps_runner.py` exposes:

| Function | Purpose |
|---|---|
| `available()` | whether the environment and CLI are installed |
| `run(nifti_path)` | run inference, return paths to the semantic and instance masks |
| `instance_slice(path)` | mid-sagittal slice of the instance mask as a label map |
| `overlay(img, inst)` | per-vertebra colouring with each vertebra numbered |

---

## Input requirements

- **Sagittal T2-weighted**, 3D NIfTI (`.nii` / `.nii.gz`).
- Single-slice exports (the `_i00001`-style files in the hackathon data) are
  not suitable — SPINEPS needs the stack. Run
  `python src/dataset_splits_report.py` or the check in the session notes to
  list which files qualify; **51 of the supplied spine volumes do.**

---

## Reference

Möller et al., *SPINEPS — automatic whole spine segmentation of T2-weighted MR
images using a two-phase approach to multi-class semantic and instance
segmentation*, **European Radiology** (2025). Apache-2.0.
Reported accuracy: Dice **0.92** vertebrae, **0.967** intervertebral discs,
**0.958** spinal canal.
