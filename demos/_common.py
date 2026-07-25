"""
Shared helpers for the demo scripts.

Every demo script in this folder is standalone and runnable:

    python demos/01_brain_he.py

They exist so any single stage can be shown on demand -- a judge can ask
"show me just CLAHE" or "show me just the tumour model" and there is one file
that does exactly that, prints the real numbers, and saves the picture.
"""

import os
import sys

# Every path in the project is relative to the repo root, so anchor there
# regardless of where the script was launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402

OUT = "outputs/demo_runs"
os.makedirs(OUT, exist_ok=True)

BRAIN = "showcase/for_enhancement/BRAIN_BraTS20_Training_004_flair.nii"
SPINE = "showcase/for_spineps/SPINE_SP5_NORM_T2.nii.gz"
SPINE_PATH = "showcase/for_spineps/SPINE_SP11_PATH_T2.nii.gz"


def head(title, asked=""):
    print("\n" + "=" * 74)
    print(title)
    if asked:
        print("What the brief asked for: " + asked)
    print("=" * 74)


def kv(k, v):
    print(f"  {k:<34} {v}")


def load_slice(path=None, region="brain"):
    """Middle usable slice from a NIfTI, normalised to [0,1]."""
    from nifti_utils import load_volume
    from enhancement_dataset import extract_training_slices
    p = path or (BRAIN if region == "brain" else SPINE)
    sls = extract_training_slices(load_volume(p))
    if not sls:
        raise SystemExit(f"no usable slices in {p}")
    return sls[len(sls) // 2], p


def save(name, img, gray=True):
    """Write a [0,1] float or BGR uint8 image into outputs/demo_runs/."""
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8) if gray else img
    p = os.path.join(OUT, name)
    cv2.imwrite(p, arr)
    print(f"  -> saved {p}")
    return p


def side_by_side(images, labels, name, height=320):
    """Stack panels horizontally with labels, so one PNG tells the story."""
    panels = []
    for im, lab in zip(images, labels):
        a = (np.clip(im, 0, 1) * 255).astype(np.uint8) if im.ndim == 2 else im
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        h = height
        w = int(a.shape[1] * h / a.shape[0])
        a = cv2.resize(a, (w, h))
        cv2.rectangle(a, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(a, lab[:30], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(a)
    return save(name, np.hstack(panels), gray=False)


def noise_level(img01):
    """High-frequency residual as a cheap noise estimate (same as Stage 1)."""
    lap = cv2.Laplacian((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.CV_64F)
    return float(lap.std() / 255.0)
