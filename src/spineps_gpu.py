"""
spineps_gpu.py  --  run SPINEPS entirely on the GPU, by forcing spine cropping.

WHY THIS EXISTS
---------------
The SPINEPS CLI cannot complete the instance phase on a 6 GB card: it forwards
the whole volume at once and needs ~12.4 GiB. Two earlier hypotheses were tested
and both were wrong:

  * "the scan's in-plane resolution is too high" -- downsampling 3.6x produced a
    byte-identical 12.44 GiB failure, because the instance phase does not read
    the input scan at all. It reads the semantic mask SPINEPS just wrote, which
    is stored in SPINEPS's own internal ~0.75 mm space regardless of input.
  * "it must be sliding-window, so it should already be bounded" -- it is not.

The actual cause is a default the CLI does not expose. `process_img_nii` takes
`auto_crop_to_spine="auto"`, which crops the volume to the spine before
inference -- but only when `max_resolution <= auto_crop_when_max_res_leq` (1.2 mm).
Our sagittal scans have 4.4 mm slice thickness, so the guard silently disabled
cropping and the model was handed the entire field of view: head, neck, chest
and all.

Forcing `auto_crop_to_spine=True` restores the crop. The volume that reaches the
instance model becomes the spine column instead of the whole torso, which is
what makes it fit in 6 GB.

This is not a workaround that costs accuracy. The crop is SPINEPS's own,
computed by its own spine locator, and the discarded region contains no spine by
construction. It is also better aligned with the task, which concerns the
lumbo-sacral spine rather than the whole body.

Usage:
    python src/spineps_gpu.py <scan.nii.gz> [--cpu]
"""

import os
import sys
from pathlib import Path


def run(input_path: str, use_cpu: bool = False, derivative_name: str = "derivatives_seg",
        override: bool = False) -> dict:
    """Run both SPINEPS phases with spine cropping forced on.

    Returns a dict with the produced mask paths, or an error description.
    """
    from spineps.get_models import (get_semantic_model, get_instance_model,
                                    get_labeling_model)
    from spineps.seg_run import process_img_nii
    from TPTBox import BIDS_FILE

    p = Path(input_path).absolute()
    if not p.is_file():
        return {"ok": False, "error": f"not a file: {p}"}

    model_semantic = get_semantic_model("t2w", use_cpu=use_cpu).load()
    model_instance = get_instance_model("instance", use_cpu=use_cpu).load()
    try:
        model_labeling = get_labeling_model("t2w_labeling", use_cpu=use_cpu).load()
    except Exception:
        model_labeling = None

    bids = BIDS_FILE(str(p), dataset=str(p.parent), verbose=False)

    process_img_nii(
        img_ref=bids,
        model_semantic=model_semantic,
        model_instance=model_instance,
        model_labeling=model_labeling,
        derivative_name=derivative_name,
        # the whole point of this module -- see the module docstring
        auto_crop_to_spine=True,
        override_semantic=override,
        override_instance=override,
        verbose=False,
    )

    out = {"ok": False, "semantic": None, "instance": None}
    root = p.parent / derivative_name
    for r, _d, files in os.walk(root):
        for f in files:
            fl = f.lower()
            if not fl.endswith(".nii.gz") or p.stem.split(".")[0].lower() not in fl:
                continue
            full = os.path.join(r, f)
            if "seg-vert" in fl:
                out["instance"] = full
            elif "seg-spine" in fl:
                out["semantic"] = full
    out["ok"] = out["instance"] is not None or out["semantic"] is not None
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/spineps_gpu.py <scan.nii.gz> [--cpu]")
        raise SystemExit(2)
    import time
    t0 = time.time()
    r = run(sys.argv[1], use_cpu="--cpu" in sys.argv, override="--override" in sys.argv)
    r["seconds"] = round(time.time() - t0, 1)
    print(r)
