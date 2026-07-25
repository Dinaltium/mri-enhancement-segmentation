"""
spineps_runner.py   --   per-vertebra instance segmentation via SPINEPS

SPINEPS (Möller et al., European Radiology 2025, Apache-2.0) is the first
publicly available whole-spine segmentation model for sagittal T2-weighted MRI.
It produces exactly what the reference literature in the problem statement
shows: a semantic mask of 14 spinal structures AND an instance mask separating
individual vertebrae and intervertebral discs. Reported Dice: 0.92 vertebrae,
0.967 discs, 0.958 spinal canal.

IMPORTANT — provenance, state this in the report:
SPINEPS ships weights pretrained on the public SPIDER dataset and the German
National Cohort. We do not train it and we supply it no annotations, but its
weights do encode external data. Whether that is permitted under the "no
external data for spine" rule is a question for the organisers. It is therefore
kept SEPARATE from our own annotation-free pipeline and reported as a
reference/upper-bound comparison unless explicitly cleared for primary use.

This module shells out to the installed `spineps` CLI in its own Python 3.11
environment (SPINEPS requires 3.11; the project runs on 3.10), then loads the
resulting masks back for display.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

# the dedicated environment created for SPINEPS
SPINEPS_PY = os.path.expanduser(
    r"~/.conda/envs/spineps/python.exe").replace("\\", "/")
SPINEPS_ENV_SCRIPTS = os.path.expanduser(
    r"~/.conda/envs/spineps/Scripts").replace("\\", "/")


def available() -> bool:
    """True if the SPINEPS environment and CLI are installed."""
    if not os.path.exists(SPINEPS_PY):
        return False
    try:
        r = subprocess.run([SPINEPS_PY, "-c", "import spineps"],
                           capture_output=True, timeout=90)
        return r.returncode == 0
    except Exception:
        return False


def run(nifti_path: str, out_dir: str = None, timeout: int = 1800) -> dict:
    """Run SPINEPS on one sagittal T2w NIfTI. Returns paths to the produced
    semantic and instance masks, or an error description."""
    if not available():
        return {"ok": False, "error": "SPINEPS environment not installed"}

    work = out_dir or tempfile.mkdtemp(prefix="spineps_")
    os.makedirs(work, exist_ok=True)
    # SPINEPS writes derivatives next to the input, so copy the scan in first
    local = os.path.join(work, os.path.basename(nifti_path))
    if os.path.abspath(local) != os.path.abspath(nifti_path):
        shutil.copy(nifti_path, local)

    cmd = [SPINEPS_PY, "-m", "spineps", "sample",
           "-i", local, "-model_semantic", "t2w", "-model_instance", "instance"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"SPINEPS timed out after {timeout}s"}

    sem = inst = None
    for root, _dirs, files in os.walk(work):
        for f in files:
            fl = f.lower()
            if not (fl.endswith(".nii") or fl.endswith(".nii.gz")):
                continue
            p = os.path.join(root, f)
            if "seg-vert" in fl or "instance" in fl:
                inst = p
            elif "seg-spine" in fl or "semantic" in fl or "subreg" in fl:
                sem = p
    ok = inst is not None or sem is not None
    return {"ok": ok, "work_dir": work, "semantic": sem, "instance": inst,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-1500:], "stderr": proc.stderr[-1500:],
            "error": None if ok else "no mask produced — see stderr"}


# What each value in the SPINEPS semantic mask means (TPTBox vert_constants).
# These are structure TYPES -- distinct from the instance mask's per-vertebra
# ID numbers, which say "separate bone", not "which structure".
SPINEPS_LABELS_HELP = {
    41: "Arcus vertebrae (vertebral arch)", 42: "Spinous process",
    43: "Costal process, left", 44: "Costal process, right",
    45: "Superior articular process, left", 46: "Superior articular process, right",
    47: "Inferior articular process, left", 48: "Inferior articular process, right",
    49: "Vertebral body (corpus)", 60: "Spinal cord", 61: "Spinal canal",
    62: "Endplate", 100: "Intervertebral disc",
}


def run_semantic_live(nifti_path: str, work_root: str = "outputs/spineps/live",
                      timeout: int = 300, key: str = None) -> dict:
    """Run ONLY the SPINEPS semantic phase, on the GPU, on an uploaded scan.

    Why semantic-only: the instance phase forwards its whole working volume at
    once and needs ~12.4 GiB, which does not fit in 6 GB. It is therefore run
    offline on CPU (401 s) and shown as a precomputed reference. The semantic
    phase does fit, and completes in under a minute, so it CAN run live.

    Why the process is killed rather than allowed to finish: SPINEPS has no
    "semantic only" flag. It writes the semantic mask, then starts the instance
    phase and dies on OOM. Waiting for that adds a minute and an alarming
    traceback for no benefit, so we watch for the mask and stop the process the
    moment it lands. The mask is fully written and post-processed by then --
    SPINEPS saves it as a completed step, not incrementally.
    """
    import glob
    import time

    if not available():
        return {"ok": False, "error": "SPINEPS environment not installed"}

    # Cache on the ORIGINAL filename, not the path we were handed. Uploads
    # arrive as a randomly-named temp file, so keying on that meant the cache
    # never hit and every click re-ran the model for a minute.
    stem = os.path.basename(key or nifti_path)
    for ext in (".nii.gz", ".nii"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    safe = "".join(c for c in stem if c.isalnum())[:32] or "upload"
    work = os.path.join(work_root, safe)
    os.makedirs(work, exist_ok=True)
    local = os.path.join(work, f"sub-{safe}_T2w.nii.gz")
    if os.path.abspath(local) != os.path.abspath(nifti_path):
        shutil.copy(nifti_path, local)

    hit = glob.glob(os.path.join(work, "**", "*seg-spine*.nii.gz"), recursive=True)
    if hit:                                     # already computed for this scan
        return {"ok": True, "semantic": hit[0], "seconds": 0.0, "cached": True}

    env = dict(os.environ)
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    code = ("from spineps.entrypoint import entry_point\n"
            "import sys\n"
            f"sys.argv=['spineps','sample','-i',r'{local}',"
            "'-model_semantic','t2w','-model_instance','instance']\n"
            "entry_point()")
    # Log to a FILE, never to subprocess.PIPE. SPINEPS is very chatty (nnU-Net
    # progress bars); with a pipe nobody drains, the 64 KB OS buffer fills and
    # the child blocks forever on write — which looks exactly like the model
    # hanging, and cost us every run timing out at 300 s.
    logf = open(os.path.join(work, "spineps.log"), "wb")
    proc = subprocess.Popen([SPINEPS_PY, "-u", "-c", code], env=env,
                            stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.time()
    found = None
    try:
        while time.time() - t0 < timeout:
            hit = glob.glob(os.path.join(work, "**", "*seg-spine*.nii.gz"),
                            recursive=True)
            if hit:
                # give the writer a moment to flush before we kill the process
                time.sleep(1.5)
                found = hit[0]
                break
            if proc.poll() is not None:
                break
            time.sleep(1.0)
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        try:
            logf.close()
        except Exception:
            pass

    if not found:
        return {"ok": False, "error": f"no semantic mask after {int(time.time()-t0)}s"}
    return {"ok": True, "semantic": found, "seconds": round(time.time() - t0, 1),
            "cached": False}


def mask_in_scan_space(mask_path: str, scan_path: str) -> np.ndarray:
    """Resample a SPINEPS mask back into the ORIGINAL scan's voxel grid.

    Necessary because SPINEPS does not return the mask on the input grid: it
    resamples to ~0.8 mm isotropic AND reorients to a canonical axis order
    before inference (measured: a 512x512x12 input came back as 503x503x53).
    Matching masks to the scan by array index therefore produces an overlay
    that is visibly offset and rotated — that failure is a rendering error,
    not a segmentation error.

    The affine of each image maps its voxel indices to the same physical
    (scanner) space, so composing the two affines maps mask voxels onto scan
    voxels correctly regardless of resolution or orientation. Nearest-neighbour
    only: these are integer labels, interpolation would invent new ones.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to
    m, s = nib.load(mask_path), nib.load(scan_path)
    r = resample_from_to(m, (s.shape[:3], s.affine), order=0, cval=0)
    return np.asarray(r.dataobj).astype(np.int32)


def instance_slice(instance_path: str, z: int = None,
                   scan_path: str = None) -> np.ndarray:
    """Mid-sagittal slice of the instance mask, as an integer label map.

    Pass `scan_path` whenever the slice will be drawn over the original scan —
    without it the mask stays in SPINEPS's own resampled space and will not
    line up. See mask_in_scan_space().
    """
    import nibabel as nib
    if scan_path:
        vol = mask_in_scan_space(instance_path, scan_path)
    else:
        vol = nib.load(instance_path).get_fdata()
    if vol.ndim == 2:
        return vol.astype(np.int32)
    # pick the slice carrying the most labelled voxels
    counts = [(vol[..., k] > 0).sum() for k in range(vol.shape[2])]
    k = int(np.argmax(counts)) if z is None else z
    return vol[..., k].astype(np.int32)


PALETTE = np.array([
    [66, 135, 245], [80, 200, 120], [240, 170, 60], [220, 90, 90],
    [170, 110, 220], [60, 200, 210], [235, 130, 190], [150, 160, 90],
    [110, 140, 200], [200, 200, 110], [130, 200, 160], [230, 150, 120],
], dtype=np.uint8)


def overlay(img01: np.ndarray, inst: np.ndarray) -> np.ndarray:
    """Per-vertebra colouring with the label number drawn on each body."""
    import cv2
    if inst.shape != img01.shape:
        inst = cv2.resize(inst.astype(np.int32), (img01.shape[1], img01.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    base = cv2.cvtColor((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    col = np.zeros_like(base)
    ids = [int(u) for u in np.unique(inst) if u > 0]
    for k, u in enumerate(ids):
        col[inst == u] = PALETTE[k % len(PALETTE)][::-1]
    out = cv2.addWeighted(base, 1.0, col, 0.55, 0)
    for u in ids:
        ys, xs = np.nonzero(inst == u)
        if xs.size:
            cv2.putText(out, str(u), (int(xs.mean()) - 5, int(ys.mean()) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    import sys
    print("SPINEPS available:", available())
    if len(sys.argv) > 1:
        r = run(sys.argv[1])
        print({k: v for k, v in r.items() if k not in ("stdout", "stderr")})
        if not r["ok"]:
            print("--- stderr ---"); print(r.get("stderr", "")[:1200])
