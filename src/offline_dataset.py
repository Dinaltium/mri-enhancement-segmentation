"""
offline_dataset.py

Discovery + sub-modality classification for the OFFLINE hackathon datasets
(Brain Normal/Pathological, Spine Normal/Pathological). Unlike BraTS these
are raw scanner exports with messy, inconsistent filenames, so we can't use
a fixed template like nifti_utils.find_brats_cases. Two realities to handle:

  1. Brain-Pathological (BRP1..BRP10): flat folders, BraTS-ish names
     (007_flair.nii, t1ce.nii, 09_t2.nii, 024__t2.nii) - inconsistent
     prefixes/typos, so we keyword-match, not template-match.

  2. Brain-Normal (S1..S10) and Spine (SP*): nested "2D MRI"/"3D MRI"
     subfolders, cryptic Philips sequence names (eT1W_SE, eFLAIR_longTR_SPIR,
     sT1W_3D_TFE_PRE_GD, eSTIR_longTE, eT2W_TSE_DRIVE_HR) plus out-of-scope
     sequences (DWI, ADC, SWI, BOLD, survey/localizer) we must exclude.

The classifier maps each .nii/.nii.gz file to one of:
    T1, T1c, T2, FLAIR, STIR   (or None = out of scope / unrecognised)

This is best-effort and intentionally transparent - classify_modality()
returns the reason, so dataset_stats can print an audit of every file's
assigned bucket for the report (honest about ambiguous cases).
"""

import os

# sequences that are NOT one of the in-scope structural sub-modalities
_EXCLUDE = [
    "dwi", "adc", "_dti", "swi", "bold", "perf", "asl",
    "survey", "localizer", "localiser", "scout", "mobiview",
    "_reg_", "reg_-", "_ven_", "venbold", "b1map", "b0map",
    "_iso_", "dwi_iso",
]


def classify_modality(filename: str) -> tuple[str | None, str]:
    """Returns (modality, reason). modality is one of
    {T1, T1c, T2, FLAIR, STIR} or None (out of scope)."""
    f = filename.lower()

    for ex in _EXCLUDE:
        if ex in f:
            return None, f"excluded (matched '{ex}')"

    # STIR is unambiguous
    if "stir" in f:
        return "STIR", "matched 'stir'"

    # FLAIR is unambiguous, check before T1/T2 keyword hunt
    if "flair" in f:
        return "FLAIR", "matched 'flair'"

    is_t1 = ("t1w" in f or "t1_" in f or "_t1" in f or "t1c" in f
             or f.startswith("t1") or "_t1w" in f or "st1w" in f)
    is_t2 = ("t2w" in f or "t2_" in f or "_t2" in f or f.startswith("t2")
             or "_t2w" in f)

    # explicit contrast tag
    if "t1ce" in f or "t1c_" in f:
        return "T1c", "matched 't1ce'/'t1c'"

    # gadolinium markers => contrast T1, but "PRE_GD"/"PRE_GADO" is pre-contrast
    def _has_contrast(s: str) -> bool:
        pre = ("pre_gd" in s or "pregd" in s or "pre_gado" in s or "pregado" in s)
        return (not pre) and ("gd" in s or "gado" in s or "post_gd" in s or "postgd" in s)

    if is_t1 and not is_t2:
        if _has_contrast(f):
            return "T1c", "T1 + gadolinium marker"
        return "T1", "matched T1 keyword"
    if is_t2 and not is_t1:
        return "T2", "matched T2 keyword"
    if is_t1 and is_t2:
        # ambiguous filename mentioning both - fall back to whichever appears first
        return ("T1", "ambiguous T1/T2, T1 first") if f.find("t1") < f.find("t2") \
            else ("T2", "ambiguous T1/T2, T2 first")

    return None, "no modality keyword matched"


def find_nifti_files(root: str) -> list[str]:
    """Recursively find every .nii/.nii.gz under root."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def find_offline_cases(root: str) -> list[str]:
    """Each immediate subfolder of `root` is one patient case
    (S1.., BRP1.., SP1..). Returns sorted case dir paths."""
    if not os.path.isdir(root):
        return []
    cases = [os.path.join(root, d) for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d))]
    return sorted(cases)


def classify_case_files(case_dir: str) -> dict:
    """Maps every nifti file in a case to its modality. Returns
    {modality: [paths]} plus an 'unclassified' bucket, and an audit list
    of (relpath, modality, reason) for reporting."""
    buckets: dict[str, list[str]] = {}
    audit = []
    for path in find_nifti_files(case_dir):
        mod, reason = classify_modality(os.path.basename(path))
        key = mod if mod is not None else "unclassified"
        buckets.setdefault(key, []).append(path)
        audit.append((os.path.relpath(path, case_dir), mod, reason))
    return {"buckets": buckets, "audit": audit}


# canonical registry of the four offline dataset roots, relative to project dir
OFFLINE_ROOTS = {
    "brain_normal": os.path.join("Brain DATASETS", "Normal brain Datasets"),
    "brain_pathological": os.path.join("Brain DATASETS", "Pathological brain MRI Datasets"),
    "spine_normal": os.path.join("Spine DATASETS", "Normal Spine MRI Datasets"),
    "spine_pathological": os.path.join("Spine DATASETS", "Pathological Spine MRI Datasets"),
}
