"""
coco_export.py

Converts predicted ROI masks into COCO-format JSON (a required hackathon
deliverable). Works for both tracks:
    - Brain : per-class tumour masks from the trained segmentation model
              (categories: necrotic/non-enhancing, edema, enhancing)
    - Spine : exploratory unsupervised ROI clusters from spine_pipeline
              (categories: roi_cluster_1..k, honestly labelled as
              unsupervised intensity clusters, not validated anatomy)

Masks are stored as COCO RLE (run-length encoding) via pycocotools.mask -
compact and lossless, the standard way to put pixel masks in COCO JSON.

Input: a directory of integer label-map .npy files (0 = background, 1..N =
classes), each with a sibling image PNG for the file_name/size reference.
This is exactly what spine_pipeline.py (*_roi.npy) and inference_report.py
(*_seg.npy) write, so this stage just harvests their outputs.

Usage:
    python coco_export.py --input_dir outputs/spine_pathological --kind spine_roi \
        --out results/spine_coco.json
    python coco_export.py --input_dir outputs/brain_offline --kind brain_seg \
        --out results/brain_coco.json
"""

import argparse
import glob
import json
import os

import numpy as np
from pycocotools import mask as maskUtils


BRAIN_CATEGORIES = {
    1: "necrotic_non_enhancing_core",
    2: "edema",
    3: "enhancing_tumor",
}


def spine_categories(k: int) -> dict:
    # honest naming: these are unsupervised intensity clusters, dark -> bright
    return {i: f"roi_cluster_{i}_unsupervised" for i in range(1, k + 1)}


def mask_to_coco_annotations(label_map: np.ndarray, image_id: int,
                             start_ann_id: int, category_ids: list[int]) -> list[dict]:
    """One annotation per present class (binary mask -> RLE + bbox + area)."""
    anns = []
    ann_id = start_ann_id
    for cat in category_ids:
        binary = np.asfortranarray((label_map == cat).astype(np.uint8))
        if binary.sum() == 0:
            continue
        rle = maskUtils.encode(binary)
        rle["counts"] = rle["counts"].decode("ascii")  # bytes -> str for JSON
        area = float(maskUtils.area({"counts": rle["counts"].encode("ascii"),
                                     "size": rle["size"]}))
        bbox = maskUtils.toBbox({"counts": rle["counts"].encode("ascii"),
                                 "size": rle["size"]}).tolist()
        anns.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat,
            "segmentation": rle,       # COCO RLE
            "area": area,
            "bbox": bbox,              # [x, y, w, h]
            "iscrowd": 0,
        })
        ann_id += 1
    return anns


def build_coco(input_dir: str, kind: str, dataset_name: str) -> dict:
    npy_files = sorted(glob.glob(os.path.join(input_dir, "*_roi.npy"))
                       + glob.glob(os.path.join(input_dir, "*_seg.npy")))
    if not npy_files:
        raise FileNotFoundError(f"no *_roi.npy / *_seg.npy label maps in {input_dir}")

    # decide category set
    if kind == "brain_seg":
        categories = BRAIN_CATEGORIES
    else:
        maxk = 1
        for f in npy_files:
            arr = np.load(f)
            maxk = max(maxk, int(arr.max()))
        categories = spine_categories(maxk)
    category_ids = sorted(categories.keys())

    images, annotations = [], []
    ann_id = 1
    for image_id, npy in enumerate(npy_files, start=1):
        label_map = np.load(npy)
        h, w = label_map.shape
        base = os.path.basename(npy).replace("_roi.npy", "").replace("_seg.npy", "")
        # prefer the enhanced/orig png as the referenced image file
        png = None
        for suffix in ("_clahe.png", "_orig.png", ".png"):
            cand = os.path.join(input_dir, base + suffix)
            if os.path.exists(cand):
                png = os.path.basename(cand)
                break
        images.append({"id": image_id, "file_name": png or (base + ".png"),
                       "width": int(w), "height": int(h)})
        annotations.extend(
            mask_to_coco_annotations(label_map, image_id, ann_id, category_ids)
        )
        ann_id = (annotations[-1]["id"] + 1) if annotations else ann_id

    return {
        "info": {"description": f"{dataset_name} ROI segmentation (MedhaDrishti hackathon)",
                 "kind": kind},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name} for cid, name in categories.items()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--kind", choices=["spine_roi", "brain_seg"], default="spine_roi")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    name = args.dataset_name or os.path.basename(args.input_dir.rstrip("/\\"))
    coco = build_coco(args.input_dir, args.kind, name)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(coco, f)
    print(f"[coco_export] {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories -> {args.out}")


if __name__ == "__main__":
    main()
