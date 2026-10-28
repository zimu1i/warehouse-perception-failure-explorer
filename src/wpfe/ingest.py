"""Build the image and annotation manifests from the raw LOCO release.

Raw files are only read. Outputs go to `<artifacts_dir>/manifest/`.

Box convention (all artifacts): absolute pixel coordinates, origin at the
image's top-left corner, continuous `xyxy` with x_max = x + w and
y_max = y + h from COCO's `[x, y, w, h]`. A box covering a whole 640x480
image is (0, 0, 640, 480).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image

from wpfe.config import Config


def stable_image_id(rel_path: str) -> str:
    """ID derived from the release-relative path, so it survives re-ingestion.

    Not derived from content: exact-duplicate files must keep distinct IDs
    so the duplicate check can report them.
    """
    return "img_" + hashlib.sha1(rel_path.encode()).hexdigest()[:12]


def parse_rel_path(rel_path: str) -> dict:
    """Extract subset, capture session and camera from a LOCO image path.

    `dataset/subset-4/2020-01-16_17/cam2/1579163807.91.jpg`
      -> subset 4, session "2020-01-16_17", camera "cam2", timestamp 1579163807.91
    subset-1 has no session/camera folders, so both are None there.
    """
    parts = PurePosixPath(rel_path).parts
    subset = int(parts[1].removeprefix("subset-"))
    session = parts[2] if len(parts) > 3 else None
    camera = "/".join(parts[3:-1]) if len(parts) > 4 else None
    try:
        timestamp = float(PurePosixPath(rel_path).stem)
    except ValueError:
        timestamp = None
    return {"subset": subset, "session": session, "camera": camera, "timestamp": timestamp}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_split(subset: int, cfg: Config) -> str:
    if subset in cfg.dev_subsets:
        return "dev"
    if subset in cfg.review_subsets:
        return "review"
    raise ValueError(f"subset {subset} is not assigned to a split in the config")


def build_images(cfg: Config, coco: dict) -> pd.DataFrame:
    coco_by_path = {img["path"].lstrip("/"): img for img in coco["images"]}
    rows = []
    for path in sorted((cfg.raw_root / "dataset").rglob("*.jpg")):
        rel = path.relative_to(cfg.raw_root).as_posix()
        meta = parse_rel_path(rel)
        with Image.open(path) as im:
            width, height = im.size
        coco_img = coco_by_path.get(rel)
        rows.append(
            {
                "image_id": stable_image_id(rel),
                "rel_path": rel,
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
                "source_release": cfg.release,
                **meta,
                "split": assign_split(meta["subset"], cfg),
                # Decision D2: images absent from the JSON are kept but never scored.
                "annotation_status": "annotated" if coco_img else "unannotated",
                "coco_image_id": coco_img["id"] if coco_img else None,
                "coco_width": coco_img["width"] if coco_img else None,
                "coco_height": coco_img["height"] if coco_img else None,
            }
        )
    df = pd.DataFrame(rows)
    df["coco_image_id"] = df["coco_image_id"].astype("Int64")
    df["coco_width"] = df["coco_width"].astype("Int64")
    df["coco_height"] = df["coco_height"].astype("Int64")

    missing = set(coco_by_path) - set(df["rel_path"])
    if missing:
        raise FileNotFoundError(f"{len(missing)} annotated images missing on disk, e.g. {sorted(missing)[:3]}")
    return df


def build_annotations(cfg: Config, coco: dict, images: pd.DataFrame, source_version: str) -> pd.DataFrame:
    class_map = cfg.class_map
    unknown = {c["id"] for c in coco["categories"]} - set(class_map)
    if unknown:
        raise ValueError(f"categories not in config class map: {sorted(unknown)}")

    image_id_by_coco = dict(
        images.dropna(subset=["coco_image_id"])[["coco_image_id", "image_id"]].itertuples(index=False)
    )
    rows = []
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        rows.append(
            {
                "ann_id": f"ann_{ann['id']}",
                "image_id": image_id_by_coco[ann["image_id"]],
                "coco_ann_id": ann["id"],
                "coco_category_id": ann["category_id"],
                "class_name": class_map[ann["category_id"]],
                # Raw values, not clipped. Validation decides what is wrong with them.
                "x_min": float(x),
                "y_min": float(y),
                "x_max": float(x) + float(w),
                "y_max": float(y) + float(h),
                "iscrowd": bool(ann.get("iscrowd", False)),
                "source_annotation_version": source_version,
            }
        )
    return pd.DataFrame(rows)


def run_ingest(cfg: Config) -> dict:
    out_dir = cfg.artifacts_dir / "manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.annotation_path) as f:
        coco = json.load(f)

    images = build_images(cfg, coco)
    annotations = build_annotations(cfg, coco, images, source_version=cfg.annotation_path.stem)

    images.to_parquet(out_dir / "images.parquet", index=False)
    annotations.to_parquet(out_dir / "annotations.parquet", index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": cfg.hash(),
        "annotation_file_sha256": sha256_file(cfg.annotation_path),
        "n_image_files": len(images),
        "n_annotated_images": int((images["annotation_status"] == "annotated").sum()),
        "n_annotations": len(annotations),
        "images_by_split": images.groupby(["split", "annotation_status"]).size().rename("n").reset_index().to_dict("records"),
    }
    (out_dir / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
