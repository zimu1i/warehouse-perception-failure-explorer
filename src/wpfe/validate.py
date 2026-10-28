"""Data-quality checks over the manifests. Writes `quality_findings.parquet`.

Severity levels separate three kinds of finding (brief section 8):
  - "error":   structural corruption; the record cannot be used as-is.
  - "warning": potential annotation problem; keep, but review.
  - "info":    legitimately difficult example; keep and flag, never remove.
Findings never modify raw data. `disposition` records what we decided to do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from wpfe.config import Config

SEVERITIES = ("error", "warning", "info")
OVERSHOOT_WARN_FRAC = 0.01  # D4: more than 1% of the axis -> warning
OVERSHOOT_ERROR_FRAC = 0.10  # D4: more than 10% of the axis -> error


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    evidence: str

def check_box(
    x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int
) -> list[Finding]:
    """Return every problem found with one ground-truth box (empty list = OK).

    Coordinates follow the manifest convention: continuous xyxy pixels, origin
    top-left, so a full-image box is (0, 0, width, height).

    Rules (thresholds from decision D4):
      - non_finite_coords (error): any coordinate is NaN or infinite. Returns early.
      - non_positive_size (error): x_max <= x_min or y_max <= y_min. Returns early.
      - box_fully_outside (error): no overlap with the image.
      - box_overshoot_major (error): overshoot > OVERSHOOT_ERROR_FRAC of an axis.
      - box_overshoot_minor (warning): overshoot > OVERSHOOT_WARN_FRAC of an axis.
    Small boxes are not a quality finding; object size is an evaluation slice.
    """

    # _is_finite has input signature *values, thus can input as many as wanted
    if not _is_finite(x_min, y_min, x_max, y_max):
        return [Finding(rule="non_finite_coords",
                       severity="error",
                       evidence=f"coords=({x_min}, {y_min}, {x_max}, {y_max})",)]
    if not (x_min < x_max) or not (y_min < y_max):
        return [Finding(rule="non_positive_size",
                       severity="error",
                       evidence=f"x_max - x_min = {x_max - x_min}, y_max - y_min = {y_max - y_min}",)]

    over_x = max(0 - x_min, x_max - width, 0)
    over_y = max(0 - y_min, y_max - height, 0)
    frac_x = over_x/width
    frac_y = over_y/height
    frac = max(frac_x, frac_y)
    
    findings = []
    if (x_min >= width or x_max <= 0 or y_min >= height or y_max <= 0):
        findings.append(Finding(rule="box_fully_outside",
                               severity="error",
                               evidence=f"box ({x_min}, {y_min}, {x_max}, {y_max}) vs image {width}x{height}"))
    elif frac > OVERSHOOT_ERROR_FRAC:
        findings.append(Finding(rule="box_overshoot_major",
                               severity="error",
                               evidence=f"overshoot {frac:.1%} (x {over_x}px of {width}, y {over_y}px of {height})"))
    elif frac > OVERSHOOT_WARN_FRAC:
        findings.append(Finding(rule="box_overshoot_minor",
                               severity="warning",
                               evidence=f"overshoot {frac:.1%} (x {over_x}px of {width}, y {over_y}px of {height})"))
    return findings

def check_images(images: pd.DataFrame) -> pd.DataFrame:
    """Image-level structural checks: duplicate IDs, size mismatch vs JSON, exact duplicate content."""
    rows = []
    for image_id in images.loc[images["image_id"].duplicated(), "image_id"]:
        rows.append(("image", image_id, "duplicate_image_id", "error", "image_id appears more than once"))

    annotated = images[images["annotation_status"] == "annotated"]
    mismatch = annotated[(annotated["width"] != annotated["coco_width"]) | (annotated["height"] != annotated["coco_height"])]
    for r in mismatch.itertuples():
        rows.append(("image", r.image_id, "size_mismatch", "error",
                     f"file {r.width}x{r.height} vs json {r.coco_width}x{r.coco_height}"))

    for _, group in images[images["sha256"].duplicated(keep=False)].groupby("sha256"):
        ids = ",".join(group["image_id"])
        for image_id in group["image_id"]:
            rows.append(("image", image_id, "exact_duplicate_content", "warning", f"same sha256 as {ids}"))

    return pd.DataFrame(rows, columns=["ref_type", "ref_id", "rule", "severity", "evidence"])


def check_annotations(annotations: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ann_id in annotations.loc[annotations["ann_id"].duplicated(), "ann_id"]:
        rows.append(("annotation", ann_id, "duplicate_ann_id", "error", "ann_id appears more than once"))

    sizes = images.set_index("image_id")[["width", "height"]]
    orphan = ~annotations["image_id"].isin(sizes.index)
    for ann_id in annotations.loc[orphan, "ann_id"]:
        rows.append(("annotation", ann_id, "orphan_annotation", "error", "image_id not in image manifest"))

    joined = annotations[~orphan].join(sizes, on="image_id")
    for r in joined.itertuples():
        for f in check_box(r.x_min, r.y_min, r.x_max, r.y_max, r.width, r.height):
            assert f.severity in SEVERITIES, f"unknown severity {f.severity!r} from rule {f.rule}"
            rows.append(("annotation", r.ann_id, f.rule, f.severity, f.evidence))

    return pd.DataFrame(rows, columns=["ref_type", "ref_id", "rule", "severity", "evidence"])


def run_validate(cfg: Config) -> pd.DataFrame:
    manifest = cfg.artifacts_dir / "manifest"
    images = pd.read_parquet(manifest / "images.parquet")
    annotations = pd.read_parquet(manifest / "annotations.parquet")

    findings = pd.concat([check_images(images), check_annotations(annotations, images)], ignore_index=True)
    # Default disposition; errors get excluded from scoring, everything else is kept.
    findings["disposition"] = findings["severity"].map(
        {"error": "exclude", "warning": "keep_flagged", "info": "keep_flagged"}
    )
    findings.to_parquet(manifest / "quality_findings.parquet", index=False)
    return findings


def _is_finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)
