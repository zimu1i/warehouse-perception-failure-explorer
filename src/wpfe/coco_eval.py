"""Standard COCO metrics via pycocotools, computed from our manifest/prediction tables.

This is the established evaluator for AP. The failure-event logic for review
selection (one operating threshold, one-to-one matching) lives separately.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# LOCO images hold up to 345 GT boxes, so COCO's default cap of 100 detections
# per image would truncate recall. The last value is the one used for AP.
MAX_DETS = [10, 100, 500]


def coco_metrics(images: pd.DataFrame, gt: pd.DataFrame, preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Per-class AP@[.5:.95], AP50 and recall@IoU0.5 (max 500 dets), plus an 'all' row."""
    img_ids = {iid: i for i, iid in enumerate(images["image_id"])}
    cat_ids = {c: i + 1 for i, c in enumerate(classes)}

    gt_json = {
        "images": [{"id": img_ids[r.image_id], "width": int(r.width), "height": int(r.height)} for r in images.itertuples()],
        "categories": [{"id": i, "name": c} for c, i in cat_ids.items()],
        "annotations": [
            {
                "id": k + 1,
                "image_id": img_ids[r.image_id],
                "category_id": cat_ids[r.class_name],
                "bbox": [r.x_min, r.y_min, r.x_max - r.x_min, r.y_max - r.y_min],
                "area": (r.x_max - r.x_min) * (r.y_max - r.y_min),
                "iscrowd": int(r.iscrowd),
            }
            for k, r in enumerate(gt.itertuples())
        ],
    }
    dets = preds[preds["class_name"].isin(cat_ids)]
    dt_list = [
        {
            "image_id": img_ids[r.image_id],
            "category_id": cat_ids[r.class_name],
            "bbox": [r.x_min, r.y_min, r.x_max - r.x_min, r.y_max - r.y_min],
            "score": r.score,
        }
        for r in dets.itertuples()
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_json
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_list) if dt_list else None
    if coco_dt is None:
        return pd.DataFrame([{"class_name": c, "n_gt": int((gt["class_name"] == c).sum()), "ap": 0.0, "ap50": 0.0, "recall50": 0.0} for c in classes])

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.maxDets = MAX_DETS
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()

    precision = ev.eval["precision"]  # [iou, recall_pts, class, area, maxdet]
    recall = ev.eval["recall"]  # [iou, class, area, maxdet]
    rows = []
    for k, c in enumerate(classes):
        p_all = precision[:, :, k, 0, -1]
        p50 = precision[0, :, k, 0, -1]
        r50 = recall[0, k, 0, -1]
        rows.append(
            {
                "class_name": c,
                "n_gt": int((gt["class_name"] == c).sum()),
                "ap": _mean_valid(p_all),
                "ap50": _mean_valid(p50),
                "recall50": float(r50) if r50 > -1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _mean_valid(a: np.ndarray) -> float:
    valid = a[a > -1]
    return float(valid.mean()) if valid.size else float("nan")
