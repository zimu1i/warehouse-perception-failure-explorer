"""Box overlap and prediction-to-ground-truth matching for the failure event (D1).

COCO AP comes from pycocotools (coco_eval.py). This module is the simpler,
single-operating-point view used to decide whether an image is a severe
failure: which GT boxes were found by a prediction, and which were missed.
"""

from __future__ import annotations


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two xyxy boxes (same convention as the manifests).

    Returns a value in [0, 1]: 0 when the boxes do not overlap (including when
    they only touch along an edge), 1 when they are identical.
    """
    overlap = [max(a[0], b[0]),
              max(a[1], b[1]),
              min(a[2], b[2]),
              min(a[3], b[3])]
    width_overlap = max(overlap[2] - overlap[0], 0.0)
    height_overlap = max(overlap[3] - overlap[1], 0.0)
    area_overlap = width_overlap * height_overlap
    area_a = ((a[2] - a[0]) * (a[3] - a[1]))
    area_b = ((b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - area_overlap
    return area_overlap/union
    

def match_image(
    gt_boxes: list[tuple[float, float, float, float]],
    gt_classes: list[str],
    pred_boxes: list[tuple[float, float, float, float]],
    pred_classes: list[str],
    pred_scores: list[float],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of predictions to GT boxes in one image (decision D8).

    Predictions are visited from highest to lowest score (ties: lower index
    first). Each one claims the still-unclaimed GT box of the same class with
    the highest IoU, if that IoU is >= iou_threshold. Every GT box and every
    prediction is used at most once. This is the same rule pycocotools uses.

    Returns (pred_index, gt_index) pairs. Unclaimed GT boxes are misses;
    unmatched predictions are false positives or duplicates.
    """
    order = sorted(range(len(pred_scores)), key = lambda i: -pred_scores[i])
    claimed = set()
    pairs = []

    for p in order:
        best_gt = None
        best_iou = 0.0
        for g in range(len(gt_boxes)):
            if (g in claimed or gt_classes[g] != pred_classes[p]):
                continue
            v = iou(pred_boxes[p], gt_boxes[g])
            if v > best_iou:
                best_gt = g
                best_iou = v
        if best_gt is not None and best_iou >= iou_threshold:
            claimed.add(best_gt)
            pairs.append((p, best_gt))
    return pairs
