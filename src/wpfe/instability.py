"""Label-blind flip-instability score per image (decision D9).

Boxes from the original view are matched to boxes from the flipped view (already
mapped back to original coordinates) with the same rule as the failure event:
class-aware, greedy one-to-one, IoU >= 0.5. The score is the share of total
confidence carried by boxes that found no partner in the other view:

    instability = sum(scores of unpaired boxes, both views) / sum(all scores, both views)

0 means both views agree on every box; 1 means no box has a partner. Images with
no boxes in either view score 0: they look stable even if the detector missed
everything (a known blind spot, reported as "consistent misses").
"""

from __future__ import annotations

from wpfe.matching import match_image

Box = tuple[float, float, float, float]


def instability(
    orig_boxes: list[Box],
    orig_classes: list[str],
    orig_scores: list[float],
    flip_boxes: list[Box],
    flip_classes: list[str],
    flip_scores: list[float],
    iou_threshold: float = 0.5,
) -> float:
    pairs = match_image(orig_boxes, orig_classes, flip_boxes, flip_classes, flip_scores, iou_threshold)
    total = sum(orig_scores) + sum(flip_scores)
    paired = 0.0
    for (p, g) in pairs:
        paired += flip_scores[p] + orig_scores[g]
    if total == 0:
        return 0.0
    return (total - paired) / total


def score_images(preds: "pd.DataFrame", image_ids: list[str], iou_threshold: float = 0.5) -> "pd.DataFrame":
    """Instability for every image in `image_ids` from two-view predictions.

    Images with no predictions at all still get a row (instability 0), so they
    stay visible as candidates for review.
    """
    import pandas as pd

    cols = ["x_min", "y_min", "x_max", "y_max"]
    groups = {k: g for k, g in preds.groupby(["image_id", "view"])}
    empty = preds.iloc[:0]
    rows = []
    for image_id in image_ids:
        o, f = groups.get((image_id, "orig"), empty), groups.get((image_id, "flip"), empty)
        value = instability(
            list(o[cols].itertuples(index=False, name=None)), o["class_name"].tolist(), o["score"].tolist(),
            list(f[cols].itertuples(index=False, name=None)), f["class_name"].tolist(), f["score"].tolist(),
            iou_threshold,
        )
        rows.append({"image_id": image_id, "instability": value, "n_orig": len(o), "n_flip": len(f),
                     "score_mass": float(o["score"].sum() + f["score"].sum())})
    return pd.DataFrame(rows)
