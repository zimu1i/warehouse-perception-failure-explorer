import pandas as pd

from wpfe.experiment import _evaluate_pool, image_recall


def test_perfect_signal_finds_all_failures_and_beats_random():
    n = 100
    cands = pd.DataFrame({"image_id": [f"i{j:03d}" for j in range(n)], "seq": "s", "pos": range(n),
                          "instability": [1.0 if j < 20 else 0.0 for j in range(n)], "subset": 1})
    failure = pd.Series([j < 20 for j in range(n)], index=cands["image_id"])
    rows, _ = _evaluate_pool(cands, failure, [10], n_seeds=200, gap=0)
    r = {x["selector"]: x for x in rows}
    assert r["instability"]["failures"] == 10
    assert abs(r["random"]["failures"] - 2.0) < 0.5  # 20% base rate x 10
    assert r["instability"]["p_vs_random"] < 0.01


def test_image_recall_uses_operating_point_and_orig_view():
    box = {"x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0}
    gt = pd.DataFrame([{"image_id": "a", "class_name": "pallet", **box}, {"image_id": "a", "class_name": "pallet",
                       "x_min": 50.0, "y_min": 0.0, "x_max": 60.0, "y_max": 10.0}])
    preds = pd.DataFrame([
        {"image_id": "a", "view": "orig", "class_name": "pallet", "score": 0.9, **box},
        {"image_id": "a", "view": "flip", "class_name": "pallet", "score": 0.9, "x_min": 50.0, "y_min": 0.0,
         "x_max": 60.0, "y_max": 10.0},  # flip view never counts
        {"image_id": "a", "view": "orig", "class_name": "pallet", "score": 0.1, "x_min": 50.0, "y_min": 0.0,
         "x_max": 60.0, "y_max": 10.0},  # below the operating threshold
    ])
    r = image_recall(preds, gt, ["a"], score_threshold=0.4, iou_threshold=0.5).iloc[0]
    assert r["found"] == 1 and r["recall"] == 0.5
