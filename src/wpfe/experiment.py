"""Run the frozen selection experiment once (decision D11).

This is the only module that reads review-pool annotations. Selectors run
first, on the label-free candidate table; labels are joined afterwards only to
count how many selected images are severe failures (D10 rule).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wpfe.config import Config
from wpfe.matching import match_image
from wpfe.select import candidate_table, select_diverse, select_random, select_top

BOX = ["x_min", "y_min", "x_max", "y_max"]


def image_recall(preds: pd.DataFrame, gt: pd.DataFrame, image_ids, score_threshold: float, iou_threshold: float) -> pd.DataFrame:
    """Per-image recall at the operating point, original view only."""
    preds = preds[(preds["view"] == "orig") & (preds["score"] >= score_threshold)]
    gt_by, pr_by = dict(tuple(gt.groupby("image_id"))), dict(tuple(preds.groupby("image_id")))
    rows = []
    for image_id in image_ids:
        g, p = gt_by.get(image_id), pr_by.get(image_id, preds.iloc[:0])
        n_gt = 0 if g is None else len(g)
        found = 0 if g is None else len(match_image(
            list(g[BOX].itertuples(index=False, name=None)), g["class_name"].tolist(),
            list(p[BOX].itertuples(index=False, name=None)), p["class_name"].tolist(), p["score"].tolist(),
            iou_threshold,
        ))
        rows.append({"image_id": image_id, "n_gt": n_gt, "found": found, "n_pred": len(p),
                     "recall": found / n_gt if n_gt else float("nan")})
    return pd.DataFrame(rows)


def _evaluate_pool(cands: pd.DataFrame, failure: pd.Series, budgets, n_seeds: int, gap: int) -> tuple[list[dict], dict]:
    rows, picks = [], {}
    for k in budgets:
        rand = np.array([failure[select_random(cands, k, seed)].sum() for seed in range(n_seeds)])
        rows.append({"selector": "random", "budget": k, "failures": float(rand.mean()),
                     "random_p2_5": float(np.percentile(rand, 2.5)), "random_p97_5": float(np.percentile(rand, 97.5))})
        for name, ids in (("instability", select_top(cands, k)), ("instability_diverse", select_diverse(cands, k, gap))):
            found = int(failure[ids].sum())
            rows.append({"selector": name, "budget": k, "failures": found,
                         "lift": found / rand.mean() if rand.mean() else float("nan"),
                         "p_vs_random": float((rand >= found).mean()), "n_selected": len(ids)})
            picks[f"{name}@{k}"] = ids
    return rows, picks


def density_check(per_image: pd.DataFrame, picks: dict, budgets, n_seeds: int) -> pd.DataFrame:
    """Post-hoc (not in the frozen design): is the instability gain just "pick sparse images"?

    Compares top-k instability with (a) the label-free rule "fewest detections at the
    operating point" and (b) random batches with the same mix of detection counts.
    """
    pi = per_image.reset_index(drop=True)
    fail = pi.set_index("image_id")["failure"]
    bucket = pd.cut(pi["n_pred"], [-1, 2, 5, 10, 20, 50, np.inf])
    rows = []
    for k in budgets:
        top = picks[f"all/instability@{k}"]
        fewest = pi.sort_values(["n_pred", "image_id"])["image_id"].head(k)
        mix = bucket[pi["image_id"].isin(top)].value_counts()
        rng = np.random.default_rng(0)
        matched = np.array([
            fail[np.concatenate([rng.choice(pi.loc[bucket == b, "image_id"], n, replace=False) for b, n in mix.items() if n])].sum()
            for _ in range(n_seeds)
        ])
        rows.append({"budget": k, "instability": int(fail[top].sum()), "fewest_detections": int(fail[fewest].sum()),
                     "random_same_density_mean": float(matched.mean()),
                     "p_instability_vs_same_density": float((matched >= fail[top].sum()).mean())})
    return pd.DataFrame(rows)


def run_experiment(cfg: Config) -> dict:
    fe, ex = cfg.raw["failure_event"], cfg.raw["experiment"]
    cands = candidate_table(cfg)  # label-free; selectors see only this

    # --- labels are read from here on ------------------------------------------
    preds = pd.read_parquet(cfg.artifacts_dir / "predictions" / cfg.raw["frozen_model"] / "predictions.parquet")
    ann = pd.read_parquet(cfg.artifacts_dir / "manifest" / "annotations.parquet")
    ann = ann[ann["image_id"].isin(cands["image_id"])]

    out_dir = cfg.artifacts_dir / "experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, all_picks, per_image_tables = [], {}, []
    # Primary IoU first; 0.3 is the sensitivity check promised in D8.
    for iou_thr, label in ((fe["iou_threshold"], "primary"), (0.3, "sensitivity_iou0.3")):
        rec = image_recall(preds, ann, cands["image_id"], fe["score_threshold"], iou_thr)
        rec["failure"] = rec["recall"] < fe["tau"]
        per_image = cands.merge(rec, on="image_id")
        per_image_tables.append(per_image.assign(analysis=label))
        failure = per_image.set_index("image_id")["failure"]
        scopes = [("all", cands)] + [(f"subset_{s}", cands[cands["subset"] == s]) for s in cfg.review_subsets]
        for scope, pool in scopes:
            rows, picks = _evaluate_pool(pool.reset_index(drop=True), failure, ex["budgets"], ex["random_seeds"],
                                         ex["diversity_gap_frames"])
            base = float(failure[pool["image_id"]].mean())
            results += [{"analysis": label, "scope": scope, "pool_size": len(pool), "base_rate": base, **r} for r in rows]
            if label == "primary":
                all_picks |= {f"{scope}/{key}": ids for key, ids in picks.items()}

    results = pd.DataFrame(results)
    per_image = pd.concat(per_image_tables, ignore_index=True)
    results.to_csv(out_dir / "results.csv", index=False)
    per_image.to_parquet(out_dir / "per_image.parquet", index=False)
    (out_dir / "selections.json").write_text(json.dumps(all_picks, indent=1))

    primary = per_image[per_image["analysis"] == "primary"]
    exploratory = density_check(primary, all_picks, ex["budgets"], ex["random_seeds"])
    exploratory.to_csv(out_dir / "exploratory_density_check.csv", index=False)
    summary = {
        "frozen_model": cfg.raw["frozen_model"],
        "failure_event": fe,
        "experiment": ex,
        "config_hash": cfg.hash(),
        "pool_size": len(cands),
        "base_rate_primary": float(primary["failure"].mean()),
        "spearman_instability_vs_recall": float(primary[["instability", "recall"]].corr("spearman").iloc[0, 1]),
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "results": results, "exploratory": exploratory}
