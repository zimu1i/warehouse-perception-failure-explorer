"""Label-blind review selectors (decision D11).

Every selector receives only the candidate table built here: image ids, camera
sequence, frame position and the flip-instability score. Annotations are never
passed in, so a selector cannot read review-pool labels even by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wpfe.config import Config

CANDIDATE_COLUMNS = ["image_id", "subset", "seq", "pos", "instability"]


def candidate_table(cfg: Config) -> pd.DataFrame:
    """Review pool (annotated images only, D2) with label-free features."""
    images = pd.read_parquet(cfg.artifacts_dir / "manifest" / "images.parquet")
    scores = pd.read_parquet(cfg.artifacts_dir / "predictions" / cfg.raw["frozen_model"] / "instability.parquet")
    pool = images[images["subset"].isin(cfg.review_subsets) & (images["annotation_status"] == "annotated")]
    pool = pool.merge(scores[["image_id", "instability"]], on="image_id", how="left", validate="one_to_one")
    if pool["instability"].isna().any():
        raise ValueError("some review images have no instability score; rerun inference")
    pool = pool.assign(
        seq=pool["subset"].astype(str) + "/" + pool["session"].fillna("-") + "/" + pool["camera"].fillna("-"),
        # Filename stems are capture times with mixed units and decimal separators,
        # but they sort correctly within one camera sequence.
        t=pool["rel_path"].str.rsplit("/", n=1).str[-1].str.removesuffix(".jpg").str.replace(",", ".").astype(float),
    ).sort_values(["seq", "t", "image_id"])
    pool["pos"] = pool.groupby("seq").cumcount()
    return pool[CANDIDATE_COLUMNS].reset_index(drop=True)


def _ranked(cands: pd.DataFrame) -> pd.DataFrame:
    """Highest instability first; image_id breaks ties so the order is deterministic."""
    return cands.sort_values(["instability", "image_id"], ascending=[False, True])


def select_random(cands: pd.DataFrame, k: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return cands["image_id"].to_numpy()[rng.permutation(len(cands))[:k]].tolist()


def select_top(cands: pd.DataFrame, k: int) -> list[str]:
    return _ranked(cands)["image_id"].head(k).tolist()


def select_diverse(cands: pd.DataFrame, k: int, gap: int) -> list[str]:
    """Walk the instability ranking; skip an image if an already *picked* image in
    the same sequence is within `gap` frame positions of it. Returns at most k
    image ids in pick order. gap = 0 is the same as select_top.
    """
    ranking = _ranked(cands)
    picked = {}
    out = []
    for row in ranking.itertuples():
        nearby = picked.get(row.seq, [])
        if any(abs(row.pos - q) <= gap for q in nearby):
            continue
        picked.setdefault(row.seq, []).append(row.pos)
        out.append(row.image_id)
        if len(out) == k:
            break
    return out
        
        
