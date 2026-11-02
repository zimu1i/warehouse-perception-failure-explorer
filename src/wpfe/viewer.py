"""Streamlit viewer: `uv run --group viewer streamlit run src/wpfe/viewer.py`.

Two modes:
- Review queue: what a reviewer would see. Images ranked by a selector, with
  original and mirrored predictions. Annotations are never loaded in this mode.
- Results: the experiment outcome, with labels revealed for inspection.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from wpfe.config import load_config
from wpfe.matching import match_image
from wpfe.select import candidate_table, select_diverse, select_random, select_top

BOX = ["x_min", "y_min", "x_max", "y_max"]
GT_COLOR, ORIG_COLOR, FLIP_COLOR = "#ffffff", "#2a78d6", "#eb6834"
SITES = {"Both review sites": None, "Site 1": 1, "Site 4": 4}

cfg = load_config()
EXP = cfg.artifacts_dir / "experiment"


@st.cache_data
def candidates() -> pd.DataFrame:
    scores = pd.read_parquet(cfg.artifacts_dir / "predictions" / cfg.raw["frozen_model"] / "instability.parquet")
    return candidate_table(cfg).merge(scores[["image_id", "n_orig", "n_flip", "score_mass"]], on="image_id")


@st.cache_data
def predictions() -> dict[str, pd.DataFrame]:
    preds = pd.read_parquet(cfg.artifacts_dir / "predictions" / cfg.raw["frozen_model"] / "predictions.parquet")
    return dict(tuple(preds.groupby("image_id")))


@st.cache_data
def rel_paths() -> dict[str, str]:
    images = pd.read_parquet(cfg.artifacts_dir / "manifest" / "images.parquet")
    return dict(zip(images["image_id"], images["rel_path"]))


@st.cache_data
def labels() -> dict[str, pd.DataFrame]:
    """Only called in results mode."""
    ann = pd.read_parquet(cfg.artifacts_dir / "manifest" / "annotations.parquet")
    return dict(tuple(ann.groupby("image_id")))


@st.cache_data
def per_image() -> pd.DataFrame:
    return pd.read_parquet(EXP / "per_image.parquet").query("analysis == 'primary'")


def _param(name: str, default: int, low: int, high: int) -> int:
    """Initial widget value from the URL, e.g. ?mode=results&show=2&image=3 or ?rank=20."""
    try:
        return min(max(int(st.query_params.get(name, default)), low), high)
    except ValueError:
        return default


def _tuples(df: pd.DataFrame) -> list[tuple]:
    return list(df[BOX].itertuples(index=False, name=None))


def split_views(image_id: str, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Original and mirrored predictions above the threshold, each with a `paired` flag."""
    p = predictions().get(image_id, pd.DataFrame(columns=["view", "score", "class_name", *BOX]))
    p = p[p["score"] >= threshold]
    o, f = p[p["view"] == "orig"].reset_index(drop=True), p[p["view"] == "flip"].reset_index(drop=True)
    pairs = match_image(_tuples(o), o["class_name"].tolist(), _tuples(f), f["class_name"].tolist(), f["score"].tolist())
    o["paired"] = o.index.isin([g for _, g in pairs])
    f["paired"] = f.index.isin([q for q, _ in pairs])
    return o, f


def draw(image_id: str, orig: pd.DataFrame, flip: pd.DataFrame, gt: pd.DataFrame | None = None, width: int = 1100) -> Image.Image:
    with Image.open(cfg.raw_root / rel_paths()[image_id]) as im:
        im = im.convert("RGB")
    s = width / im.width
    im = im.resize((width, round(im.height * s)))
    d = ImageDraw.Draw(im)
    if gt is not None:
        for r in gt.itertuples():
            d.rectangle([r.x_min * s, r.y_min * s, r.x_max * s, r.y_max * s], outline=GT_COLOR, width=1)
    for boxes, color in ((orig, ORIG_COLOR), (flip, FLIP_COLOR)):
        for r in boxes.itertuples():
            # unpaired boxes (seen in only one view) are drawn thicker: that is the disagreement
            d.rectangle([r.x_min * s, r.y_min * s, r.x_max * s, r.y_max * s], outline=color, width=2 if r.paired else 5)
    return im


def legend(with_labels: bool) -> None:
    parts = [f"<span style='color:{ORIG_COLOR}'>■</span> original view",
             f"<span style='color:{FLIP_COLOR}'>■</span> mirrored view (mapped back)",
             "thick outline = no partner in the other view"]
    if with_labels:
        parts.insert(0, "□ ground truth (white)")
    st.caption(" &nbsp;·&nbsp; ".join(parts), unsafe_allow_html=True)


def review_queue() -> None:
    st.header("Review queue")
    st.write("This is what a reviewer would see. Images are ranked without labels, so this page never loads them.")
    selector = st.sidebar.selectbox("Selector", ["Flip instability", "Instability with spacing", "Random"])
    k = st.sidebar.select_slider("Batch size", [25, 50, 100], value=50)
    site = st.sidebar.selectbox("Site", list(SITES))
    # The instability score uses every box down to the 0.05 floor, so show those by default.
    threshold = st.sidebar.slider("Show boxes with score at least", 0.05, 0.95, cfg.raw["detector"]["score_floor"], 0.05)

    cands = candidates()
    if SITES[site] is not None:
        cands = cands[cands["subset"] == SITES[site]].reset_index(drop=True)
    if selector == "Flip instability":
        ids = select_top(cands, k)
    elif selector == "Instability with spacing":
        ids = select_diverse(cands, k, cfg.raw["experiment"]["diversity_gap_frames"])
    else:
        ids = select_random(cands, k, seed=st.sidebar.number_input("Random seed", 0, 999, 0))

    queue = cands.set_index("image_id").loc[ids].reset_index()
    queue.insert(0, "rank", range(1, len(queue) + 1))
    left, right = st.columns([2, 5])
    with left:
        st.dataframe(queue[["rank", "image_id", "subset", "instability", "score_mass"]].round(3),
                     hide_index=True, height=560)
        rank = st.number_input("Open image at rank", 1, len(queue), _param("rank", 1, 1, len(queue)))
    row = queue.iloc[rank - 1]
    orig, flip = split_views(row.image_id, threshold)
    with right:
        st.image(draw(row.image_id, orig, flip), use_container_width=True)
        legend(with_labels=False)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Instability", f"{row.instability:.2f}")
        m2.metric("Total confidence", f"{row.score_mass:.2f}", help="Sum of all box scores in both views. "
                  "A high instability with very little confidence comes from a few faint boxes.")
        m3.metric("Boxes shown", f"{len(orig)} / {len(flip)}", help="original / mirrored, at the threshold in the sidebar")
        m4.metric("Without a partner", int((~orig["paired"]).sum() + (~flip["paired"]).sum()))


def results() -> None:
    st.header("Results (labels revealed)")
    st.write("Labels are opened only here, to grade the batches each selector chose.")
    st.image(str(cfg.root / "docs" / "figures" / "failures_by_budget.png"), width=800)

    res = pd.read_csv(EXP / "results.csv").query("analysis == 'primary'")
    scope = st.selectbox("Pool", ["all", *[f"subset_{s}" for s in cfg.review_subsets]],
                         format_func=lambda s: "Both review sites" if s == "all" else s.replace("subset_", "Site "))
    table = res[res["scope"] == scope].pivot(index="budget", columns="selector", values="failures").round(1)
    st.dataframe(table.rename(columns={"random": "Random (average)", "instability": "Flip instability",
                                       "instability_diverse": "Instability with spacing"}))

    st.subheader("Inspect images")
    pi = per_image()
    picked = set(json.loads((EXP / "selections.json").read_text())["all/instability@50"])
    median = pi["instability"].median()
    views = {
        "Picked in the top 50 and a real failure": pi[pi["image_id"].isin(picked) & pi["failure"]],
        "Picked in the top 50 but not a failure": pi[pi["image_id"].isin(picked) & ~pi["failure"]],
        "Consistent misses (failure, below-median instability)": pi[pi["failure"] & (pi["instability"] < median)],
        "All failures": pi[pi["failure"]],
    }
    left, right = st.columns([2, 5])
    with left:
        choice = st.radio("Show", list(views), index=_param("show", 0, 0, len(views) - 1))
        subset = views[choice].sort_values("instability", ascending=choice.startswith("Consistent")).reset_index(drop=True)
        st.caption(f"{len(subset)} images")
        if subset.empty:
            return
        i = st.number_input("Image", 1, len(subset), _param("image", 1, 1, len(subset))) - 1
        row = subset.iloc[i]
        st.metric("Objects found", f"{row.found:.0f} of {row.n_gt:.0f}")
        st.metric("Image recall", f"{row.recall:.2f}")
        st.metric("Severe failure", "yes" if row.failure else "no")
        st.metric("Instability", f"{row.instability:.2f}")
    with right:
        orig, flip = split_views(row.image_id, cfg.raw["failure_event"]["score_threshold"])
        st.image(draw(row.image_id, orig, flip, labels().get(row.image_id)), use_container_width=True)
        legend(with_labels=True)


st.set_page_config(page_title="Warehouse Perception Failure Explorer", layout="wide")
st.sidebar.title("Failure Explorer")
mode = st.sidebar.radio("Mode", ["Review queue", "Results"], index=int(st.query_params.get("mode") == "results"))
st.sidebar.divider()
if mode == "Review queue":
    review_queue()
else:
    results()
