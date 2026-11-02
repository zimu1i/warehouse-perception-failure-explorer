"""Static figures for the README and report, built from experiment artifacts."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from wpfe.config import Config

# Categorical slots 1-3 of the reference palette (validated light-mode set);
# random is a neutral gray because it is the baseline, not a method.
COLORS = {"random": "#8a8a85", "instability": "#2a78d6", "instability_diverse": "#eb6834", "fewest_detections": "#1baf7a"}
LABELS = {"random": "Random (mean, 95% range)", "instability": "Flip instability",
          "instability_diverse": "Instability + diversity", "fewest_detections": "Fewest detections (post-hoc)"}
INK, MUTED = "#2b2b28", "#6b6b66"


def budget_chart(cfg: Config, path) -> None:
    exp = cfg.artifacts_dir / "experiment"
    res = pd.read_csv(exp / "results.csv").query("analysis == 'primary' and scope == 'all'")
    dens = pd.read_csv(exp / "exploratory_density_check.csv").set_index("budget")
    budgets = sorted(res["budget"].unique())
    series = ["random", "instability", "instability_diverse", "fewest_detections"]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    width = 0.2
    for i, name in enumerate(series):
        xs = [b + (i - 1.5) * width for b in range(len(budgets))]
        if name == "fewest_detections":
            ys = [dens.loc[b, "fewest_detections"] for b in budgets]
        else:
            ys = [res.query("selector == @name and budget == @b")["failures"].iloc[0] for b in budgets]
        bars = ax.bar(xs, ys, width * 0.92, color=COLORS[name], label=LABELS[name],
                      hatch="//" if name == "fewest_detections" else None, edgecolor="white", linewidth=0)
        if name == "random":
            rows = [res.query("selector == 'random' and budget == @b").iloc[0] for b in budgets]
            ax.errorbar(xs, ys, yerr=[[r.failures - r.random_p2_5 for r in rows], [r.random_p97_5 - r.failures for r in rows]],
                        fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        tops = [r.random_p97_5 for r in rows] if name == "random" else ys
        for bar, y, top in zip(bars, ys, tops):  # random's label sits above its whisker
            ax.annotate(f"{y:.0f}", (bar.get_x() + bar.get_width() / 2, top), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(budgets)), [f"{b} images" for b in budgets])
    ax.set_ylabel("Severe failures found in the batch", color=INK)
    ax.set_title("Review batches from 2,277 images at two unseen warehouses (15.6% are failures)",
                 fontsize=10, color=INK, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color="#e6e6e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _panel(cfg: Config, image_id: str, preds: pd.DataFrame, gt: pd.DataFrame, title: str, width: int = 640) -> Image.Image:
    images = pd.read_parquet(cfg.artifacts_dir / "manifest" / "images.parquet").set_index("image_id")
    with Image.open(cfg.raw_root / images.loc[image_id, "rel_path"]) as im:
        im = im.convert("RGB")
    scale = width / im.width
    im = im.resize((width, round(im.height * scale)))
    draw = ImageDraw.Draw(im)
    thr = cfg.raw["failure_event"]["score_threshold"]
    layers = [
        (gt[gt["image_id"] == image_id], "#ffffff", 1),
        (preds[(preds["image_id"] == image_id) & (preds["view"] == "orig") & (preds["score"] >= thr)], COLORS["instability"], 2),
        (preds[(preds["image_id"] == image_id) & (preds["view"] == "flip") & (preds["score"] >= thr)], COLORS["instability_diverse"], 1),
    ]
    for rows, color, w in layers:
        for r in rows.itertuples():
            draw.rectangle([r.x_min * scale, r.y_min * scale, r.x_max * scale, r.y_max * scale], outline=color, width=w)
    canvas = Image.new("RGB", (im.width, im.height + 34), "white")
    canvas.paste(im, (0, 34))
    ImageDraw.Draw(canvas).text((6, 7), title, fill=INK, font=ImageFont.load_default(size=17))
    return canvas


def example_panels(cfg: Config, path) -> None:
    """Three label-revealed examples: a caught failure, a false alarm, a consistent miss."""
    exp = cfg.artifacts_dir / "experiment"
    pi = pd.read_parquet(exp / "per_image.parquet").query("analysis == 'primary'")
    preds = pd.read_parquet(cfg.artifacts_dir / "predictions" / cfg.raw["frozen_model"] / "predictions.parquet")
    gt = pd.read_parquet(cfg.artifacts_dir / "manifest" / "annotations.parquet")
    picked = set(json.loads((exp / "selections.json").read_text())["all/instability@50"])
    top = pi[pi["image_id"].isin(picked)]
    dense = pi[pi["n_gt"] >= 5]
    caught = top[top["failure"] & (top["n_gt"] >= 5)].nlargest(1, "instability")
    if caught.empty:
        caught = top[top["failure"]].nlargest(1, "instability")
    false_alarm = top[~top["failure"]].nlargest(1, "instability")
    miss = dense[dense["failure"]].nsmallest(1, "instability")
    panels = []
    for row, kind in ((caught, "Picked, real failure"), (false_alarm, "Picked, not a failure"), (miss, "Consistent miss")):
        r = row.iloc[0]
        title = f"{kind}: instability {r.instability:.2f}, recall {r.found:.0f}/{r.n_gt:.0f}"
        panels.append(_panel(cfg, r.image_id, preds, gt, title))
    h = max(p.height for p in panels)
    sheet = Image.new("RGB", (sum(p.width for p in panels) + 8 * (len(panels) - 1), h), "white")
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 8
    sheet.save(path, quality=82)


def run_figures(cfg: Config) -> list[str]:
    out = cfg.root / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    budget_chart(cfg, out / "failures_by_budget.png")
    example_panels(cfg, out / "examples.jpg")
    return [str(p.relative_to(cfg.root)) for p in sorted(out.iterdir())]
