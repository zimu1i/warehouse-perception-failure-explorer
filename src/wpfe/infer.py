"""Cached two-view inference with the frozen fine-tuned detector.

Every image in the tune and review subsets is scored twice: as-is ("orig") and
mirrored left-right ("flip"). Flip-view boxes are mapped back to original image
coordinates, so both views can be compared box for box. Unannotated images are
included for the optional D2 side analysis.

This step never reads annotations: it is label-blind by construction. Output is
written in shards so an interrupted run resumes where it stopped; each shard
also lists the images it covered, because an image with no detections has no
prediction rows.
"""

from __future__ import annotations

import json
import os
import platform
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import hflip, to_tensor

from wpfe.boxes import hflip_boxes
from wpfe.config import Config
from wpfe.detect import pick_device
from wpfe.train import _release_cache, build_model

COLUMNS = ["image_id", "view", "class_name", "score", "x_min", "y_min", "x_max", "y_max"]
SHARD_SIZE = 200


@torch.inference_mode()
def predict_both_views(model, image: torch.Tensor, image_id: str, classes: list[str], device: str) -> list[dict]:
    """Run `model` on the image and its mirror; return rows in original coordinates."""
    width = float(image.shape[-1])
    rows = []
    for view, x in (("orig", image), ("flip", hflip(image))):
        out = model([x.to(device)])[0]
        boxes = out["boxes"].cpu()
        if view == "flip":
            boxes = hflip_boxes(boxes, width)
        for box, label, score in zip(boxes.tolist(), out["labels"].tolist(), out["scores"].tolist()):
            x_min, y_min, x_max, y_max = box
            rows.append({"image_id": image_id, "view": view, "class_name": classes[label - 1], "score": score,
                         "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max})
    return rows


def load_trained_model(cfg: Config, run_id: str, device: str):
    run_dir = cfg.artifacts_dir / "models" / run_id
    ckpt = sorted(run_dir.glob("epoch_*.pt"))[-1]
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if state["epoch"] != cfg.raw["training"]["epochs"]:
        raise ValueError(f"{ckpt.name} is not the final epoch; training is incomplete")
    classes = list(cfg.class_map.values())
    model = build_model(len(classes), cfg.raw["training"], cfg.raw["detector"]["score_floor"], pretrained=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), ckpt.name, classes


def run_infer(cfg: Config, run_id: str, device: str = "auto", limit: int | None = None) -> dict:
    """Score tune + review images in both views. `limit` is for smoke tests."""
    device = pick_device(device)
    images = pd.read_parquet(cfg.artifacts_dir / "manifest" / "images.parquet")
    subsets = cfg.tune_subsets + cfg.review_subsets
    todo = images[images["subset"].isin(subsets)].sort_values("image_id").reset_index(drop=True)
    if limit:
        todo = todo.head(limit)

    out = cfg.artifacts_dir / "predictions" / (run_id if not limit else f"smoke_{run_id}")
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in shards.glob("covered_*.parquet"):
        done |= set(pd.read_parquet(f)["image_id"])
    remaining = todo[~todo["image_id"].isin(done)]
    print(f"{len(todo)} images, {len(done)} already done, {len(remaining)} to go", flush=True)

    model, ckpt_name, classes = load_trained_model(cfg, run_id, device)
    times = []
    for start in range(0, len(remaining), SHARD_SIZE):
        chunk = remaining.iloc[start : start + SHARD_SIZE]
        rows = []
        for i, r in enumerate(chunk.itertuples()):
            with Image.open(cfg.raw_root / r.rel_path) as im:
                image = to_tensor(im.convert("RGB"))
            t = time.perf_counter()
            rows += predict_both_views(model, image, r.image_id, classes, device)
            times.append(time.perf_counter() - t)
            if i % 50 == 0:
                _release_cache(device)
        shard_id = chunk["image_id"].iloc[0]
        # Predictions first, coverage last: a shard only counts as done once both exist.
        pd.DataFrame(rows, columns=COLUMNS).to_parquet(shards / f"preds_{shard_id}.parquet", index=False)
        chunk[["image_id"]].to_parquet(shards / f"covered_{shard_id}.parquet", index=False)
        print(f"{len(done) + start + len(chunk)}/{len(todo)} images, {sorted(times)[len(times) // 2]:.2f} s/image (2 views)",
              flush=True)

    preds = pd.concat([pd.read_parquet(f) for f in sorted(shards.glob("preds_*.parquet"))], ignore_index=True)
    covered = pd.concat([pd.read_parquet(f) for f in sorted(shards.glob("covered_*.parquet"))], ignore_index=True)
    if set(covered["image_id"]) != set(todo["image_id"]):
        raise RuntimeError("coverage does not match the image list; rerun to resume")
    preds.to_parquet(out / "predictions.parquet", index=False)
    covered.to_parquet(out / "covered_images.parquet", index=False)

    summary = {
        "model_run_id": run_id,
        "checkpoint": ckpt_name,
        "subsets": subsets,
        "n_images": len(todo),
        "n_preds": {v: int((preds["view"] == v).sum()) for v in ("orig", "flip")},
        "score_floor": cfg.raw["detector"]["score_floor"],
        "sec_per_image_median_this_session": round(sorted(times)[len(times) // 2], 3) if times else None,
        "device": device,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "config_hash": cfg.hash(),
        "reads_annotations": False,
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    return summary
