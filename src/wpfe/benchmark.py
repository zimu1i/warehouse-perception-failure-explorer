"""Small-sample detector benchmark on dev images: runtime, memory, unmapped phrases, COCO metrics."""

from __future__ import annotations

import json
import platform
import time

import pandas as pd
import torch
from PIL import Image

from wpfe.coco_eval import coco_metrics
from wpfe.config import Config
from wpfe.detect import load_detector, pick_device


def sample_dev_images(images: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic sample of annotated dev images, spread evenly across subsets."""
    pool = images[(images["split"] == "dev") & (images["annotation_status"] == "annotated")]
    subsets = sorted(pool["subset"].unique())
    per = [n // len(subsets) + (1 if i < n % len(subsets) else 0) for i in range(len(subsets))]
    parts = [pool[pool["subset"] == s].sample(k, random_state=seed) for s, k in zip(subsets, per)]
    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def run_benchmark(cfg: Config, model_key: str, prompt_set_key: str, device: str = "auto") -> dict:
    device = pick_device(device)
    det_cfg, bench_cfg = cfg.raw["detector"], cfg.raw["benchmark"]
    manifest = cfg.artifacts_dir / "manifest"
    images = pd.read_parquet(manifest / "images.parquet")
    annotations = pd.read_parquet(manifest / "annotations.parquet")
    sample = sample_dev_images(images, bench_cfg["n_images"], bench_cfg["seed"])

    t0 = time.perf_counter()
    detector = load_detector(model_key, prompt_set_key, det_cfg, device)
    load_s = time.perf_counter() - t0

    rows, times = [], []
    for i, r in enumerate(sample.itertuples()):
        with Image.open(cfg.raw_root / r.rel_path) as im:
            image = im.convert("RGB")
        _sync(device)
        t = time.perf_counter()
        preds = detector.predict(image, det_cfg["score_floor"])
        _sync(device)
        if i > 0:  # first image includes warm-up / kernel compilation
            times.append(time.perf_counter() - t)
        rows += [{"image_id": r.image_id, **p} for p in preds]

    preds = pd.DataFrame(rows, columns=["image_id", "class_name", "phrase", "score", "x_min", "y_min", "x_max", "y_max"])
    gt = annotations[annotations["image_id"].isin(sample["image_id"])]
    metrics = coco_metrics(sample, gt, preds, list(cfg.class_map.values()))

    run_id = f"bench_{model_key}_{prompt_set_key}"
    out = cfg.artifacts_dir / "benchmarks" / run_id
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "predictions.parquet", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)

    summary = {
        "run_id": run_id,
        "model": det_cfg["models"][model_key],
        "prompt_set": prompt_set_key,
        "prompts": det_cfg["prompt_sets"][prompt_set_key],
        "score_floor": det_cfg["score_floor"],
        "device": device,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "config_hash": cfg.hash(),
        "n_images": len(sample),
        "image_ids": sample["image_id"].tolist(),
        "load_s": round(load_s, 1),
        "sec_per_image_median": round(sorted(times)[len(times) // 2], 3) if times else None,
        "sec_per_image_mean": round(sum(times) / len(times), 3) if times else None,
        "mps_driver_mem_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if device == "mps" else None,
        "n_preds": len(preds),
        "n_unmapped_preds": int(preds["class_name"].isna().sum()),
        "unmapped_phrases": preds.loc[preds["class_name"].isna(), "phrase"].value_counts().head(10).to_dict(),
        "n_gt": len(gt),
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "metrics": metrics}
