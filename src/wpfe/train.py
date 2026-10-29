"""Fine-tune a COCO-pretrained Faster R-CNN on the dev training subsets (decision D6).

Resumable: a checkpoint is written after every epoch and training continues
from the latest one. Only `training.train_subsets` images are used for gradient
updates; a fixed sample of `training.tune_subsets` images is scored after each
epoch for monitoring. The schedule is fixed in advance and the last epoch is the
model used downstream, so the tune subset is never used to pick an epoch.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import hflip, to_tensor

from wpfe.benchmark import sample_dev_images
from wpfe.boxes import hflip_boxes
from wpfe.coco_eval import coco_metrics
from wpfe.config import Config
from wpfe.detect import pick_device


def check_training_split(cfg: Config) -> None:
    """Leakage guard: train and tune subsets are disjoint and both inside dev."""
    train, tune = set(cfg.train_subsets), set(cfg.tune_subsets)
    dev, review = set(cfg.dev_subsets), set(cfg.review_subsets)
    if not train or not tune:
        raise ValueError("train_subsets and tune_subsets must both be non-empty")
    if train & tune:
        raise ValueError(f"subsets {sorted(train & tune)} are in both train and tune")
    if (train | tune) & review:
        raise ValueError(f"subsets {sorted((train | tune) & review)} belong to the review pool")
    if not (train | tune) <= dev:
        raise ValueError(f"subsets {sorted((train | tune) - dev)} are not dev subsets")


def build_model(n_classes: int, tcfg: dict, score_floor: float, pretrained: bool = True):
    """Faster R-CNN with a new box head for n_classes (+ background)."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=tcfg["min_size"],
        max_size=tcfg["max_size"],
        box_score_thresh=score_floor,
        box_detections_per_img=500,  # LOCO has up to 345 GT boxes per image
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes + 1)
    return model


class LocoDetection(torch.utils.data.Dataset):
    """Images + targets from the manifests. Labels are 1-based class indices."""

    def __init__(self, cfg: Config, images: pd.DataFrame, annotations: pd.DataFrame, hflip_prob: float = 0.0):
        self.root = cfg.raw_root
        self.classes = list(cfg.class_map.values())
        self.images = images.reset_index(drop=True)
        self.by_image = {k: g for k, g in annotations.groupby("image_id")}
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        r = self.images.iloc[i]
        with Image.open(self.root / r.rel_path) as im:
            image = to_tensor(im.convert("RGB"))
        a = self.by_image.get(r.image_id, pd.DataFrame(columns=["x_min", "y_min", "x_max", "y_max", "class_name"]))
        boxes = torch.tensor(a[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype="float32")).reshape(-1, 4)
        # Validation found no box past an edge by more than 0.5 px; clip so the
        # loss never sees coordinates outside the image.
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, r.width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, r.height)
        labels = torch.tensor([self.classes.index(c) + 1 for c in a["class_name"]], dtype=torch.int64)
        if self.hflip_prob and random.random() < self.hflip_prob:
            image, boxes = hflip(image), hflip_boxes(boxes, float(r.width))
        return image, {"boxes": boxes, "labels": labels, "image_id": r.image_id}


def _collate(batch):
    return tuple(zip(*batch))


def _memory_gb(device: str) -> str:
    """Live tensors vs everything the MPS driver holds (incl. the allocator cache)."""
    if device != "mps":
        return ""
    return f" mem {torch.mps.current_allocated_memory() / 1e9:.1f}/{torch.mps.driver_allocated_memory() / 1e9:.1f}GB"


def _release_cache(device: str) -> None:
    # The MPS allocator keeps freed blocks cached; with varied image sizes and
    # box counts the cache grew to ~14 GB in epoch 1 and pushed the OS into swap.
    if device == "mps":
        torch.mps.empty_cache()


def _lr_factor(it: int, epoch: int, tcfg: dict) -> float:
    warm = min(1.0, (it + 1) / tcfg["warmup_iters"])
    drops = sum(epoch >= e for e in tcfg["lr_drop_epochs"])
    return warm * (0.1**drops)


@torch.inference_mode()
def predict_images(model, dataset: LocoDetection, device: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for i in range(len(dataset)):
        image, target = dataset[i]
        out = model([image.to(device)])[0]
        for box, label, score in zip(out["boxes"].cpu().tolist(), out["labels"].tolist(), out["scores"].tolist()):
            x_min, y_min, x_max, y_max = box
            rows.append({"image_id": target["image_id"], "class_name": dataset.classes[label - 1], "score": score,
                         "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max})
    return pd.DataFrame(rows, columns=["image_id", "class_name", "score", "x_min", "y_min", "x_max", "y_max"])


def run_train(cfg: Config, device: str = "auto", max_iters: int | None = None) -> dict:
    """Train (or resume) and return the run summary. max_iters is for smoke tests."""
    check_training_split(cfg)
    tcfg = cfg.raw["training"]
    if tcfg["hflip_prob"] is None:
        raise ValueError("training.hflip_prob is not set (decision D7)")
    device = pick_device(device)
    random.seed(tcfg["seed"])
    torch.manual_seed(tcfg["seed"])

    manifest = cfg.artifacts_dir / "manifest"
    images = pd.read_parquet(manifest / "images.parquet")
    annotations = pd.read_parquet(manifest / "annotations.parquet")
    excluded = pd.read_parquet(manifest / "quality_findings.parquet").query("disposition == 'exclude'")
    bad_images = set(excluded.loc[excluded["ref_type"] == "image", "ref_id"])
    bad_anns = set(excluded.loc[excluded["ref_type"] == "annotation", "ref_id"])
    annotations = annotations[~annotations["ann_id"].isin(bad_anns)]
    annotated = images[(images["annotation_status"] == "annotated") & ~images["image_id"].isin(bad_images)]
    train_imgs = annotated[annotated["subset"].isin(cfg.train_subsets)].sort_values("image_id")
    tune_pool = images[images["subset"].isin(cfg.tune_subsets)].assign(split="dev")
    monitor_imgs = sample_dev_images(tune_pool, tcfg["monitor_images"], tcfg["seed"])

    train_ds = LocoDetection(cfg, train_imgs, annotations, hflip_prob=tcfg["hflip_prob"])
    monitor_ds = LocoDetection(cfg, monitor_imgs, annotations)
    shuffle_gen = torch.Generator()
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=_collate, generator=shuffle_gen
    )

    run_id = f"frcnn_{cfg.hash()}" if max_iters is None else f"smoke_{cfg.hash()}"
    out = cfg.artifacts_dir / "models" / run_id
    out.mkdir(parents=True, exist_ok=True)
    classes = list(cfg.class_map.values())
    model = build_model(len(classes), tcfg, cfg.raw["detector"]["score_floor"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=tcfg["lr"], momentum=tcfg["momentum"], weight_decay=tcfg["weight_decay"])

    history, start_epoch, it = [], 1, 0
    ckpts = sorted(out.glob("epoch_*.pt"))
    if ckpts:
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        history, start_epoch, it = state["history"], state["epoch"] + 1, state["iter"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["py_rng"])
        print(f"resuming after epoch {state['epoch']}", flush=True)

    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        shuffle_gen.manual_seed(tcfg["seed"] + epoch)  # same order per epoch, even after a resume
        model.train()
        t0, losses = time.perf_counter(), []
        for step, (imgs, targets) in enumerate(loader):
            for g in opt.param_groups:
                g["lr"] = tcfg["lr"] * _lr_factor(it, epoch, tcfg)
            imgs = [im.to(device) for im in imgs]
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
            loss = sum(model(imgs, targets).values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            it += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.3f}{_memory_gb(device)}", flush=True)
                _release_cache(device)
            if max_iters and step + 1 >= max_iters:
                break

        _release_cache(device)
        preds = predict_images(model, monitor_ds, device)
        gt = annotations[annotations["image_id"].isin(monitor_imgs["image_id"])]
        metrics = coco_metrics(monitor_imgs, gt, preds, classes)
        history.append({
            "epoch": epoch,
            "train_loss_mean": sum(losses) / len(losses),
            "train_minutes": round((time.perf_counter() - t0) / 60, 1),
            "monitor": metrics.round(4).to_dict(orient="records"),
        })
        print(metrics.round(3).to_string(index=False), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "iter": it,
                    "history": history, "torch_rng": torch.get_rng_state(), "py_rng": random.getstate()},
                   out / f"epoch_{epoch:02d}.pt")
        for old in sorted(out.glob("epoch_*.pt"))[:-1]:  # keep only the latest checkpoint
            old.unlink()
        if max_iters:
            break

    summary = {
        "run_id": run_id,
        "arch": tcfg["arch"],
        "pretrained": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "classes": classes,
        "train_subsets": cfg.train_subsets,
        "tune_subsets": cfg.tune_subsets,
        "n_train_images": len(train_imgs),
        "monitor_image_ids": monitor_imgs["image_id"].tolist(),
        "training": tcfg,
        "device": device,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "config_hash": cfg.hash(),
        "history": history,
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    return summary
