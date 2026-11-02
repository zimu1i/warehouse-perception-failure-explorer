"""Command-line entry point: `uv run wpfe <command>`."""

from __future__ import annotations

import argparse
import json

from wpfe.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wpfe")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="build image and annotation manifests from the raw release")
    sub.add_parser("validate", help="run data-quality checks and write quality_findings.parquet")
    bench = sub.add_parser("benchmark", help="run a detector on a small dev sample and report speed and COCO metrics")
    bench.add_argument("--model", required=True, choices=["grounding_dino_tiny", "owlv2_base"])
    bench.add_argument("--prompts", default="literal")
    bench.add_argument("--device", default="auto")
    train = sub.add_parser("train", help="fine-tune Faster R-CNN on the dev training subsets (resumable)")
    train.add_argument("--device", default="auto")
    train.add_argument("--smoke", type=int, metavar="N", help="stop after N iterations (pipeline check only)")
    infer = sub.add_parser("infer", help="score tune + review images (original and flipped) with the trained model")
    infer.add_argument("--run", help="model run id (default: frozen_model in the config)")
    infer.add_argument("--device", default="auto")
    infer.add_argument("--smoke", type=int, metavar="N", help="only the first N images (pipeline check only)")
    sub.add_parser("experiment", help="run the frozen selection experiment once (reads review labels)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.command == "ingest":
        from wpfe.ingest import run_ingest

        print(json.dumps(run_ingest(cfg), indent=2))
    elif args.command == "validate":
        from wpfe.validate import run_validate

        findings = run_validate(cfg)
        summary = findings.groupby(["severity", "rule"]).size().rename("n").reset_index()
        print(summary.to_string(index=False) if len(summary) else "no findings")
    elif args.command == "benchmark":
        from wpfe.benchmark import run_benchmark

        result = run_benchmark(cfg, args.model, args.prompts, args.device)
        s = result["summary"]
        keys = ["run_id", "device", "load_s", "sec_per_image_median", "mps_driver_mem_gb", "n_preds", "n_unmapped_preds", "unmapped_phrases", "n_gt"]
        print(json.dumps({k: s[k] for k in keys}, indent=2))
        print(result["metrics"].round(3).to_string(index=False))
    elif args.command == "train":
        from wpfe.train import run_train

        summary = run_train(cfg, args.device, max_iters=args.smoke)
        print(json.dumps({k: summary[k] for k in ["run_id", "n_train_images", "device"]}, indent=2))
    elif args.command == "infer":
        from wpfe.infer import run_infer

        summary = run_infer(cfg, args.run or cfg.raw["frozen_model"], args.device, limit=args.smoke)
        print(json.dumps(summary, indent=2))
    elif args.command == "experiment":
        from wpfe.experiment import run_experiment

        result = run_experiment(cfg)
        print(json.dumps(result["summary"], indent=2))
        print(result["results"].round(3).to_string(index=False))
        print("\npost-hoc density check:\n" + result["exploratory"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
