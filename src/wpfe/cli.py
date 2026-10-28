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


if __name__ == "__main__":
    main()
