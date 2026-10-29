"""Load the YAML config and resolve repo-relative paths."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict
    root: Path

    @property
    def raw_root(self) -> Path:
        return self.root / self.raw["dataset"]["raw_root"]

    @property
    def annotation_path(self) -> Path:
        return self.raw_root / self.raw["dataset"]["annotation_file"]

    @property
    def artifacts_dir(self) -> Path:
        return self.root / self.raw["artifacts_dir"]

    @property
    def release(self) -> str:
        return self.raw["dataset"]["release"]

    @property
    def class_map(self) -> dict[int, str]:
        return {int(k): v for k, v in self.raw["classes"].items()}

    @property
    def dev_subsets(self) -> list[int]:
        return list(self.raw["split"]["dev_subsets"])

    @property
    def review_subsets(self) -> list[int]:
        return list(self.raw["split"]["review_subsets"])

    @property
    def train_subsets(self) -> list[int]:
        return list(self.raw["training"]["train_subsets"])

    @property
    def tune_subsets(self) -> list[int]:
        return list(self.raw["training"]["tune_subsets"])

    def hash(self) -> str:
        """Stable hash of the config content, recorded with every artifact."""
        text = yaml.safe_dump(self.raw, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_config(path: str | Path = "configs/default.yaml", root: Path = REPO_ROOT) -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    with open(path) as f:
        return Config(raw=yaml.safe_load(f), root=root)
