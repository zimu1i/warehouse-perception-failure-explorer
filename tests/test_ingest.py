"""Tests for path parsing, stable IDs and split assignment. Synthetic inputs only."""

import pytest

from wpfe.config import Config
from wpfe.ingest import assign_split, parse_rel_path, stable_image_id


def test_parse_path_with_session_and_nested_camera():
    meta = parse_rel_path("dataset/subset-2/2019-11-25_0/RealSense/color/0/1574.5.jpg")
    assert meta == {"subset": 2, "session": "2019-11-25_0", "camera": "RealSense/color/0", "timestamp": 1574.5}


def test_parse_flat_subset_one_path_has_no_session():
    meta = parse_rel_path("dataset/subset-1/1564565315.0281892.jpg")
    assert meta["subset"] == 1 and meta["session"] is None and meta["camera"] is None


def test_stable_id_depends_on_path_not_content():
    assert stable_image_id("dataset/subset-1/a.jpg") == stable_image_id("dataset/subset-1/a.jpg")
    assert stable_image_id("dataset/subset-1/a.jpg") != stable_image_id("dataset/subset-1/b.jpg")


def test_split_assignment_and_unassigned_subset_fails():
    cfg = Config(raw={"split": {"dev_subsets": [2], "review_subsets": [1]}}, root=None)
    assert assign_split(2, cfg) == "dev"
    assert assign_split(1, cfg) == "review"
    with pytest.raises(ValueError):
        assign_split(9, cfg)
