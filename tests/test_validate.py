"""Tests for box validation. Synthetic values only; never real dataset files."""

import math

import pytest

from wpfe.validate import check_box

# A box that is valid on its own (width 90, height 40), so each test can break exactly one thing.
GOOD_BOX = (10.0, 10.0, 100.0, 50.0)


def rules(findings):
    return {f.rule for f in findings}


def test_clean_box_has_no_findings():
    assert check_box(10, 20, 110, 220, width=640, height=480) == []


def test_full_image_box_is_valid():
    # Continuous xyxy: the full-image box ends exactly at (width, height).
    assert check_box(0, 0, 640, 480, width=640, height=480) == []


# Test inverted/broken box where x_max <= x_min or y_max <= y_min
def test_inverted_box():
    findings = check_box(100, 20, 0, 5, width=640, height=480)
    assert rules(findings) == {"non_positive_size"}


def test_zero_width_is_non_positive_size():
    findings = check_box(10, 10, 10, 50, width=640, height=480)
    assert rules(findings) == {"non_positive_size"}


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_any_non_finite_coord_is_caught(position, bad_value):
    coords = list(GOOD_BOX)
    coords[position] = bad_value
    findings = check_box(*coords, width=640, height=480)
    assert rules(findings) == {"non_finite_coords"}

@pytest.mark.parametrize(
    "box, expected",
    [
        ((0, 0, 1005, 100), set()),                      # 0.5%: within tolerance
        ((0, 0, 1050, 100), {"box_overshoot_minor"}),    # 5% right
        ((0, 0, 1200, 100), {"box_overshoot_major"}),    # 20% right
        ((1000, 0, 1100, 100), {"box_fully_outside"}),   # starts exactly at the right edge
        ((-50, 0, 100, 100), {"box_overshoot_minor"}),   # 5% left
        ((0, 0, 2000, 100), {"box_overshoot_major"}),    # covers the image, then 1000 px more
    ],
)
def test_edge_rules(box, expected):
    findings = check_box(*box, width=1000, height=1000)
    assert rules(findings) == expected


def test_annotation_pipeline_reports_corrupted_fixture():
    # Guards against "no findings" meaning "checks never ran": a deliberately corrupted
    # synthetic fixture must surface every problem through check_annotations.
    import pandas as pd

    from wpfe.validate import check_annotations

    images = pd.DataFrame({"image_id": ["img_a"], "width": [1000], "height": [1000]})
    annotations = pd.DataFrame(
        {
            "ann_id": ["ok", "outside", "orphan", "ok"],
            "image_id": ["img_a", "img_a", "img_missing", "img_a"],
            "x_min": [10.0, 1000.0, 10.0, 10.0],
            "y_min": [10.0, 0.0, 10.0, 10.0],
            "x_max": [100.0, 1100.0, 100.0, 100.0],
            "y_max": [50.0, 100.0, 50.0, 50.0],
        }
    )
    found = set(check_annotations(annotations, images)[["ref_id", "rule"]].itertuples(index=False, name=None))
    assert found == {
        ("ok", "duplicate_ann_id"),
        ("outside", "box_fully_outside"),
        ("orphan", "orphan_annotation"),
    }
