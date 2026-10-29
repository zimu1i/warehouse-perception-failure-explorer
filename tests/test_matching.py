import random

import pytest
import torch
from torchvision.ops import box_iou

from wpfe.matching import iou

A = (0.0, 0.0, 10.0, 10.0)


def test_identical_boxes():
    assert iou(A, A) == pytest.approx(1.0)


def test_half_overlap():
    # overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou(A, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(50 / 150)


def test_box_inside_box():
    assert iou(A, (0.0, 0.0, 5.0, 5.0)) == pytest.approx(25 / 100)


@pytest.mark.parametrize(
    "b",
    [
        (20.0, 0.0, 30.0, 10.0),  # far to the right
        (10.0, 0.0, 20.0, 10.0),  # touching edge only
        (0.0, 20.0, 10.0, 30.0),  # below
        (20.0, 20.0, 30.0, 30.0),  # diagonal: overlaps in neither axis
        (5.0, 20.0, 15.0, 30.0),  # overlaps in x but not in y
    ],
)
def test_no_overlap_is_zero(b):
    assert iou(A, b) == 0.0


def test_symmetric():
    b = (3.0, 4.0, 17.0, 9.0)
    assert iou(A, b) == pytest.approx(iou(b, A))


def test_matches_torchvision_on_random_boxes():
    rng = random.Random(0)
    for _ in range(200):
        boxes = []
        for _ in range(2):
            x1, y1 = rng.uniform(0, 100), rng.uniform(0, 100)
            boxes.append((x1, y1, x1 + rng.uniform(1, 60), y1 + rng.uniform(1, 60)))
        expected = box_iou(torch.tensor([boxes[0]]), torch.tensor([boxes[1]])).item()
        assert iou(*boxes) == pytest.approx(expected, abs=1e-6)


# --- match_image -------------------------------------------------------------

from wpfe.matching import match_image  # noqa: E402


def match(gt, preds, thr=0.5):
    """gt: [(box, cls)], preds: [(box, cls, score)] -> set of (pred_idx, gt_idx)."""
    return set(match_image(
        [g[0] for g in gt], [g[1] for g in gt],
        [p[0] for p in preds], [p[1] for p in preds], [p[2] for p in preds], thr,
    ))


def test_empty_inputs():
    assert match([], []) == set()
    assert match([(A, "pallet")], []) == set()
    assert match([], [(A, "pallet", 0.9)]) == set()


def test_perfect_match():
    assert match([(A, "pallet")], [(A, "pallet", 0.9)]) == {(0, 0)}


def test_class_must_agree():
    assert match([(A, "pallet")], [(A, "stillage", 0.9)]) == set()


def test_threshold_is_inclusive():
    # IoU of (0,0,10,10) and (0,0,10,20) is exactly 100 / 200 = 0.5
    assert match([(A, "pallet")], [((0.0, 0.0, 10.0, 20.0), "pallet", 0.9)]) == {(0, 0)}
    assert match([(A, "pallet")], [((0.0, 0.0, 10.0, 20.0), "pallet", 0.9)], thr=0.51) == set()


def test_one_big_box_over_a_stack_finds_at_most_one():
    stack = [((10.0 * i, 0.0, 10.0 * i + 10, 10.0), "pallet") for i in range(4)]
    big = ((0.0, 0.0, 40.0, 10.0), "pallet", 0.9)  # IoU 0.25 with each pallet
    assert match(stack, [big]) == set()
    two_wide = ((0.0, 0.0, 15.0, 10.0), "pallet", 0.9)  # IoU 0.67 with p0, 0.25 with p1
    assert match(stack, [two_wide]) == {(0, 0)}


def test_duplicate_prediction_does_not_find_a_second_object():
    gt = [(A, "pallet")]
    preds = [(A, "pallet", 0.6), ((0.5, 0.0, 10.5, 10.0), "pallet", 0.9)]
    assert match(gt, preds) == {(1, 0)}  # higher score claims it; the other is a duplicate


def test_prediction_claims_its_best_overlap():
    g1, g2 = (0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)
    pred = ((3.0, 0.0, 23.0, 10.0), "pallet", 0.9)  # IoU 7/23 with g1, 3/27 with g2: neither
    assert match([(g1, "pallet"), (g2, "pallet")], [pred]) == set()
    pred = ((19.0, 0.0, 29.0, 10.0), "pallet", 0.9)  # IoU 9/11 with g2
    assert match([(g1, "pallet"), (g2, "pallet")], [pred]) == {(0, 1)}


def test_greedy_by_score_not_globally_optimal():
    # Documents the known limitation shared with pycocotools.
    g1, g2 = (0.0, 0.0, 10.0, 10.0), (2.0, 0.0, 12.0, 10.0)
    a = ((0.5, 0.0, 10.5, 10.0), "pallet", 0.9)  # IoU 0.90 with g1, 0.74 with g2
    b = ((-2.0, 0.0, 8.0, 10.0), "pallet", 0.5)  # IoU 0.67 with g1, 0.43 with g2
    # a goes first and takes g1; b only overlaps g1 enough, so it is left out.
    # An optimal assignment (a->g2, b->g1) would have found both.
    assert match([(g1, "pallet"), (g2, "pallet")], [a, b]) == {(0, 0)}
