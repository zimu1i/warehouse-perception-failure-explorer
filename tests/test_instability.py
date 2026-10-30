import pytest

from wpfe.instability import instability

A = (0.0, 0.0, 10.0, 10.0)
B = (50.0, 0.0, 60.0, 10.0)
C = (0.0, 50.0, 10.0, 60.0)


def score(orig, flip):
    """orig / flip: [(box, cls, score)]"""
    return instability(
        [o[0] for o in orig], [o[1] for o in orig], [o[2] for o in orig],
        [f[0] for f in flip], [f[1] for f in flip], [f[2] for f in flip],
    )


def test_no_boxes_anywhere_is_zero():
    assert score([], []) == 0.0


def test_identical_views_are_stable():
    boxes = [(A, "pallet", 0.9), (B, "forklift", 0.4)]
    assert score(boxes, boxes) == pytest.approx(0.0)


def test_one_view_empty_is_fully_unstable():
    boxes = [(A, "pallet", 0.9), (B, "pallet", 0.4)]
    assert score(boxes, []) == pytest.approx(1.0)
    assert score([], boxes) == pytest.approx(1.0)


def test_worked_example():
    # A pairs (0.9 + 0.8); B only in orig (0.5) -> 0.5 / 2.2
    orig = [(A, "pallet", 0.9), (B, "pallet", 0.5)]
    flip = [(A, "pallet", 0.8)]
    assert score(orig, flip) == pytest.approx(0.5 / 2.2)


def test_unpaired_boxes_in_both_views_count():
    # A pairs; B only in orig (0.6); C only in flip (0.3) -> 0.9 / (1.0 + 0.6 + 1.0 + 0.3)
    orig = [(A, "pallet", 1.0), (B, "pallet", 0.6)]
    flip = [(A, "pallet", 1.0), (C, "pallet", 0.3)]
    assert score(orig, flip) == pytest.approx(0.9 / 2.9)


def test_class_change_breaks_the_pair():
    assert score([(A, "pallet", 0.9)], [(A, "stillage", 0.9)]) == pytest.approx(1.0)


def test_confident_vanishing_box_outranks_low_score_flicker():
    stable = [((10.0 * i, 0.0, 10.0 * i + 8, 8.0), "pallet", 0.8) for i in range(20)]
    flicker = [((10.0 * i, 30.0, 10.0 * i + 8, 38.0), "pallet", 0.06) for i in range(5)]
    dense_flicker = score(stable + flicker, stable)
    sparse_confident = score([(A, "pallet", 0.95), (B, "pallet", 0.9)], [(A, "pallet", 0.9)])
    assert sparse_confident > dense_flicker


def test_in_unit_interval():
    orig = [(A, "pallet", 0.7), (B, "forklift", 0.2)]
    flip = [(A, "pallet", 0.3), (C, "pallet", 0.9)]
    assert 0.0 <= score(orig, flip) <= 1.0


def test_score_images_keeps_images_without_predictions():
    import pandas as pd

    from wpfe.instability import score_images

    preds = pd.DataFrame([
        {"image_id": "a", "view": v, "class_name": "pallet", "score": 0.9,
         "x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0} for v in ("orig", "flip")
    ])
    out = score_images(preds, ["a", "empty"]).set_index("image_id")
    assert out.loc["a", "instability"] == pytest.approx(0.0)
    assert out.loc["empty", "instability"] == 0.0 and out.loc["empty", "n_orig"] == 0
