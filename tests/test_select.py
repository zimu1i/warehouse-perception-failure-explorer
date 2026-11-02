import pandas as pd

from wpfe.select import select_diverse, select_random, select_top


def cands(rows):
    """rows: (image_id, seq, pos, instability)"""
    return pd.DataFrame(rows, columns=["image_id", "seq", "pos", "instability"]).assign(subset=1)


ONE_SEQ = cands([("a", "s", 0, 0.9), ("b", "s", 1, 0.8), ("c", "s", 2, 0.7), ("d", "s", 3, 0.6), ("e", "s", 9, 0.5)])


def test_top_is_sorted_by_instability():
    assert select_top(ONE_SEQ, 3) == ["a", "b", "c"]


def test_top_breaks_ties_by_image_id():
    tied = cands([("z", "s", 0, 0.5), ("m", "s", 5, 0.5), ("a", "s", 9, 0.5)])
    assert select_top(tied, 3) == ["a", "m", "z"]


def test_random_is_reproducible_and_without_repeats():
    a, b = select_random(ONE_SEQ, 3, seed=7), select_random(ONE_SEQ, 3, seed=7)
    assert a == b and len(set(a)) == 3


def test_gap_zero_equals_top():
    assert select_diverse(ONE_SEQ, 4, gap=0) == select_top(ONE_SEQ, 4)


def test_neighbours_of_a_pick_are_skipped():
    # a (pos 0) is picked; b (pos 1) is within 1; c (pos 2) is 2 away from a -> picked; d next to c -> skipped
    assert select_diverse(ONE_SEQ, 5, gap=1) == ["a", "c", "e"]


def test_skipped_images_do_not_block_others():
    # with gap 1, c is 1 away from the *skipped* b but 2 away from the picked a, so c is allowed
    assert "c" in select_diverse(ONE_SEQ, 5, gap=1)


def test_gap_is_inclusive():
    assert select_diverse(ONE_SEQ, 5, gap=2) == ["a", "d", "e"]  # b, c within 2 of a; d is 3 away


def test_other_sequences_are_not_blocked():
    two = cands([("a", "s1", 0, 0.9), ("b", "s2", 0, 0.8), ("c", "s1", 1, 0.7)])
    assert select_diverse(two, 3, gap=2) == ["a", "b"]


def test_stops_at_k():
    assert select_diverse(ONE_SEQ, 2, gap=1) == ["a", "c"]


def test_does_not_read_labels():
    # selectors must work on the label-free candidate columns alone
    assert select_diverse(ONE_SEQ[["image_id", "seq", "pos", "instability"]], 2, gap=1) == ["a", "c"]
