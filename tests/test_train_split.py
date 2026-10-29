import copy

import pytest

from wpfe.config import load_config
from wpfe.train import check_training_split

BASE = load_config()


def with_split(train, tune):
    raw = copy.deepcopy(BASE.raw)
    raw["training"]["train_subsets"], raw["training"]["tune_subsets"] = train, tune
    return type(BASE)(raw=raw, root=BASE.root)


def test_default_split_is_valid():
    check_training_split(BASE)


@pytest.mark.parametrize(
    "train, tune, message",
    [
        ([2, 3], [3], "both train and tune"),
        ([2, 1], [3], "review pool"),
        ([2], [4], "review pool"),
        ([2, 9], [3], "not dev"),
        ([], [3], "non-empty"),
    ],
)
def test_leaky_splits_are_rejected(train, tune, message):
    with pytest.raises(ValueError, match=message):
        check_training_split(with_split(train, tune))
