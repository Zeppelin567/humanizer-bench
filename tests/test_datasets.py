"""Tests for datasets."""

import pytest

from humanizer_bench.datasets import Example, HFDataset, ToyDataset
from humanizer_bench.datasets.loaders import load_hf_dataset


def test_toy_loads_examples():
    examples = list(ToyDataset())
    assert len(examples) > 0
    assert all(isinstance(e, Example) for e in examples)


def test_labels_are_binary():
    assert all(e.label in (0, 1) for e in ToyDataset())


def test_has_both_classes():
    labels = {e.label for e in ToyDataset()}
    assert labels == {0, 1}


def test_iteration_is_repeatable():
    dataset = ToyDataset()
    first = [e.text for e in dataset]
    second = [e.text for e in dataset]
    assert first == second


def test_len_matches_iteration():
    dataset = ToyDataset()
    assert len(dataset) == len(list(dataset))


# --- HuggingFace loaders -------------------------------------------------------

def test_hf_dataset_constructs_without_downloading():
    # No Hub access until iterated: construction must stay cheap.
    ds = HFDataset("ag_news", label=0, limit=4)
    assert ds.name == "hf"
    assert ds.label == 0
    assert ds.source == "ag_news"


def test_hf_dataset_invalid_label_raises():
    with pytest.raises(ValueError):
        HFDataset("ag_news", label=2)


def test_hf_dataset_invalid_limit_raises():
    with pytest.raises(ValueError):
        HFDataset("ag_news", limit=-1)


@pytest.mark.models
def test_load_hf_dataset_tiny_slice():
    ds = load_hf_dataset(limit=4)
    examples = list(ds)
    assert len(examples) == 8
    assert all(isinstance(e, Example) for e in examples)
    assert {e.label for e in examples} == {0, 1}
    assert all(e.text.strip() for e in examples)
    assert {e.source for e in examples} == {"tatsu-lab/alpaca", "ag_news"}


@pytest.mark.models
def test_load_hf_dataset_iteration_repeatable():
    ds = load_hf_dataset(limit=4)
    assert [e.text for e in ds] == [e.text for e in ds]
