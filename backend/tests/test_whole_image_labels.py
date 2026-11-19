import pytest

from app.detectors.whole_image import resolve_binary_labels


def test_resolves_documented_artificial_human_mapping():
    ai_index, real_index = resolve_binary_labels({"0": "artificial", "1": "human"})
    assert (ai_index, real_index) == (0, 1)


def test_resolves_fake_real_mapping_in_reverse_order():
    ai_index, real_index = resolve_binary_labels({"0": "Real", "1": "Fake"})
    assert (ai_index, real_index) == (1, 0)


def test_rejects_ambiguous_or_non_binary_labels():
    with pytest.raises(ValueError):
        resolve_binary_labels({"0": "class 0", "1": "class 1"})


def test_rejects_duplicate_semantic_labels():
    with pytest.raises(ValueError):
        resolve_binary_labels({"0": "artificial", "1": "synthetic"})
