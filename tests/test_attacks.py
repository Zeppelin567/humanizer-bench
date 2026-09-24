"""Tests for attacks."""

import pytest

from humanizer_bench.attacks import (
    BackTranslationAttack,
    NoiseAttack,
    SentenceMergeAttack,
)


# --- NoiseAttack ---------------------------------------------------------------

def test_noise_transform_returns_str():
    attack = NoiseAttack(rate=0.1, seed=0)
    out = attack.transform("The quick brown fox jumps over the lazy dog.")
    assert isinstance(out, str)


def test_noise_deterministic_with_seed():
    text = "The quick brown fox jumps over the lazy dog."
    a = NoiseAttack(rate=0.3, seed=42).transform(text)
    b = NoiseAttack(rate=0.3, seed=42).transform(text)
    assert a == b


def test_noise_zero_rate_is_identity():
    text = "No changes should happen here."
    assert NoiseAttack(rate=0.0).transform(text) == text


def test_noise_invalid_rate_raises():
    with pytest.raises(ValueError):
        NoiseAttack(rate=1.5)


# --- SentenceMergeAttack -------------------------------------------------------

def test_merge_returns_str():
    attack = SentenceMergeAttack(rate=1.0)
    out = attack.transform("First sentence here. Second sentence here. Third one too.")
    assert isinstance(out, str)


def test_merge_zero_rate_is_identity():
    text = "First sentence here. Second sentence here."
    assert SentenceMergeAttack(rate=0.0).transform(text) == text


def test_merge_reduces_sentence_count():
    text = "One two three four. Five six seven eight. Nine ten eleven twelve."
    merged = SentenceMergeAttack(rate=1.0).transform(text)
    assert merged.count(".") < text.count(".")


def test_merge_single_sentence_unchanged():
    text = "Only one sentence with no merge possible."
    assert SentenceMergeAttack(rate=1.0).transform(text) == text


def test_merge_invalid_rate_raises():
    with pytest.raises(ValueError):
        SentenceMergeAttack(rate=-0.1)


# --- shared / callable ---------------------------------------------------------

def test_callable_alias():
    attack = SentenceMergeAttack(rate=1.0)
    text = "First sentence here. Second sentence here."
    assert attack(text) == SentenceMergeAttack(rate=1.0).transform(text)


# --- BackTranslationAttack -----------------------------------------------------

def test_back_translation_registers_without_models():
    # Constructing must stay cheap: no heavy imports until transform() runs.
    from humanizer_bench.registry import ATTACKS

    assert ATTACKS["back_translation"] is BackTranslationAttack
    attack = BackTranslationAttack()
    assert attack.name == "back_translation"
    assert attack.pivot == "fr"


def test_back_translation_empty_passthrough():
    # Blank input short-circuits before any model is touched.
    assert BackTranslationAttack().transform("   ") == "   "


@pytest.mark.models
def test_back_translation_returns_str():
    attack = BackTranslationAttack()
    out = attack.transform("The quick brown fox jumps over the lazy dog.")
    assert isinstance(out, str)
    assert out.strip()


@pytest.mark.models
def test_back_translation_round_trip_changes_text():
    # Complex phrasing gets paraphrased; very simple sentences can round-trip
    # intact, so this uses one with enough idiom to shift.
    text = (
        "The committee convened to deliberate on the ramifications "
        "of the proposed legislation."
    )
    assert BackTranslationAttack().transform(text) != text


@pytest.mark.models
def test_back_translation_deterministic():
    text = "The quick brown fox jumps over the lazy dog."
    first = BackTranslationAttack().transform(text)
    second = BackTranslationAttack().transform(text)
    assert first == second
