"""Unit tests for scripts/zuco_mat_to_pickle.py with synthetic .mat-like structs."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from zuco_mat_to_pickle import (  # noqa: E402
    FREQ_BANDS,
    convert_sentence,
    is_fixated,
    subject_name,
)

EMPTY = np.array([])


def make_word(content: str, n_fixations, ffd=EMPTY, gd=EMPTY, trt=EMPTY, gpt=EMPTY) -> SimpleNamespace:
    """A word struct as scipy.io.loadmat(squeeze_me=True, struct_as_record=False) returns it."""
    fields = {"content": content, "nFixations": n_fixations, "FFD": ffd, "GD": gd, "TRT": trt, "GPT": gpt}
    for measure in ["FFD", "TRT", "GD"]:
        for i, band in enumerate(FREQ_BANDS):
            fields[f"{measure}_{band}"] = np.full(105, i, dtype=np.float64)
    return SimpleNamespace(**fields)


def make_sentence(words) -> SimpleNamespace:
    fields = {"content": "a b c", "word": words}
    for band in FREQ_BANDS:
        fields[f"mean_{band}"] = np.zeros(105)
        fields[f"answer_mean_{band}"] = np.zeros(105)
    return SimpleNamespace(**fields)


def test_is_fixated_handles_empty_and_zero():
    assert not is_fixated(make_word("a", EMPTY))
    assert not is_fixated(make_word("a", 0))
    assert is_fixated(make_word("a", 2))


def test_convert_sentence_keeps_gaze_and_skips_unfixated_words():
    words = [
        make_word("Fixated", 2, ffd=107, gd=120, trt=300, gpt=150),
        make_word("skipped", EMPTY),
        make_word("again", 1, ffd=90, gd=90, trt=90, gpt=95),
    ]
    sent = convert_sentence(make_sentence(np.array(words, dtype=object)), "task1-SR")

    assert [w["content"] for w in sent["word"]] == ["Fixated", "again"]
    first = sent["word"][0]
    assert first["nFixations"] == 2
    assert (first["FFD"], first["GD"], first["TRT"], first["GPT"]) == (107.0, 120.0, 300.0, 150.0)
    assert set(first["word_level_EEG"]) == {"FFD", "TRT", "GD"}
    assert np.array_equal(first["word_level_EEG"]["GD"]["GD_a1"], np.full(105, 2.0))
    assert sent["word_tokens_with_mask"] == ["Fixated", "[MASK]", "again"]
    assert sent["word_tokens_has_fixation"] == ["Fixated", "again"]
    assert sent["word_tokens_all"] == ["Fixated", "skipped", "again"]
    assert "answer_EEG" in sent


def test_convert_sentence_returns_none_without_word_data():
    assert convert_sentence(make_sentence(float("nan")), "task2-NR") is None


def test_answer_eeg_only_for_task1():
    sent = convert_sentence(make_sentence(np.array([make_word("a", 1, 1, 1, 1, 1)], dtype=object)), "task2-NR")
    assert "answer_EEG" not in sent


def test_subject_name():
    assert subject_name(Path("resultsZAB_SR.mat")) == "ZAB"
