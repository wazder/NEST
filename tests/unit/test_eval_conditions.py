"""Tests for scripts/eval_conditions.py: statistics helpers, and an end-to-end
run (train a tiny model, then evaluate it) on synthetic data."""

import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_conditions import (  # noqa: E402
    compare_conditions,
    holm_correction,
    paired_permutation_test,
    sentence_errors,
)
from tests.synthetic_zuco import write_synthetic_zuco  # noqa: E402

import train_nest_v2 as tnv2  # noqa: E402


class TestSentenceErrors:
    @pytest.mark.unit
    def test_sums_match_corpus_rates(self):
        hyps = ["the cat sat", "a dog", "", "red mat far away"]
        refs = ["the cat sat on a mat", "a big dog", "ran", "red mat"]
        word_errors, word_lengths = sentence_errors(hyps, refs, "word")
        char_errors, char_lengths = sentence_errors(hyps, refs, "char")
        assert word_errors.sum() / word_lengths.sum() == pytest.approx(tnv2.compute_wer(hyps, refs))
        assert char_errors.sum() / char_lengths.sum() == pytest.approx(tnv2.compute_cer(hyps, refs))
        assert word_errors.tolist() == [3, 1, 1, 2]

    @pytest.mark.unit
    def test_unknown_unit(self):
        with pytest.raises(ValueError):
            sentence_errors(["a"], ["a"], "token")


class TestPermutationTest:
    @pytest.mark.unit
    def test_identical_conditions_give_p_one(self):
        errors = np.array([3, 1, 4, 1, 5])
        assert paired_permutation_test(errors, errors, 500, np.random.default_rng(0)) == 1.0

    @pytest.mark.unit
    def test_consistent_difference_gives_minimum_p(self):
        a = np.full(40, 5)
        b = np.full(40, 1)
        p = paired_permutation_test(a, b, 999, np.random.default_rng(0))
        assert p == pytest.approx(1 / 1000)

    @pytest.mark.unit
    def test_matches_exact_enumeration(self):
        a = np.array([4, 2, 5, 3, 6, 2, 4, 3])
        b = np.array([2, 2, 3, 4, 3, 1, 2, 3])
        diff = a - b
        observed = abs(diff.sum())
        all_signs = np.array(list(product([-1, 1], repeat=len(diff))))
        exact = np.mean(np.abs(all_signs @ diff) >= observed)
        p = paired_permutation_test(a, b, 20000, np.random.default_rng(1))
        assert p == pytest.approx(exact, abs=0.01)

    @pytest.mark.unit
    def test_symmetric_in_conditions(self):
        a, b = np.array([3, 0, 2, 5, 1]), np.array([1, 1, 2, 2, 0])
        p_ab = paired_permutation_test(a, b, 2000, np.random.default_rng(3))
        p_ba = paired_permutation_test(b, a, 2000, np.random.default_rng(3))
        assert p_ab == p_ba


class TestHolm:
    @pytest.mark.unit
    def test_known_values(self):
        # sorted: 0.01*4=0.04, 0.02*3=0.06, 0.03*2=0.06 (monotone), 0.5*1=0.5
        assert holm_correction([0.03, 0.01, 0.5, 0.02]) == pytest.approx([0.06, 0.04, 0.5, 0.06])

    @pytest.mark.unit
    def test_capped_at_one(self):
        assert holm_correction([0.4, 0.6]) == pytest.approx([0.8, 0.8])


@pytest.mark.unit
def test_compare_conditions_rows():
    rng = np.random.default_rng(0)
    refs = ["the cat sat on a mat"] * 20
    hyps = {
        "eeg": ["the cat sat on a mat"] * 20,
        "gaze": ["the cat sat"] * 20,
        "noise": [str(rng.choice(["dog", "a red dog ran"])) for _ in range(20)],
    }
    errors = {c: {"wer": sentence_errors(h, refs, "word"), "cer": sentence_errors(h, refs, "char")}
              for c, h in hyps.items()}
    rows = compare_conditions(errors, n_permutations=999, seed=0)
    assert [(r["a"], r["b"], r["metric"]) for r in rows] == [
        ("eeg", "gaze", "wer"), ("eeg", "gaze", "cer"),
        ("eeg", "noise", "wer"), ("eeg", "noise", "cer"),
        ("gaze", "noise", "wer"), ("gaze", "noise", "cer"),
    ]
    first = rows[0]
    assert first["rate_a"] == 0.0 and first["rate_b"] == pytest.approx(0.5)
    assert first["difference"] == pytest.approx(-0.5)
    assert first["p_value"] == pytest.approx(1 / 1000)
    assert all(r["p_holm"] >= r["p_value"] for r in rows)


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_tiny_ctc(tmp_path):
    """Train a tiny CTC model on synthetic data with EEG and gaze input, then
    run the three-condition evaluation on its checkpoint."""
    data_dir = tmp_path / "ZuCo"
    write_synthetic_zuco(data_dir, n_sentences=8)
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    run_dir = tmp_path / "run"
    train = subprocess.run(
        [sys.executable, "scripts/train_nest_v2.py", "--model", "ctc", "--data-dir", str(data_dir),
         "--tasks", "task1-SR", "--epochs", "2", "--batch-size", "4", "--grad-accum", "1",
         "--d-model", "32", "--num-layers", "1", "--nhead", "2",
         "--input-mode", "eeg", "gaze", "--output-dir", str(run_dir)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert train.returncode == 0, train.stderr[-3000:]
    config = json.loads((run_dir / "config.json").read_text())
    assert config["input_modes"] == ["eeg", "gaze"] and config["gaze_dim"] == 5
    assert set(json.loads((run_dir / "results.json").read_text())["test_by_input_mode"]) == {"eeg", "gaze"}

    output = tmp_path / "conditions.json"
    evaluate = subprocess.run(
        [sys.executable, "scripts/eval_conditions.py", "--checkpoint", str(run_dir / "best_model.pt"),
         "--data-dir", str(data_dir), "--tasks", "task1-SR", "--n-permutations", "500",
         "--output", str(output)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert evaluate.returncode == 0, evaluate.stderr[-3000:]
    print(evaluate.stdout)
    results = json.loads(output.read_text())
    assert set(results["conditions"]) == {"eeg", "gaze", "noise"}
    n = results["conditions"]["eeg"]["n"]
    assert n == 16  # 2 test subjects x 8 sentences
    assert len(results["references"]) == n
    assert all(len(h) == n for h in results["hypotheses"].values())
    assert len(results["comparisons"]) == 6
    for row in results["comparisons"]:
        assert 0 < row["p_value"] <= 1 and row["p_value"] <= row["p_holm"] <= 1
