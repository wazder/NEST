"""Unit tests for scripts/eval_noise_control.py's pure helper functions.

These tests exercise only the noise-generation, statistics-estimation,
delta-computation, and dataset-wrapper helpers. They need neither the ZuCo
dataset nor a trained checkpoint.
"""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from eval_noise_control import (  # noqa: E402
    GaussianNoiseEEGDataset,
    ShuffledEEGDataset,
    compute_delta,
    derangement,
    estimate_feature_stats,
    generate_gaussian_noise,
)

EEG_DIM = 840


class FakeWordDataset:
    """Minimal stand-in for ZuCoWordSequenceDataset: each sample's "eeg" is
    filled with its own index so provenance can be checked after shuffling,
    without needing real ZuCo data."""

    def __init__(self, n_samples: int, n_words: int = 5, eeg_dim: int = EEG_DIM):
        self.samples: List[Dict] = []
        for i in range(n_samples):
            eeg = np.full((n_words, eeg_dim), float(i), dtype=np.float32)
            self.samples.append({"eeg": eeg, "eeg_len": n_words, "text": f"sentence {i}"})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class TestEstimateFeatureStats:
    @pytest.mark.unit
    def test_shape(self):
        arrays = [np.random.randn(5, EEG_DIM).astype(np.float32) for _ in range(3)]
        mean, std = estimate_feature_stats(arrays)
        assert mean.shape == (EEG_DIM,)
        assert std.shape == (EEG_DIM,)

    @pytest.mark.unit
    def test_matches_known_values(self):
        arrays = [np.zeros((1, 4), dtype=np.float32), np.full((1, 4), 10.0, dtype=np.float32)]
        mean, std = estimate_feature_stats(arrays)
        assert np.allclose(mean, 5.0)
        assert np.allclose(std, 5.0)  # population std of {0, 10}

    @pytest.mark.unit
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            estimate_feature_stats([])


class TestGenerateGaussianNoise:
    @pytest.mark.unit
    def test_shape(self):
        rng = np.random.default_rng(0)
        mean = np.zeros(EEG_DIM, dtype=np.float32)
        std = np.ones(EEG_DIM, dtype=np.float32)
        noise = generate_gaussian_noise(7, mean, std, rng)
        assert noise.shape == (7, EEG_DIM)
        assert noise.dtype == np.float32

    @pytest.mark.unit
    def test_statistics_match_target(self):
        rng = np.random.default_rng(123)
        mean = np.full(EEG_DIM, 3.0, dtype=np.float32)
        std = np.full(EEG_DIM, 2.0, dtype=np.float32)
        noise = generate_gaussian_noise(20000, mean, std, rng)
        assert np.allclose(noise.mean(axis=0), mean, atol=0.15)
        assert np.allclose(noise.std(axis=0), std, atol=0.15)

    @pytest.mark.unit
    def test_per_feature_statistics_are_independent(self):
        rng = np.random.default_rng(7)
        mean = np.array([0.0, 100.0], dtype=np.float32)
        std = np.array([1.0, 0.01], dtype=np.float32)
        noise = generate_gaussian_noise(5000, mean, std, rng)
        assert abs(noise[:, 0].mean() - 0.0) < 0.1
        assert abs(noise[:, 1].mean() - 100.0) < 0.01
        assert noise[:, 0].std() > noise[:, 1].std()

    @pytest.mark.unit
    def test_mismatched_shapes_raise(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            generate_gaussian_noise(5, np.zeros(4), np.zeros(3), rng)

    @pytest.mark.unit
    def test_deterministic_with_seeded_rng(self):
        mean = np.zeros(EEG_DIM, dtype=np.float32)
        std = np.ones(EEG_DIM, dtype=np.float32)
        noise_a = generate_gaussian_noise(4, mean, std, np.random.default_rng(99))
        noise_b = generate_gaussian_noise(4, mean, std, np.random.default_rng(99))
        assert np.array_equal(noise_a, noise_b)


class TestDerangement:
    @pytest.mark.unit
    @pytest.mark.parametrize("n", [2, 3, 5, 10, 50])
    def test_is_permutation_with_no_fixed_points(self, n):
        rng = np.random.default_rng(n)
        perm = derangement(n, rng)
        assert sorted(perm.tolist()) == list(range(n))  # bijection: multiset preserved
        assert np.all(perm != np.arange(n))  # no fixed points

    @pytest.mark.unit
    def test_raises_for_n_below_2(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            derangement(1, rng)
        with pytest.raises(ValueError):
            derangement(0, rng)


class TestComputeDelta:
    @pytest.mark.unit
    def test_basic_difference(self):
        real = {"wer": 0.5, "cer": 0.3, "n": 100}
        noise = {"wer": 0.48, "cer": 0.31, "n": 100}
        delta = compute_delta(real, noise)
        assert delta["wer"] == pytest.approx(-0.02)
        assert delta["cer"] == pytest.approx(0.01)
        assert "n" not in delta  # sample count excluded, not a metric

    @pytest.mark.unit
    def test_zero_delta_when_equal(self):
        real = {"wer": 0.4, "cer": 0.2}
        noise = {"wer": 0.4, "cer": 0.2}
        assert compute_delta(real, noise) == {"wer": 0.0, "cer": 0.0}

    @pytest.mark.unit
    def test_only_shared_numeric_keys(self):
        real = {"wer": 0.5, "loss": 1.2, "label": "real"}
        noise = {"wer": 0.6, "extra": 3.0}
        delta = compute_delta(real, noise)
        assert delta == {"wer": pytest.approx(0.1)}

    @pytest.mark.unit
    def test_ignores_bool_values(self):
        real = {"flag": True, "wer": 0.5}
        noise = {"flag": False, "wer": 0.5}
        delta = compute_delta(real, noise)
        assert "flag" not in delta
        assert delta["wer"] == pytest.approx(0.0)


class TestGaussianNoiseEEGDataset:
    @pytest.mark.unit
    def test_shape_and_metadata_preserved(self):
        base = FakeWordDataset(n_samples=6, n_words=5)
        mean = np.zeros(EEG_DIM, dtype=np.float32)
        std = np.ones(EEG_DIM, dtype=np.float32)
        noise_ds = GaussianNoiseEEGDataset(base, mean=mean, std=std, seed=1)

        assert len(noise_ds) == len(base)
        for i in range(len(noise_ds)):
            sample = noise_ds[i]
            assert sample["eeg"].shape == base[i]["eeg"].shape
            assert sample["eeg_len"] == base[i]["eeg_len"]
            assert sample["text"] == base[i]["text"]
            assert not np.allclose(sample["eeg"], base[i]["eeg"])

    @pytest.mark.unit
    def test_statistics_over_many_samples(self):
        base = FakeWordDataset(n_samples=500, n_words=4)
        mean = np.full(EEG_DIM, 2.0, dtype=np.float32)
        std = np.full(EEG_DIM, 1.0, dtype=np.float32)
        noise_ds = GaussianNoiseEEGDataset(base, mean=mean, std=std, seed=42)

        all_noise = np.concatenate([noise_ds[i]["eeg"] for i in range(len(noise_ds))], axis=0)
        assert np.allclose(all_noise.mean(axis=0), mean, atol=0.3)
        assert np.allclose(all_noise.std(axis=0), std, atol=0.3)

    @pytest.mark.unit
    def test_deterministic_given_seed(self):
        base = FakeWordDataset(n_samples=3, n_words=4)
        mean = np.zeros(EEG_DIM, dtype=np.float32)
        std = np.ones(EEG_DIM, dtype=np.float32)
        ds_a = GaussianNoiseEEGDataset(base, mean=mean, std=std, seed=5)
        ds_b = GaussianNoiseEEGDataset(base, mean=mean, std=std, seed=5)
        for i in range(len(base)):
            assert np.array_equal(ds_a[i]["eeg"], ds_b[i]["eeg"])


class TestShuffledEEGDataset:
    @pytest.mark.unit
    def test_preserves_multiset_of_sequences(self):
        base = FakeWordDataset(n_samples=8, n_words=3)
        perm = derangement(len(base), np.random.default_rng(0))
        shuffled_ds = ShuffledEEGDataset(base, perm)

        # Each output's "eeg" is filled with a single provenance index —
        # recover it and check the multiset of provenance indices exactly
        # equals {0, ..., n-1}: no sequence is duplicated, none is lost.
        provenance = [int(shuffled_ds[i]["eeg"][0, 0]) for i in range(len(base))]
        assert sorted(provenance) == list(range(len(base)))

    @pytest.mark.unit
    def test_no_sample_keeps_its_own_eeg(self):
        base = FakeWordDataset(n_samples=8, n_words=3)
        perm = derangement(len(base), np.random.default_rng(1))
        shuffled_ds = ShuffledEEGDataset(base, perm)

        for i in range(len(base)):
            provenance = int(shuffled_ds[i]["eeg"][0, 0])
            assert provenance != i

    @pytest.mark.unit
    def test_labels_stay_at_original_index(self):
        base = FakeWordDataset(n_samples=6, n_words=3)
        perm = derangement(len(base), np.random.default_rng(2))
        shuffled_ds = ShuffledEEGDataset(base, perm)

        for i in range(len(base)):
            assert shuffled_ds[i]["text"] == base[i]["text"]

    @pytest.mark.unit
    def test_eeg_len_matches_substituted_sequence(self):
        base = FakeWordDataset(n_samples=5, n_words=3)
        # Give each sample a distinct eeg_len to verify it travels with the eeg.
        for i, sample in enumerate(base.samples):
            sample["eeg_len"] = i + 1
            sample["eeg"] = np.full((i + 1, EEG_DIM), float(i), dtype=np.float32)
        perm = derangement(len(base), np.random.default_rng(3))
        shuffled_ds = ShuffledEEGDataset(base, perm)

        for i in range(len(base)):
            sample = shuffled_ds[i]
            assert sample["eeg"].shape[0] == sample["eeg_len"]
            assert sample["eeg_len"] == base[int(perm[i])]["eeg_len"]

    @pytest.mark.unit
    def test_mismatched_permutation_length_raises(self):
        base = FakeWordDataset(n_samples=5, n_words=3)
        with pytest.raises(ValueError):
            ShuffledEEGDataset(base, np.array([0, 1, 2]))
