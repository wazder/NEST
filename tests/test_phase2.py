"""
Smoke tests for the Phase 2 preprocessing modules.

These tests check that all preprocessing modules can be imported and that
their basic functionality works.
"""

import numpy as np
import pytest

from src.data import ZuCoDataset
from src.preprocessing import (
    EEGFilter,
    ICArtifactRemoval,
    ElectrodeSelector,
    EEGAugmentation,
    EEGDataSplitter,
    PreprocessingPipeline
)


@pytest.fixture
def test_data():
    np.random.seed(42)
    return np.random.randn(32, 1000) * 10e-6


def test_zuco_dataset(tmp_path):
    dataset = ZuCoDataset(data_dir=str(tmp_path / "zuco"))
    info = dataset.get_dataset_info()

    assert "data_directory" in info


def test_eeg_filter(test_data):
    eeg_filter = EEGFilter(l_freq=0.5, h_freq=50.0, sfreq=500.0)
    filtered = eeg_filter.bandpass_filter(test_data)

    assert filtered.shape == test_data.shape


def test_ica_artifact_removal(test_data):
    ica = ICArtifactRemoval(n_components=10, method='fastica')
    ica.fit(test_data, sfreq=500.0)

    assert ica.ica.n_components_ is not None


def test_electrode_selector(test_data):
    ch_names = [f'EEG{i:03d}' for i in range(32)]
    selector = ElectrodeSelector(ch_names)

    selected_indices, _ = selector.select_by_variance(test_data, n_channels=16)
    selected_data, selected_names = selector.apply_selection(test_data, selected_indices)

    assert selected_data.shape[0] == 16


def test_eeg_augmentation(test_data):
    augmenter = EEGAugmentation(random_state=42)

    noisy = augmenter.add_gaussian_noise(test_data, noise_level=0.1)
    scaled = augmenter.amplitude_scaling(test_data)

    assert noisy.shape == test_data.shape
    assert scaled.shape == test_data.shape


def test_eeg_data_splitter():
    splitter = EEGDataSplitter(random_state=42)

    # Create test dataset
    n_samples = 100
    data = np.random.randn(n_samples, 32, 1000)
    labels = np.random.randint(0, 2, n_samples)
    subject_ids = np.repeat(np.arange(10), 10)

    splits = splitter.subject_independent_split(
        data, labels, subject_ids,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    for split in ("train", "val", "test"):
        assert splits[split][0].shape[0] > 0


def test_preprocessing_pipeline():
    pipeline = PreprocessingPipeline()

    assert pipeline.config is not None
