"""Archived v1 tests from tests/unit/test_preprocessing.py.

Not collected: they call APIs that src/ does not provide.
See README.md in this folder.
"""
import pytest
import numpy as np
import torch

from src.preprocessing.filtering import (
    bandpass_filter,
    notch_filter,
    extract_frequency_bands,
    apply_car,
    laplacian_filter,
)
from src.preprocessing.artifact_removal import (
    ICARemover,
    ThresholdRejecter,
)
from src.preprocessing.electrode_selection import (
    ElectrodeSelector,
    select_by_names,
    select_by_variance,
    select_by_mutual_information,
)
from src.preprocessing.augmentation import (
    add_gaussian_noise,
    scale_amplitude,
    time_shift,
    time_warp,
    channel_dropout,
)
from src.preprocessing.data_split import (
    train_val_test_split,
    subject_aware_split,
    temporal_split,
)


class TestArtifactRemoval:
    """Tests for artifact removal methods."""
    
    @pytest.mark.unit
    def test_variance_rejection(self, sample_eeg_data):
        """Test variance-based channel rejection."""
        eeg, _ = sample_eeg_data
        
        rejecter = ThresholdRejecter(variance_threshold=10)
        bad_channels = rejecter.detect_bad_channels(eeg)
        
        assert isinstance(bad_channels, list)


class TestElectrodeSelection:
    """Tests for electrode selection methods."""
    
    @pytest.mark.unit
    def test_pca_reduction(self, sample_eeg_data):
        """Test PCA-based dimensionality reduction."""
        eeg, _ = sample_eeg_data
        
        selector = ElectrodeSelector(method="pca", n_channels=50)
        reduced = selector.fit_transform(eeg)
        
        assert reduced.shape[0] == 50
        assert reduced.shape[1] == eeg.shape[1]


class TestDataSplit:
    """Tests for data splitting strategies."""
    
    @pytest.mark.unit
    def test_basic_split(self, sample_eeg_data, sample_text_data):
        """Test basic train/val/test split."""
        eeg, _ = sample_eeg_data
        texts = sample_text_data["texts"]
        
        # Create dataset list
        dataset = [(eeg[:, i:i+500], texts[i % len(texts)]) 
                   for i in range(0, eeg.shape[1] - 500, 100)]
        
        train, val, test = train_val_test_split(
            dataset, 
            val_ratio=0.15, 
            test_ratio=0.15,
            random_state=42
        )
        
        total = len(train) + len(val) + len(test)
        assert total == len(dataset)
        assert len(val) / total == pytest.approx(0.15, abs=0.05)
        assert len(test) / total == pytest.approx(0.15, abs=0.05)
        
    @pytest.mark.unit
    def test_subject_aware_split(self):
        """Test subject-aware splitting."""
        # Create mock dataset with subject IDs
        dataset = [{"eeg": np.random.randn(105, 500), 
                    "text": "test", 
                    "subject_id": i // 20} 
                   for i in range(100)]
        
        train, val, test = subject_aware_split(
            dataset,
            subject_key="subject_id",
            val_ratio=0.2,
            test_ratio=0.2,
        )
        
        # Extract subject IDs
        train_subjects = set(d["subject_id"] for d in train)
        val_subjects = set(d["subject_id"] for d in val)
        test_subjects = set(d["subject_id"] for d in test)
        
        # Ensure no subject appears in multiple splits
        assert len(train_subjects & val_subjects) == 0
        assert len(train_subjects & test_subjects) == 0
        assert len(val_subjects & test_subjects) == 0
        
    @pytest.mark.unit
    def test_temporal_split(self):
        """Test temporal (chronological) splitting."""
        dataset = [{"eeg": np.random.randn(105, 500), 
                    "text": "test", 
                    "timestamp": i} 
                   for i in range(100)]
        
        train, val, test = temporal_split(
            dataset,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        
        # Ensure temporal order
        assert all(train[i]["timestamp"] < train[i+1]["timestamp"] 
                   for i in range(len(train)-1))
        assert train[-1]["timestamp"] < val[0]["timestamp"]
        assert val[-1]["timestamp"] < test[0]["timestamp"]


class TestPipeline:
    """Tests for complete preprocessing pipeline."""
    
    @pytest.mark.unit
    def test_pipeline_execution(self, sample_eeg_data, preprocessing_config):
        """Test complete preprocessing pipeline."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        eeg, fs = sample_eeg_data
        
        pipeline = PreprocessingPipeline(preprocessing_config)
        processed = pipeline.process(eeg, fs)
        
        assert processed.shape[0] <= eeg.shape[0]  # Channels may be reduced
        assert not np.isnan(processed).any()
        
    @pytest.mark.unit
    def test_pipeline_reproducibility(self, sample_eeg_data, preprocessing_config):
        """Test pipeline reproducibility with same seed."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        
        eeg, fs = sample_eeg_data
        
        pipeline1 = PreprocessingPipeline(preprocessing_config, random_state=42)
        pipeline2 = PreprocessingPipeline(preprocessing_config, random_state=42)
        
        processed1 = pipeline1.process(eeg, fs)
        processed2 = pipeline2.process(eeg, fs)
        
        np.testing.assert_array_almost_equal(processed1, processed2)
