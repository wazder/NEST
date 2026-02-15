"""Unit tests for preprocessing modules."""
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


class TestFiltering:
    """Tests for signal filtering functions."""
    
    @pytest.mark.unit
    def test_bandpass_filter(self, sample_eeg_data):
        """Test bandpass filtering."""
        eeg, fs = sample_eeg_data
        
        filtered = bandpass_filter(eeg, low=0.5, high=50.0, fs=fs)
        
        assert filtered.shape == eeg.shape
        assert not np.isnan(filtered).any()
        assert not np.isinf(filtered).any()
        
    @pytest.mark.unit
    def test_notch_filter(self, sample_eeg_data):
        """Test notch filtering for powerline noise."""
        eeg, fs = sample_eeg_data
        
        filtered = notch_filter(eeg, freq=50.0, fs=fs, quality=30)
        
        assert filtered.shape == eeg.shape
        assert not np.isnan(filtered).any()
        
    @pytest.mark.unit
    def test_frequency_band_extraction(self, sample_eeg_data):
        """Test extraction of frequency bands."""
        eeg, fs = sample_eeg_data
        
        bands = extract_frequency_bands(eeg, fs=fs)
        
        assert "delta" in bands
        assert "theta" in bands
        assert "alpha" in bands
        assert "beta" in bands
        assert "gamma" in bands
        
        for band_name, band_data in bands.items():
            assert band_data.shape == eeg.shape
            
    @pytest.mark.unit
    def test_common_average_reference(self, sample_eeg_data):
        """Test Common Average Reference (CAR)."""
        eeg, _ = sample_eeg_data
        
        car_eeg = apply_car(eeg)
        
        assert car_eeg.shape == eeg.shape
        # CAR should make mean across channels close to zero
        assert np.abs(car_eeg.mean(axis=0)).max() < 1e-10
        
    @pytest.mark.unit
    def test_laplacian_filter(self, sample_eeg_data):
        """Test Laplacian spatial filtering."""
        eeg, _ = sample_eeg_data
        
        # Create mock neighbor indices
        neighbors = {i: [max(0, i-1), min(eeg.shape[0]-1, i+1)] 
                     for i in range(eeg.shape[0])}
        
        filtered = laplacian_filter(eeg, neighbors)
        
        assert filtered.shape == eeg.shape


class TestArtifactRemoval:
    """Tests for artifact removal methods."""
    
    @pytest.mark.unit
    def test_ica_removal_fit(self, sample_eeg_data):
        """Test ICA artifact removal fitting."""
        eeg, _ = sample_eeg_data
        
        remover = ICARemover(n_components=10, method="fastica")
        remover.fit(eeg)
        
        assert remover.ica_ is not None
        assert remover.mixing_.shape == (eeg.shape[0], 10)
        
    @pytest.mark.unit
    def test_ica_removal_transform(self, sample_eeg_data):
        """Test ICA artifact removal transformation."""
        eeg, _ = sample_eeg_data
        
        remover = ICARemover(n_components=10, method="fastica")
        remover.fit(eeg)
        
        # Mark first component as artifact
        cleaned = remover.transform(eeg, exclude=[0])
        
        assert cleaned.shape == eeg.shape
        assert not np.array_equal(cleaned, eeg)
        
    @pytest.mark.unit
    def test_threshold_rejection(self, sample_eeg_data):
        """Test amplitude threshold-based rejection."""
        eeg, _ = sample_eeg_data
        
        # Add some extreme values
        eeg_with_artifacts = eeg.copy()
        eeg_with_artifacts[0, 100] = 1000  # Spike
        
        rejecter = ThresholdRejecter(amplitude_threshold=100)
        mask = rejecter.detect_bad_samples(eeg_with_artifacts)
        
        assert mask.sum() > 0  # Should detect the spike
        
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
    def test_selection_by_names(self):
        """Test electrode selection by channel names."""
        all_channels = [f"CH{i}" for i in range(105)]
        selected_names = ["CH0", "CH1", "CH2"]
        
        indices = select_by_names(all_channels, selected_names)
        
        assert len(indices) == 3
        assert indices == [0, 1, 2]
        
    @pytest.mark.unit
    def test_selection_by_variance(self, sample_eeg_data):
        """Test electrode selection by variance."""
        eeg, _ = sample_eeg_data
        
        selector = ElectrodeSelector(method="variance", n_channels=50)
        selected_indices = selector.fit_select(eeg)
        
        assert len(selected_indices) == 50
        assert max(selected_indices) < eeg.shape[0]
        
    @pytest.mark.unit
    def test_selection_by_mutual_information(self, sample_eeg_data, sample_text_data):
        """Test electrode selection by mutual information."""
        eeg, _ = sample_eeg_data
        
        # Create mock labels
        labels = np.random.randint(0, 10, size=eeg.shape[1])
        
        selector = ElectrodeSelector(method="mutual_info", n_channels=50)
        selected_indices = selector.fit_select(eeg, labels)
        
        assert len(selected_indices) == 50
        
    @pytest.mark.unit
    def test_pca_reduction(self, sample_eeg_data):
        """Test PCA-based dimensionality reduction."""
        eeg, _ = sample_eeg_data
        
        selector = ElectrodeSelector(method="pca", n_channels=50)
        reduced = selector.fit_transform(eeg)
        
        assert reduced.shape[0] == 50
        assert reduced.shape[1] == eeg.shape[1]


class TestAugmentation:
    """Tests for data augmentation techniques."""
    
    @pytest.mark.unit
    def test_gaussian_noise_addition(self, sample_eeg_data):
        """Test Gaussian noise augmentation."""
        eeg, _ = sample_eeg_data
        
        augmented = add_gaussian_noise(eeg, noise_std=0.1)
        
        assert augmented.shape == eeg.shape
        assert not np.array_equal(augmented, eeg)
        
    @pytest.mark.unit
    def test_amplitude_scaling(self, sample_eeg_data):
        """Test amplitude scaling augmentation."""
        eeg, _ = sample_eeg_data
        
        augmented = scale_amplitude(eeg, scale_range=(0.8, 1.2))
        
        assert augmented.shape == eeg.shape
        # Check that scaling was applied
        assert not np.allclose(augmented, eeg)
        
    @pytest.mark.unit
    def test_time_shift(self, sample_eeg_data):
        """Test time shifting augmentation."""
        eeg, _ = sample_eeg_data
        
        augmented = time_shift(eeg, max_shift=50)
        
        assert augmented.shape == eeg.shape
        
    @pytest.mark.unit
    def test_time_warp(self, sample_eeg_data):
        """Test time warping augmentation."""
        eeg, _ = sample_eeg_data
        
        augmented = time_warp(eeg, warp_factor=1.1)
        
        # Shape might change slightly due to warping
        assert augmented.shape[0] == eeg.shape[0]
        
    @pytest.mark.unit
    def test_channel_dropout(self, sample_eeg_data):
        """Test channel dropout augmentation."""
        eeg, _ = sample_eeg_data
        
        augmented = channel_dropout(eeg, dropout_rate=0.1)
        
        assert augmented.shape == eeg.shape
        # Some channels should be zeroed out
        assert (augmented.sum(axis=1) == 0).any()


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
