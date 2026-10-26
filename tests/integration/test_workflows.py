"""Integration tests for end-to-end workflows."""
import pytest
import torch
import numpy as np
from pathlib import Path


class TestEndToEndPipeline:
    """Tests for complete end-to-end pipeline."""
    
    @pytest.mark.integration
    def test_data_loading_and_augmentation(
        self, sample_eeg_data, preprocessing_config
    ):
        """Test data loading with augmentation."""
        from src.preprocessing.augmentation import EEGAugmentation
        
        eeg, _ = sample_eeg_data
        augmenter = EEGAugmentation()
        
        # Apply multiple augmentations
        augmented = eeg.copy()
        augmented = augmenter.add_gaussian_noise(augmented, noise_level=0.1)
        augmented = augmenter.amplitude_scaling(augmented, scale_range=(0.9, 1.1))
        augmented = augmenter.time_shift(augmented, shift_range=(-20, 20))
        
        assert augmented.shape == eeg.shape
        assert not np.array_equal(augmented, eeg)
        
    @pytest.mark.integration
    @pytest.mark.data
    def test_zuco_dataset_loading(self):
        """Test ZuCo dataset loading (requires dataset)."""
        pytest.skip("Requires ZuCo dataset download")
        
        from src.data.zuco_dataset import ZuCoDataset
        
        dataset = ZuCoDataset(
            data_dir="data/raw/zuco",
            task="task1",
        )
        
        # Load first sample
        sample = dataset[0]
        
        assert "eeg" in sample
        assert "text" in sample
        assert sample["eeg"].shape[0] == 105  # ZuCo has 105 channels


class TestModelInteroperability:
    """Tests for model interoperability and compatibility."""
    
    @pytest.mark.integration
    def test_model_save_and_load(self, tmp_path):
        """Test model saving and loading."""
        from src.models.nest import NEST_CTC
        
        # Create and save model
        model = NEST_CTC(
            n_channels=105,
            vocab_size=1000,
            encoder_hidden_dim=128,
        )
        
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        loaded_model = NEST_CTC(
            n_channels=105,
            vocab_size=1000,
            encoder_hidden_dim=128,
        )
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        x = torch.randn(2, 105, 500)
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            output1 = model(x)
            output2 = loaded_model(x)
        
        assert torch.allclose(output1, output2)
        
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_model_device_transfer(self, device):
        """Test model transfer between CPU and GPU."""
        from src.models.nest import NEST_CTC
        
        model = NEST_CTC(
            n_channels=105,
            vocab_size=1000,
            encoder_hidden_dim=128,
        )
        
        # Transfer to device
        model = model.to(device)
        x = torch.randn(2, 105, 500).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.device == device


class TestRobustnessFeatures:
    """Tests for robustness features."""
    
    @pytest.mark.integration
    def test_noise_injection_training(self):
        """Test training with noise injection."""
        from src.training.robustness import NoiseInjection
        
        injector = NoiseInjection(noise_level=0.1, apply_prob=0.5)
        
        x = torch.randn(4, 105, 500)
        noisy_x = injector(x)
        
        assert noisy_x.shape == x.shape


class TestDeploymentWorkflow:
    """Tests for deployment workflows."""
    
    @pytest.mark.integration
    def test_model_export_onnx(self, tmp_path):
        """Test model export to ONNX."""
        pytest.skip("ONNX export requires careful setup")
        
        from src.evaluation.deployment import export_to_onnx
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        model.eval()
        
        output_path = tmp_path / "model.onnx"
        dummy_input = torch.randn(1, 105, 500)
        
        export_to_onnx(model, dummy_input, output_path)
        
        assert output_path.exists()
        
    @pytest.mark.integration
    def test_torchscript_export(self, tmp_path):
        """Test TorchScript export."""
        from src.models.nest import NEST_CTC
        
        model = NEST_CTC(
            n_channels=105,
            vocab_size=1000,
            encoder_hidden_dim=128,
        )
        
        model.eval()
        
        # Trace model
        dummy_input = torch.randn(1, 105, 500)
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Save
        save_path = tmp_path / "model_traced.pt"
        torch.jit.save(traced_model, save_path)
        
        assert save_path.exists()
        
        # Load and test
        loaded_model = torch.jit.load(save_path)
        
        with torch.no_grad():
            output = loaded_model(dummy_input)
        
        assert output is not None
