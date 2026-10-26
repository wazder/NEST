"""Archived v1 tests from tests/integration/test_workflows.py.

Not collected: they call APIs that src/ does not provide.
See README.md in this folder.
"""
import pytest
import torch
import numpy as np
from pathlib import Path


class TestEndToEndPipeline:
    """Tests for complete end-to-end pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_preprocessing_to_model(
        self, sample_eeg_data, preprocessing_config, model_config
    ):
        """Test complete pipeline from raw EEG to model output."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        from src.models.nest import NESTWithCTC
        
        eeg, fs = sample_eeg_data
        
        # Preprocessing
        pipeline = PreprocessingPipeline(preprocessing_config)
        processed = pipeline.process(eeg, fs)
        
        # Convert to tensor and add batch dimension
        eeg_tensor = torch.from_numpy(processed).float().unsqueeze(0)
        
        # Model forward pass
        model = NESTWithCTC(
            n_channels=processed.shape[0],
            vocab_size=1000,
            spatial_channels=[64, 128],
            temporal_hidden=256,
        )
        
        model.eval()
        with torch.no_grad():
            output = model(eeg_tensor)
        
        assert output.shape[0] == 1
        assert output.shape[2] == 1000
        
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_and_evaluation_workflow(
        self, temp_checkpoint_dir, model_config
    ):
        """Test training and evaluation workflow."""
        from src.models.nest import NESTWithCTC
        from src.training.trainer import Trainer
        from src.evaluation.benchmark import BenchmarkSuite
        
        # Create model
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=model_config,
            checkpoint_dir=temp_checkpoint_dir,
        )
        
        # Create dummy training data
        train_batch = {
            "eeg": torch.randn(4, 105, 500),
            "text_ids": torch.randint(0, 1000, (4, 50)),
            "text_lengths": torch.tensor([45, 50, 48, 50]),
        }
        
        # Training step
        train_loss = trainer.training_step(train_batch)
        assert train_loss > 0
        
        # Validation step
        with torch.no_grad():
            val_loss = trainer.validation_step(train_batch)
        assert val_loss > 0
        
        # Save checkpoint
        state = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "loss": train_loss,
        }
        checkpoint_path = trainer.checkpoint_manager.save_checkpoint(
            state, epoch=0, metric_value=val_loss
        )
        
        assert checkpoint_path.exists()
        
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
    def test_different_encoder_decoder_combinations(self):
        """Test different encoder-decoder combinations."""
        from src.models.factory import ModelFactory
        
        configs = [
            {"encoder_type": "lstm", "decoder_type": "ctc"},
            {"encoder_type": "gru", "decoder_type": "ctc"},
            {"encoder_type": "transformer", "decoder_type": "attention"},
        ]
        
        for config in configs:
            model = ModelFactory.create_model(
                model_type="nest",
                n_channels=105,
                vocab_size=1000,
                **config,
            )
            
            assert model is not None
            
            # Test forward pass
            x = torch.randn(2, 105, 500)
            with torch.no_grad():
                output = model(x)
            
            assert output is not None


class TestSubjectAdaptation:
    """Tests for subject adaptation techniques."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_domain_adaptation_dann(self):
        """Test Domain-Adversarial Neural Network adaptation."""
        from src.models.adaptation import DANNAdapter
        from src.models.nest import NESTWithCTC
        
        # Create base model
        base_model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        # Create DANN adapter
        adapter = DANNAdapter(
            base_model=base_model,
            hidden_dim=128,
            domain_loss_weight=0.1,
        )
        
        # Create dummy data
        source_eeg = torch.randn(4, 105, 500)
        target_eeg = torch.randn(4, 105, 500)
        
        # Forward pass
        output, domain_loss = adapter(source_eeg, target_eeg)
        
        assert output.shape[0] == 4
        assert domain_loss.item() >= 0
        
    @pytest.mark.integration
    def test_coral_adaptation(self):
        """Test CORAL domain adaptation."""
        from src.models.adaptation import CORALAdapter
        from src.models.nest import NESTWithCTC
        
        base_model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        adapter = CORALAdapter(
            base_model=base_model,
            coral_weight=0.5,
        )
        
        source_eeg = torch.randn(4, 105, 500)
        target_eeg = torch.randn(4, 105, 500)
        
        output, coral_loss = adapter(source_eeg, target_eeg)
        
        assert coral_loss.item() >= 0


class TestRobustnessFeatures:
    """Tests for robustness features."""
    
    @pytest.mark.integration
    def test_adversarial_training(self):
        """Test adversarial training robustness."""
        from src.training.robustness import generate_adversarial_examples
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        x = torch.randn(2, 105, 500)
        targets = torch.randint(0, 1000, (2, 50))
        
        # Generate adversarial examples
        adv_x = generate_adversarial_examples(
            model, x, targets, epsilon=0.01
        )
        
        assert adv_x.shape == x.shape
        assert not torch.equal(adv_x, x)


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
    @pytest.mark.slow
    def test_realtime_inference_latency(self):
        """Test real-time inference latency requirements."""
        from src.evaluation.realtime_inference import RealtimeInference
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[32],  # Smaller for faster inference
            temporal_hidden=128,
        )
        
        inference = RealtimeInference(model, max_latency_ms=100)
        
        # Test single inference
        x = torch.randn(1, 105, 500)
        
        output, latency_ms = inference.infer(x)
        
        assert output is not None
        assert latency_ms < inference.max_latency_ms * 2  # Allow some slack
