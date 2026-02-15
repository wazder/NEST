"""Unit tests for training and evaluation modules."""
import pytest
import torch
import numpy as np
from pathlib import Path

from src.training.trainer import Trainer
from src.training.metrics import (
    calculate_wer,
    calculate_cer,
    calculate_bleu,
    MetricsTracker,
)
from src.training.checkpoint import CheckpointManager
from src.evaluation.beam_search import BeamSearchDecoder
from src.evaluation.benchmark import BenchmarkSuite


class TestMetrics:
    """Tests for evaluation metrics."""
    
    @pytest.mark.unit
    def test_wer_calculation(self):
        """Test Word Error Rate calculation."""
        reference = "the quick brown fox"
        hypothesis = "the quick brown dog"
        
        wer = calculate_wer(reference, hypothesis)
        
        assert 0 <= wer <= 1
        assert wer == 0.25  # 1 substitution out of 4 words
        
    @pytest.mark.unit
    def test_wer_perfect_match(self):
        """Test WER with perfect match."""
        reference = "hello world"
        hypothesis = "hello world"
        
        wer = calculate_wer(reference, hypothesis)
        
        assert wer == 0.0
        
    @pytest.mark.unit
    def test_wer_complete_mismatch(self):
        """Test WER with complete mismatch."""
        reference = "hello world"
        hypothesis = "goodbye universe"
        
        wer = calculate_wer(reference, hypothesis)
        
        assert wer == 1.0
        
    @pytest.mark.unit
    def test_cer_calculation(self):
        """Test Character Error Rate calculation."""
        reference = "hello"
        hypothesis = "helo"
        
        cer = calculate_cer(reference, hypothesis)
        
        assert 0 <= cer <= 1
        assert cer == pytest.approx(0.2, abs=0.01)  # 1 deletion out of 5 chars
        
    @pytest.mark.unit
    def test_bleu_calculation(self):
        """Test BLEU score calculation."""
        references = [["the quick brown fox jumps over the lazy dog"]]
        hypothesis = "the quick brown fox jumps over the dog"
        
        bleu = calculate_bleu(references, hypothesis)
        
        assert 0 <= bleu <= 1
        
    @pytest.mark.unit
    def test_metrics_tracker(self):
        """Test metrics tracking."""
        tracker = MetricsTracker()
        
        # Add some metrics
        tracker.update("loss", 0.5)
        tracker.update("loss", 0.4)
        tracker.update("wer", 0.3)
        
        assert tracker.get_average("loss") == 0.45
        assert tracker.get_average("wer") == 0.3
        assert tracker.get_latest("loss") == 0.4
        
    @pytest.mark.unit
    def test_metrics_tracker_reset(self):
        """Test metrics tracker reset."""
        tracker = MetricsTracker()
        
        tracker.update("loss", 0.5)
        tracker.reset()
        
        assert len(tracker.metrics["loss"]) == 0


class TestCheckpointManager:
    """Tests for checkpoint management."""
    
    @pytest.mark.unit
    def test_checkpoint_save(self, temp_checkpoint_dir):
        """Test checkpoint saving."""
        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=3)
        
        # Create dummy model state
        state = {
            "epoch": 1,
            "model_state_dict": {"weight": torch.randn(10, 10)},
            "optimizer_state_dict": {},
            "loss": 0.5,
        }
        
        checkpoint_path = manager.save_checkpoint(state, epoch=1, metric_value=0.5)
        
        assert checkpoint_path.exists()
        
    @pytest.mark.unit
    def test_checkpoint_load(self, temp_checkpoint_dir):
        """Test checkpoint loading."""
        manager = CheckpointManager(temp_checkpoint_dir)
        
        # Save checkpoint
        state = {
            "epoch": 1,
            "model_state_dict": {"weight": torch.randn(10, 10)},
            "loss": 0.5,
        }
        checkpoint_path = manager.save_checkpoint(state, epoch=1)
        
        # Load checkpoint
        loaded_state = manager.load_checkpoint(checkpoint_path)
        
        assert loaded_state["epoch"] == 1
        assert "model_state_dict" in loaded_state
        
    @pytest.mark.unit
    def test_best_checkpoint_tracking(self, temp_checkpoint_dir):
        """Test tracking of best checkpoint."""
        manager = CheckpointManager(temp_checkpoint_dir, mode="min")
        
        # Save checkpoints with different metrics
        for epoch in range(3):
            state = {"epoch": epoch, "loss": 0.5 - epoch * 0.1}
            manager.save_checkpoint(
                state, epoch=epoch, metric_value=state["loss"]
            )
        
        best_path = manager.get_best_checkpoint()
        assert best_path is not None
        
        loaded = manager.load_checkpoint(best_path)
        assert loaded["epoch"] == 2  # Last epoch has lowest loss
        
    @pytest.mark.unit
    def test_max_checkpoints_limit(self, temp_checkpoint_dir):
        """Test maximum checkpoint limit."""
        manager = CheckpointManager(temp_checkpoint_dir, max_checkpoints=2)
        
        # Save 3 checkpoints
        for epoch in range(3):
            state = {"epoch": epoch}
            manager.save_checkpoint(state, epoch=epoch)
        
        # Should only have 2 checkpoints
        checkpoints = list(temp_checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) <= 3  # 2 regular + 1 best


class TestTrainer:
    """Tests for training functionality."""
    
    @pytest.mark.unit
    def test_trainer_initialization(self, model_config, temp_checkpoint_dir):
        """Test trainer initialization."""
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64, 128],
            temporal_hidden=256,
        )
        
        trainer = Trainer(
            model=model,
            config=model_config,
            checkpoint_dir=temp_checkpoint_dir,
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        
    @pytest.mark.unit
    @pytest.mark.slow
    def test_single_training_step(self, model_config, temp_checkpoint_dir):
        """Test single training step."""
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        trainer = Trainer(
            model=model,
            config=model_config,
            checkpoint_dir=temp_checkpoint_dir,
        )
        
        # Create dummy batch
        batch = {
            "eeg": torch.randn(2, 105, 500),
            "text_ids": torch.randint(0, 1000, (2, 50)),
            "text_lengths": torch.tensor([40, 50]),
        }
        
        loss = trainer.training_step(batch)
        
        assert isinstance(loss, float)
        assert loss > 0
        
    @pytest.mark.unit
    def test_validation_step(self, model_config, temp_checkpoint_dir):
        """Test validation step."""
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        trainer = Trainer(
            model=model,
            config=model_config,
            checkpoint_dir=temp_checkpoint_dir,
        )
        
        # Create dummy batch
        batch = {
            "eeg": torch.randn(2, 105, 500),
            "text_ids": torch.randint(0, 1000, (2, 50)),
            "text_lengths": torch.tensor([40, 50]),
        }
        
        with torch.no_grad():
            val_loss = trainer.validation_step(batch)
        
        assert isinstance(val_loss, float)
        
    @pytest.mark.unit
    def test_learning_rate_scheduling(self, model_config, temp_checkpoint_dir):
        """Test learning rate scheduling."""
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        config = model_config.copy()
        config["training"]["scheduler"] = {
            "type": "step",
            "step_size": 10,
            "gamma": 0.1,
        }
        
        trainer = Trainer(
            model=model,
            config=config,
            checkpoint_dir=temp_checkpoint_dir,
        )
        
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        
        # Simulate training steps
        for _ in range(10):
            if trainer.scheduler:
                trainer.scheduler.step()
        
        new_lr = trainer.optimizer.param_groups[0]["lr"]
        assert new_lr <= initial_lr


class TestBeamSearch:
    """Tests for beam search decoder."""
    
    @pytest.mark.unit
    def test_beam_search_initialization(self, mock_tokenizer):
        """Test beam search decoder initialization."""
        decoder = BeamSearchDecoder(
            tokenizer=mock_tokenizer,
            beam_width=5,
            max_length=100,
        )
        
        assert decoder.beam_width == 5
        assert decoder.max_length == 100
        
    @pytest.mark.unit
    def test_beam_search_decoding(self, mock_tokenizer):
        """Test beam search decoding."""
        decoder = BeamSearchDecoder(
            tokenizer=mock_tokenizer,
            beam_width=5,
            max_length=20,
        )
        
        # Create mock log probabilities
        # Shape: (batch_size, seq_len, vocab_size)
        batch_size = 2
        seq_len = 50
        vocab_size = mock_tokenizer.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Perform beam search
        results = decoder.decode(logits)
        
        assert len(results) == batch_size
        assert all(isinstance(result, str) for result in results)
        
    @pytest.mark.unit
    def test_greedy_decoding(self, mock_tokenizer):
        """Test greedy decoding (beam_width=1)."""
        decoder = BeamSearchDecoder(
            tokenizer=mock_tokenizer,
            beam_width=1,
            max_length=20,
        )
        
        logits = torch.randn(1, 50, mock_tokenizer.vocab_size)
        results = decoder.decode(logits)
        
        assert len(results) == 1


class TestBenchmark:
    """Tests for benchmark suite."""
    
    @pytest.mark.unit
    def test_benchmark_initialization(self):
        """Test benchmark suite initialization."""
        benchmark = BenchmarkSuite(
            metrics=["wer", "cer", "bleu"],
        )
        
        assert "wer" in benchmark.metrics
        assert "cer" in benchmark.metrics
        assert "bleu" in benchmark.metrics
        
    @pytest.mark.unit
    def test_benchmark_evaluation(self):
        """Test benchmark evaluation."""
        benchmark = BenchmarkSuite(
            metrics=["wer", "cer"],
        )
        
        references = ["the quick brown fox", "hello world"]
        hypotheses = ["the quick brown dog", "hello world"]
        
        results = benchmark.evaluate(references, hypotheses)
        
        assert "wer" in results
        assert "cer" in results
        assert 0 <= results["wer"] <= 1
        assert 0 <= results["cer"] <= 1
        
    @pytest.mark.unit
    def test_benchmark_report(self, tmp_path):
        """Test benchmark report generation."""
        benchmark = BenchmarkSuite(
            metrics=["wer"],
        )
        
        references = ["test sentence"]
        hypotheses = ["test sentence"]
        
        results = benchmark.evaluate(references, hypotheses)
        report_path = tmp_path / "report.json"
        
        benchmark.save_report(results, report_path)
        
        assert report_path.exists()


class TestInferenceOptimization:
    """Tests for inference optimization."""
    
    @pytest.mark.unit
    def test_model_quantization(self):
        """Test model quantization."""
        from src.evaluation.quantization import quantize_model
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        model.eval()
        
        quantized = quantize_model(model, dtype=torch.qint8)
        
        assert quantized is not None
        
    @pytest.mark.unit
    def test_model_pruning(self):
        """Test model pruning."""
        from src.evaluation.pruning import prune_model
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        pruned = prune_model(model, amount=0.3)
        
        assert pruned is not None
        
    @pytest.mark.unit
    @pytest.mark.slow
    def test_inference_profiling(self):
        """Test inference profiling."""
        from src.evaluation.profiling import profile_model
        from src.models.nest import NESTWithCTC
        
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64],
            temporal_hidden=128,
        )
        
        input_tensor = torch.randn(1, 105, 500)
        
        stats = profile_model(model, input_tensor)
        
        assert "latency_ms" in stats
        assert "memory_mb" in stats
        assert stats["latency_ms"] > 0
