"""Unit tests for training and evaluation modules."""
import pytest
import torch
import numpy as np
from pathlib import Path

from src.training.metrics import (
    word_error_rate,
    character_error_rate,
    compute_bleu,
)


class TestMetrics:
    """Tests for evaluation metrics."""
    
    @pytest.mark.unit
    def test_wer_calculation(self):
        """Test Word Error Rate calculation."""
        reference = "the quick brown fox"
        hypothesis = "the quick brown dog"
        
        wer = word_error_rate([reference], [hypothesis])
        
        assert 0 <= wer <= 1
        assert wer == 0.25  # 1 substitution out of 4 words
        
    @pytest.mark.unit
    def test_wer_perfect_match(self):
        """Test WER with perfect match."""
        reference = "hello world"
        hypothesis = "hello world"
        
        wer = word_error_rate([reference], [hypothesis])
        
        assert wer == 0.0
        
    @pytest.mark.unit
    def test_wer_complete_mismatch(self):
        """Test WER with complete mismatch."""
        reference = "hello world"
        hypothesis = "goodbye universe"
        
        wer = word_error_rate([reference], [hypothesis])
        
        assert wer == 1.0
        
    @pytest.mark.unit
    def test_cer_calculation(self):
        """Test Character Error Rate calculation."""
        reference = "hello"
        hypothesis = "helo"
        
        cer = character_error_rate([reference], [hypothesis])
        
        assert 0 <= cer <= 1
        assert cer == pytest.approx(0.2, abs=0.01)  # 1 deletion out of 5 chars
        
    @pytest.mark.unit
    def test_bleu_calculation(self):
        """Test BLEU score calculation."""
        references = [["the quick brown fox jumps over the lazy dog"]]
        hypothesis = "the quick brown fox jumps over the dog"
        
        bleu = compute_bleu(references, [hypothesis])
        
        assert 0 <= bleu <= 1
        


class TestInferenceOptimization:
    """Tests for inference optimization."""
    
    @pytest.mark.unit
    def test_model_quantization(self):
        """Test model quantization."""
        from src.evaluation.quantization import DynamicQuantizer
        from src.models.nest import NEST_CTC
        
        model = NEST_CTC(
            n_channels=105,
            vocab_size=1000,
            encoder_hidden_dim=128,
        )
        
        model.eval()

        # macOS arm64 ships qnnpack but selects no quantized engine by default.
        # torch cannot switch back to "none", so the engine stays set.
        if torch.backends.quantized.engine == "none":
            if "qnnpack" not in torch.backends.quantized.supported_engines:
                pytest.skip("No quantized engine available")
            torch.backends.quantized.engine = "qnnpack"

        quantized = DynamicQuantizer(model, dtype=torch.qint8).quantize()
        
        assert quantized is not None
        
    @pytest.mark.unit
    def test_model_pruning(self):
        """Test model pruning."""
        from src.evaluation.pruning import MagnitudePruner
        from src.models.nest import NEST_CTC
        
        model = NEST_CTC(
            n_channels=105,
            vocab_size=1000,
            encoder_hidden_dim=128,
        )
        
        pruned = MagnitudePruner(model, amount=0.3).prune()
        
        assert pruned is not None
