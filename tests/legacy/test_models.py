"""Archived v1 tests from tests/unit/test_models.py.

Not collected: they call APIs that src/ does not provide.
See README.md in this folder.
"""
import pytest
import torch
import torch.nn as nn

from src.models.spatial_cnn import SpatialCNN, EEGNet, DeepConvNet
from src.models.temporal_encoder import (
    LSTMEncoder,
    GRUEncoder,
    TransformerEncoder,
    ConformerEncoder,
)
from src.models.attention import (
    AdditiveAttention,
    MultiplicativeAttention,
    MultiHeadAttention,
)
from src.models.decoder import (
    CTCDecoder,
    AttentionDecoder,
    TransducerDecoder,
)
from src.models.nest import (
    NESTModel,
    NESTWithCTC,
    NESTWithTransducer,
)


class TestAttentionMechanisms:
    """Tests for attention mechanisms."""
    
    @pytest.mark.unit
    def test_additive_attention(self):
        """Test additive (Bahdanau) attention."""
        batch_size, seq_len, hidden_dim = 4, 100, 512
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim)
        decoder_hidden = torch.randn(batch_size, hidden_dim)
        
        attention = AdditiveAttention(hidden_dim=hidden_dim)
        
        context, weights = attention(decoder_hidden, encoder_outputs)
        
        assert context.shape == (batch_size, hidden_dim)
        assert weights.shape == (batch_size, seq_len)
        assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size))
        
    @pytest.mark.unit
    def test_multiplicative_attention(self):
        """Test multiplicative (Luong) attention."""
        batch_size, seq_len, hidden_dim = 4, 100, 512
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim)
        decoder_hidden = torch.randn(batch_size, hidden_dim)
        
        attention = MultiplicativeAttention(hidden_dim=hidden_dim)
        
        context, weights = attention(decoder_hidden, encoder_outputs)
        
        assert context.shape == (batch_size, hidden_dim)
        assert weights.shape == (batch_size, seq_len)
