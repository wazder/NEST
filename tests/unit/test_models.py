"""Unit tests for NEST model architectures."""
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


class TestSpatialCNN:
    """Tests for spatial CNN modules."""
    
    @pytest.mark.unit
    def test_spatial_cnn_forward(self, sample_batch_eeg):
        """Test SpatialCNN forward pass."""
        model = SpatialCNN(
            in_channels=105,
            out_channels=[64, 128, 256],
            kernel_size=3,
        )
        
        output = model(sample_batch_eeg)
        
        assert output.shape[0] == sample_batch_eeg.shape[0]
        assert output.shape[1] == 256
        
    @pytest.mark.unit
    def test_eegnet_forward(self, sample_batch_eeg):
        """Test EEGNet forward pass."""
        model = EEGNet(
            n_channels=105,
            n_temporal_filters=8,
            n_spatial_filters=2,
            dropout=0.5,
        )
        
        output = model(sample_batch_eeg)
        
        assert output.shape[0] == sample_batch_eeg.shape[0]
        assert len(output.shape) == 3
        
    @pytest.mark.unit
    def test_deepconvnet_forward(self, sample_batch_eeg):
        """Test DeepConvNet forward pass."""
        model = DeepConvNet(
            n_channels=105,
            n_filters=[25, 50, 100, 200],
            dropout=0.5,
        )
        
        output = model(sample_batch_eeg)
        
        assert output.shape[0] == sample_batch_eeg.shape[0]
        assert len(output.shape) == 3
        
    @pytest.mark.unit
    def test_spatial_cnn_output_shape(self):
        """Test SpatialCNN output shape with different inputs."""
        model = SpatialCNN(
            in_channels=105,
            out_channels=[64, 128],
            kernel_size=3,
        )
        
        # Test different sequence lengths
        for seq_len in [500, 1000, 2000]:
            x = torch.randn(2, 105, seq_len)
            output = model(x)
            assert output.shape[1] == 128


class TestTemporalEncoder:
    """Tests for temporal encoder modules."""
    
    @pytest.mark.unit
    def test_lstm_encoder_forward(self):
        """Test LSTM encoder forward pass."""
        batch_size, seq_len, input_dim = 4, 100, 256
        x = torch.randn(batch_size, seq_len, input_dim)
        
        encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=512,
            num_layers=2,
            bidirectional=True,
        )
        
        output, (hidden, cell) = encoder(x)
        
        assert output.shape == (batch_size, seq_len, 1024)  # 512 * 2
        assert hidden.shape == (4, batch_size, 512)  # 2 layers * 2 directions
        
    @pytest.mark.unit
    def test_gru_encoder_forward(self):
        """Test GRU encoder forward pass."""
        batch_size, seq_len, input_dim = 4, 100, 256
        x = torch.randn(batch_size, seq_len, input_dim)
        
        encoder = GRUEncoder(
            input_dim=input_dim,
            hidden_dim=512,
            num_layers=2,
            bidirectional=True,
        )
        
        output, hidden = encoder(x)
        
        assert output.shape == (batch_size, seq_len, 1024)
        assert hidden.shape == (4, batch_size, 512)
        
    @pytest.mark.unit
    def test_transformer_encoder_forward(self):
        """Test Transformer encoder forward pass."""
        batch_size, seq_len, input_dim = 4, 100, 512
        x = torch.randn(batch_size, seq_len, input_dim)
        
        encoder = TransformerEncoder(
            d_model=input_dim,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
        )
        
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, input_dim)
        
    @pytest.mark.unit
    def test_conformer_encoder_forward(self):
        """Test Conformer encoder forward pass."""
        batch_size, seq_len, input_dim = 4, 100, 512
        x = torch.randn(batch_size, seq_len, input_dim)
        
        encoder = ConformerEncoder(
            d_model=input_dim,
            nhead=8,
            num_layers=6,
        )
        
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, input_dim)


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
        
    @pytest.mark.unit
    def test_multihead_attention(self):
        """Test multi-head attention."""
        batch_size, seq_len, d_model = 4, 100, 512
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=8,
        )
        
        output, weights = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, 8, seq_len, seq_len)


class TestDecoders:
    """Tests for decoder modules."""
    
    @pytest.mark.unit
    def test_ctc_decoder_forward(self):
        """Test CTC decoder forward pass."""
        batch_size, seq_len, hidden_dim = 4, 100, 512
        vocab_size = 1000
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim)
        
        decoder = CTCDecoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        )
        
        logits = decoder(encoder_outputs)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        
    @pytest.mark.unit
    def test_attention_decoder_forward(self):
        """Test attention-based decoder forward pass."""
        batch_size, seq_len, hidden_dim = 4, 100, 512
        vocab_size = 1000
        max_length = 50
        
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim)
        target_tokens = torch.randint(0, vocab_size, (batch_size, max_length))
        
        decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_dim=hidden_dim,
            num_layers=2,
        )
        
        logits, attention_weights = decoder(
            encoder_outputs, target_tokens[:, :-1]
        )
        
        assert logits.shape == (batch_size, max_length - 1, vocab_size)
        
    @pytest.mark.unit
    def test_transducer_decoder_forward(self):
        """Test transducer decoder forward pass."""
        batch_size, target_len = 4, 50
        vocab_size = 1000
        embedding_dim = 256
        
        target_tokens = torch.randint(0, vocab_size, (batch_size, target_len))
        
        decoder = TransducerDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=512,
            num_layers=2,
        )
        
        output, hidden = decoder(target_tokens)
        
        assert output.shape == (batch_size, target_len, 512)


class TestNESTModel:
    """Tests for complete NEST model."""
    
    @pytest.mark.unit
    def test_nest_ctc_forward(self, sample_batch_eeg):
        """Test NEST with CTC decoder."""
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64, 128, 256],
            temporal_hidden=512,
            encoder_type="lstm",
        )
        
        logits = model(sample_batch_eeg)
        
        assert logits.shape[0] == sample_batch_eeg.shape[0]
        assert logits.shape[2] == 1000  # vocab_size
        
    @pytest.mark.unit
    def test_nest_attention_forward(self, sample_batch_eeg):
        """Test NEST with attention decoder."""
        batch_size = sample_batch_eeg.shape[0]
        target_len = 50
        vocab_size = 1000
        
        model = NESTModel(
            n_channels=105,
            vocab_size=vocab_size,
            spatial_channels=[64, 128, 256],
            temporal_hidden=512,
            decoder_type="attention",
        )
        
        target_tokens = torch.randint(0, vocab_size, (batch_size, target_len))
        logits, attention_weights = model(sample_batch_eeg, target_tokens[:, :-1])
        
        assert logits.shape == (batch_size, target_len - 1, vocab_size)
        
    @pytest.mark.unit
    def test_nest_transducer_forward(self, sample_batch_eeg):
        """Test NEST with transducer decoder."""
        batch_size = sample_batch_eeg.shape[0]
        target_len = 50
        vocab_size = 1000
        
        model = NESTWithTransducer(
            n_channels=105,
            vocab_size=vocab_size,
            spatial_channels=[64, 128, 256],
            temporal_hidden=512,
        )
        
        target_tokens = torch.randint(0, vocab_size, (batch_size, target_len))
        logits = model(sample_batch_eeg, target_tokens)
        
        assert logits.shape[0] == batch_size
        assert logits.shape[3] == vocab_size
        
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_nest_gpu_compatibility(self, sample_batch_eeg, device):
        """Test NEST model on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64, 128],
            temporal_hidden=256,
        ).to(device)
        
        x = sample_batch_eeg.to(device)
        logits = model(x)
        
        assert logits.device == device
        
    @pytest.mark.unit
    def test_nest_gradient_flow(self, sample_batch_eeg):
        """Test gradient flow through NEST model."""
        model = NESTWithCTC(
            n_channels=105,
            vocab_size=1000,
            spatial_channels=[64, 128],
            temporal_hidden=256,
        )
        
        logits = model(sample_batch_eeg)
        loss = logits.sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
