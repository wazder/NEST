"""
Temporal Encoders for EEG Sequence Processing

This module implements various temporal encoders (LSTM, GRU, Transformer)
to process sequential EEG features and capture temporal dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    """
    LSTM-based temporal encoder for EEG sequences.
    
    Bidirectional LSTM to capture temporal dependencies in both directions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection
        self.output_dim = hidden_dim * self.num_directions
        
        logger.info(
            f"Initialized LSTMEncoder: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, layers={num_layers}, "
            f"bidirectional={bidirectional}"
        )
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            lengths: Sequence lengths for packing (optional)
            
        Returns:
            Tuple of (output, (hidden, cell))
            - output: (batch, seq_len, hidden_dim * num_directions)
            - hidden: (num_layers * num_directions, batch, hidden_dim)
            - cell: (num_layers * num_directions, batch, hidden_dim)
        """
        if lengths is not None:
            # Pack padded sequence
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
        output, (hidden, cell) = self.lstm(x)
        
        if lengths is not None:
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
            
        return output, (hidden, cell)


class GRUEncoder(nn.Module):
    """GRU-based temporal encoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """Initialize GRU encoder."""
        super(GRUEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.output_dim = hidden_dim * self.num_directions
        
        logger.info(
            f"Initialized GRUEncoder: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, layers={num_layers}"
        )
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
        output, hidden = self.gru(x)
        
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
            
        return output, hidden


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer-based temporal encoder.
    
    Uses multi-head self-attention to capture long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        """
        Initialize Transformer encoder.
        
        Args:
            input_dim: Input feature dimension
            d_model: Embedding/model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_dim = d_model
        
        logger.info(
            f"Initialized TransformerEncoder: d_model={d_model}, "
            f"nhead={nhead}, layers={num_layers}"
        )
        
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            src_key_padding_mask: Mask for padded positions (batch, seq_len)
            
        Returns:
            Encoded tensor (batch, seq_len, d_model)
        """
        # Project input to d_model dimension
        x = self.input_proj(x)
        x = x * math.sqrt(self.d_model)  # Scale
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        output = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return output


class ConformerEncoder(nn.Module):
    """
    Conformer encoder combining convolution and self-attention.
    
    Reference: Gulati et al. (2020) "Conformer: Convolution-augmented 
    Transformer for Speech Recognition"
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        """
        Initialize Conformer encoder.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of conformer blocks
            conv_kernel_size: Kernel size for convolution module
            dropout: Dropout rate
        """
        super(ConformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                nhead=nhead,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.output_dim = d_model
        
        logger.info(
            f"Initialized ConformerEncoder: d_model={d_model}, "
            f"nhead={nhead}, layers={num_layers}"
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.input_proj(x)
        
        for block in self.conformer_blocks:
            x = block(x, mask)
            
        return x


class ConformerBlock(nn.Module):
    """Single Conformer block."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        conv_kernel_size: int,
        dropout: float
    ):
        """Initialize Conformer block."""
        super(ConformerBlock, self).__init__()
        
        # Feed-forward module 1
        self.ff1 = FeedForwardModule(d_model, dropout)
        
        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.mha_norm = nn.LayerNorm(d_model)
        
        # Convolution module
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Feed-forward module 2
        self.ff2 = FeedForwardModule(d_model, dropout)
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Conformer block."""
        # Feed-forward 1
        x = x + 0.5 * self.ff1(x)
        
        # Multi-head attention
        residual = x
        x = self.mha_norm(x)
        x, _ = self.mha(x, x, x, key_padding_mask=mask)
        x = residual + x
        
        # Convolution
        x = x + self.conv(x)
        
        # Feed-forward 2
        x = x + 0.5 * self.ff2(x)
        
        # Final norm
        x = self.norm(x)
        
        return x


class FeedForwardModule(nn.Module):
    """Feed-forward module for Conformer."""
    
    def __init__(self, d_model: int, dropout: float):
        """Initialize feed-forward module."""
        super(FeedForwardModule, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        """Initialize convolution module."""
        super(ConvolutionModule, self).__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        x = self.norm(x)
        
        # Transpose for Conv1d: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back
        x = x.transpose(1, 2)
        
        return x


def main():
    """Test temporal encoders."""
    batch_size = 4
    seq_len = 100
    input_dim = 64
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("="*60)
    print("Testing Temporal Encoders")
    print("="*60)
    
    # Test LSTM
    print("\n1. Testing LSTMEncoder...")
    model = LSTMEncoder(input_dim=input_dim, hidden_dim=128)
    output, (hidden, cell) = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Hidden shape: {hidden.shape}")
    
    # Test GRU
    print("\n2. Testing GRUEncoder...")
    model = GRUEncoder(input_dim=input_dim, hidden_dim=128)
    output, hidden = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test Transformer
    print("\n3. Testing TransformerEncoder...")
    model = TransformerEncoder(input_dim=input_dim, d_model=128, nhead=8, num_layers=4)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test Conformer
    print("\n4. Testing ConformerEncoder...")
    model = ConformerEncoder(input_dim=input_dim, d_model=128, nhead=8, num_layers=4)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n" + "="*60)
    print("All temporal encoders tested successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
