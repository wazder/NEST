"""
Attention Mechanisms for EEG-to-Text Decoding

This module implements various attention mechanisms including cross-attention
between EEG embeddings and text tokens for sequence transduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, dropout: float = 0.1):
        """
        Initialize scaled dot-product attention.
        
        Args:
            dropout: Dropout rate
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor (batch, ..., seq_len, d_k)
            key: Key tensor (batch, ..., seq_len, d_k)
            value: Value tensor (batch, ..., seq_len, d_v)
            mask: Attention mask (batch, ..., seq_len, seq_len)
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights


class CrossAttention(nn.Module):
    """
    Cross-attention between EEG embeddings and text tokens.
    
    This allows the decoder to attend to relevant parts of the EEG signal
    when generating text tokens.
    """
    
    def __init__(
        self,
        eeg_dim: int,
        text_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention.
        
        Args:
            eeg_dim: EEG embedding dimension
            text_dim: Text embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossAttention, self).__init__()
        
        self.eeg_dim = eeg_dim
        self.text_dim = text_dim
        
        # Project EEG and text to same dimension if needed
        self.d_model = max(eeg_dim, text_dim)
        
        self.eeg_proj = nn.Linear(eeg_dim, self.d_model) if eeg_dim != self.d_model else nn.Identity()
        self.text_proj = nn.Linear(text_dim, self.d_model) if text_dim != self.d_model else nn.Identity()
        
        # Multi-head attention
        self.mha = MultiHeadAttention(self.d_model, num_heads, dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)
        
        logger.info(
            f"Initialized CrossAttention: eeg_dim={eeg_dim}, "
            f"text_dim={text_dim}, num_heads={num_heads}"
        )
        
    def forward(
        self,
        eeg_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            eeg_features: EEG features (batch, eeg_seq_len, eeg_dim)
            text_embeddings: Text embeddings (batch, text_seq_len, text_dim)
            mask: Attention mask
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Project to same dimension
        eeg_proj = self.eeg_proj(eeg_features)  # (batch, eeg_seq_len, d_model)
        text_proj = self.text_proj(text_embeddings)  # (batch, text_seq_len, d_model)
        
        # Cross-attention: text queries attend to EEG keys/values
        attended, attn_weights = self.mha(
            query=text_proj,
            key=eeg_proj,
            value=eeg_proj,
            mask=mask
        )
        
        # Residual connection and normalization
        output = self.norm(text_embeddings + attended)
        
        return output, attn_weights


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.
    
    Reference: Bahdanau et al. (2015) "Neural Machine Translation by 
    Jointly Learning to Align and Translate"
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int
    ):
        """
        Initialize additive attention.
        
        Args:
            query_dim: Query dimension
            key_dim: Key dimension
            hidden_dim: Hidden dimension for attention computation
        """
        super(AdditiveAttention, self).__init__()
        
        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(key_dim, hidden_dim, bias=False)
        self.energy_proj = nn.Linear(hidden_dim, 1, bias=False)
        
        logger.info(
            f"Initialized AdditiveAttention: query_dim={query_dim}, "
            f"key_dim={key_dim}, hidden_dim={hidden_dim}"
        )
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor (batch, query_len, query_dim)
            key: Key tensor (batch, key_len, key_dim)
            value: Value tensor (batch, key_len, value_dim)
            mask: Attention mask (batch, query_len, key_len)
            
        Returns:
            Tuple of (context, attention_weights)
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        # Project query and key
        # (batch, query_len, hidden_dim)
        query_proj = self.query_proj(query)
        # (batch, key_len, hidden_dim)
        key_proj = self.key_proj(key)
        
        # Expand dimensions for broadcasting
        # (batch, query_len, 1, hidden_dim)
        query_proj = query_proj.unsqueeze(2)
        # (batch, 1, key_len, hidden_dim)
        key_proj = key_proj.unsqueeze(1)
        
        # Compute energy
        # (batch, query_len, key_len, hidden_dim)
        energy = torch.tanh(query_proj + key_proj)
        # (batch, query_len, key_len)
        energy = self.energy_proj(energy).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
            
        # Compute attention weights
        attn_weights = F.softmax(energy, dim=-1)
        
        # Compute context vector
        # (batch, query_len, value_dim)
        context = torch.bmm(attn_weights, value)
        
        return context, attn_weights


class LocationAwareAttention(nn.Module):
    """
    Location-aware attention mechanism.
    
    Takes into account previous attention weights to avoid repeating
    or skipping parts of the input sequence.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_filters: int = 32,
        kernel_size: int = 31
    ):
        """
        Initialize location-aware attention.
        
        Args:
            query_dim: Query dimension
            key_dim: Key dimension
            hidden_dim: Hidden dimension
            num_filters: Number of convolution filters
            kernel_size: Convolution kernel size
        """
        super(LocationAwareAttention, self).__init__()
        
        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(key_dim, hidden_dim, bias=False)
        
        # Location-aware convolution
        self.location_conv = nn.Conv1d(
            1, num_filters, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.location_proj = nn.Linear(num_filters, hidden_dim, bias=False)
        
        self.energy_proj = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        previous_attn: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor (batch, query_len, query_dim)
            key: Key tensor (batch, key_len, key_dim)
            value: Value tensor (batch, key_len, value_dim)
            previous_attn: Previous attention weights (batch, query_len, key_len)
            mask: Attention mask
            
        Returns:
            Tuple of (context, attention_weights)
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        
        # Project query and key
        query_proj = self.query_proj(query).unsqueeze(2)
        key_proj = self.key_proj(key).unsqueeze(1)
        
        # Process previous attention with convolution
        if previous_attn is not None:
            # (batch * query_len, 1, key_len)
            previous_attn_flat = previous_attn.view(-1, 1, previous_attn.size(-1))
            # (batch * query_len, num_filters, key_len)
            location_features = self.location_conv(previous_attn_flat)
            # (batch, query_len, key_len, num_filters)
            location_features = location_features.view(
                batch_size, query_len, -1, location_features.size(1)
            )
            # (batch, query_len, key_len, hidden_dim)
            location_features = self.location_proj(location_features.transpose(2, 3))
        else:
            location_features = 0
            
        # Compute energy
        energy = torch.tanh(query_proj + key_proj + location_features)
        energy = self.energy_proj(energy).squeeze(-1)
        
        # Apply mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
            
        # Compute attention weights
        attn_weights = F.softmax(energy, dim=-1)
        
        # Compute context
        context = torch.bmm(attn_weights, value)
        
        return context, attn_weights


def main():
    """Test attention mechanisms."""
    batch_size = 4
    eeg_seq_len = 100
    text_seq_len = 20
    eeg_dim = 128
    text_dim = 256
    
    eeg_features = torch.randn(batch_size, eeg_seq_len, eeg_dim)
    text_embeddings = torch.randn(batch_size, text_seq_len, text_dim)
    
    print("="*60)
    print("Testing Attention Mechanisms")
    print("="*60)
    
    # Test Cross-Attention
    print("\n1. Testing CrossAttention...")
    model = CrossAttention(eeg_dim=eeg_dim, text_dim=text_dim, num_heads=8)
    output, attn_weights = model(eeg_features, text_embeddings)
    print(f"   EEG features shape: {eeg_features.shape}")
    print(f"   Text embeddings shape: {text_embeddings.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Test Additive Attention
    print("\n2. Testing AdditiveAttention...")
    model = AdditiveAttention(query_dim=text_dim, key_dim=eeg_dim, hidden_dim=256)
    context, attn_weights = model(text_embeddings, eeg_features, eeg_features)
    print(f"   Context shape: {context.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Test Location-Aware Attention
    print("\n3. Testing LocationAwareAttention...")
    model = LocationAwareAttention(query_dim=text_dim, key_dim=eeg_dim, hidden_dim=256)
    context, attn_weights = model(text_embeddings, eeg_features, eeg_features)
    print(f"   Context shape: {context.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    print("\n" + "="*60)
    print("All attention mechanisms tested successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
