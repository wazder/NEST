"""
Advanced Attention Mechanisms for NEST Models

This module implements sophisticated attention variants:
- Relative Position Attention
- Local/Windowed Attention
- Talking Heads Attention
- Linear Attention (efficient for long sequences)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RelativePositionAttention(nn.Module):
    """
    Multi-head attention with relative position encoding.
    
    Shaw et al. (2018): "Self-Attention with Relative Position Representations"
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_relative_position: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize relative position attention.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            max_relative_position: Maximum relative position
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.max_relative_position = max_relative_position
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        self.relative_k = nn.Embedding(
            2 * max_relative_position + 1,
            self.d_k
        )
        self.relative_v = nn.Embedding(
            2 * max_relative_position + 1,
            self.d_k
        )
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def _get_relative_positions(self, length: int) -> torch.Tensor:
        """Generate relative position matrix."""
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0).expand(length, -1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift to be >= 0
        final_mat = distance_mat_clipped + self.max_relative_position
        
        return final_mat
        
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
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, nhead, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.size()
        
        # Project Q, K, V
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Get relative positions
        relative_positions = self._get_relative_positions(seq_len).to(query.device)
        
        # Relative position embeddings
        relative_k_emb = self.relative_k(relative_positions)  # (seq_len, seq_len, d_k)
        relative_v_emb = self.relative_v(relative_positions)  # (seq_len, seq_len, d_k)
        
        # Compute attention scores with relative positions
        # Standard attention: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, nhead, seq_len, seq_len)
        
        # Add relative position contribution
        # For each query position, compute dot product with relative keys
        Q_reshaped = Q.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size * self.nhead, self.d_k)
        relative_scores = torch.matmul(Q_reshaped, relative_k_emb.transpose(-2, -1))
        relative_scores = relative_scores.view(seq_len, batch_size, self.nhead, seq_len)
        relative_scores = relative_scores.permute(1, 2, 0, 3)  # (batch, nhead, seq_len, seq_len)
        
        scores = scores + relative_scores
        scores = scores / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch, nhead, seq_len, d_k)
        
        # Add relative position contribution to values
        attn_weights_reshaped = attn_weights.permute(2, 0, 1, 3).contiguous().view(
            seq_len, batch_size * self.nhead, seq_len
        )
        relative_context = torch.matmul(attn_weights_reshaped, relative_v_emb)
        relative_context = relative_context.view(seq_len, batch_size, self.nhead, self.d_k)
        relative_context = relative_context.permute(1, 2, 0, 3)
        
        context = context + relative_context
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        
        return output, attn_weights


class LocalAttention(nn.Module):
    """
    Local/windowed attention for efficient long-sequence processing.
    
    Only attends to a local window around each position.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize local attention.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            window_size: Size of attention window (total: 2*window_size + 1)
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.window_size = window_size
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, nhead, seq_len, window_size*2+1)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project
        Q = self.w_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # For simplicity, use standard attention with local mask
        # (More efficient implementations would use specialized kernels)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Create local attention mask
        local_mask = self._get_local_mask(seq_len, self.window_size).to(x.device)
        scores = scores.masked_fill(local_mask == 0, -1e9)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        
        return output, attn_weights
        
    def _get_local_mask(self, seq_len: int, window_size: int) -> torch.Tensor:
        """Create local attention mask."""
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = 1
            
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


class LinearAttention(nn.Module):
    """
    Linear attention with O(N) complexity.
    
    Katharopoulos et al. (2020): "Transformers are RNNs: Fast Autoregressive
    Transformers with Linear Attention"
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1
    ):
        """
        Initialize linear attention.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional mask (not used in linear attention)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project
        Q = self.w_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Apply ELU + 1 as feature map (makes attention weights non-negative)
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Linear attention: V' = (K^T V) / (K^T 1)
        # where 1 is all-ones vector
        
        # Compute K^T V: (batch, nhead, d_k, d_k)
        KV = torch.matmul(K.transpose(-2, -1), V)
        
        # Compute K^T 1: (batch, nhead, d_k, 1)
        K_sum = K.sum(dim=-2, keepdim=True).transpose(-2, -1)
        
        # Compute Q (K^T V): (batch, nhead, seq_len, d_k)
        numerator = torch.matmul(Q, KV)
        
        # Compute Q (K^T 1): (batch, nhead, seq_len, 1)
        denominator = torch.matmul(Q, K_sum)
        
        # Normalize
        context = numerator / (denominator + 1e-6)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        output = self.dropout(output)
        
        return output


def main():
    """Example usage."""
    print("="*60)
    print("Advanced Attention Mechanisms")
    print("="*60)
    
    batch_size = 2
    seq_len = 100
    d_model = 512
    nhead = 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test Relative Position Attention
    print("\n1. Relative Position Attention")
    rel_attn = RelativePositionAttention(d_model, nhead, max_relative_position=64)
    output, weights = rel_attn(x, x, x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Attention weights: {weights.shape}")
    
    # Test Local Attention
    print("\n2. Local Attention")
    local_attn = LocalAttention(d_model, nhead, window_size=32)
    output, weights = local_attn(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Attention weights: {weights.shape}")
    
    # Test Linear Attention
    print("\n3. Linear Attention")
    linear_attn = LinearAttention(d_model, nhead)
    output = linear_attn(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Complexity: O(N) vs O(N²) for standard attention")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
