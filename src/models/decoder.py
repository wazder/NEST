"""
Decoder Modules and CTC Loss for Sequence Transduction

This module implements various decoder architectures and CTC (Connectionist
Temporal Classification) loss for EEG-to-text sequence transduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CTCDecoder(nn.Module):
    """
    CTC (Connectionist Temporal Classification) decoder.
    
    Maps encoder output to character/token probabilities at each timestep.
    CTC allows alignment-free training for sequence transduction.
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        blank_id: int = 0
    ):
        """
        Initialize CTC decoder.
        
        Args:
            input_dim: Input feature dimension from encoder
            vocab_size: Vocabulary size (including blank token)
            blank_id: ID of blank token (default: 0)
        """
        super(CTCDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        
        # Linear projection to vocabulary
        self.output_proj = nn.Linear(input_dim, vocab_size)
        
        logger.info(
            f"Initialized CTCDecoder: input_dim={input_dim}, "
            f"vocab_size={vocab_size}, blank_id={blank_id}"
        )
        
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            encoder_output: Encoder output (batch, seq_len, input_dim)
            
        Returns:
            Log probabilities (batch, seq_len, vocab_size)
        """
        # Project to vocabulary
        logits = self.output_proj(encoder_output)
        
        # Apply log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
        
    def decode_greedy(
        self,
        log_probs: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Greedy decoding (pick most likely token at each step).
        
        Args:
            log_probs: Log probabilities (batch, seq_len, vocab_size)
            input_lengths: Length of each sequence in batch
            
        Returns:
            List of decoded sequences (one per batch item)
        """
        batch_size = log_probs.size(0)
        
        # Get most likely tokens
        pred_tokens = log_probs.argmax(dim=-1)  # (batch, seq_len)
        
        decoded = []
        
        for b in range(batch_size):
            seq = pred_tokens[b]
            
            if input_lengths is not None:
                seq = seq[:input_lengths[b]]
                
            # Remove consecutive duplicates and blanks
            decoded_seq = []
            prev_token = None
            
            for token in seq.tolist():
                if token != self.blank_id and token != prev_token:
                    decoded_seq.append(token)
                prev_token = token
                
            decoded.append(decoded_seq)
            
        return decoded


class AttentionDecoder(nn.Module):
    """
    Attention-based decoder for sequence-to-sequence transduction.
    
    Uses cross-attention to attend to encoder outputs while generating
    output sequence.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_type: str = 'additive'
    ):
        """
        Initialize attention decoder.
        
        Args:
            vocab_size: Output vocabulary size
            embedding_dim: Token embedding dimension
            encoder_dim: Encoder output dimension
            hidden_dim: Decoder hidden dimension
            num_layers: Number of decoder layers
            dropout: Dropout rate
            attention_type: Type of attention ('additive', 'multiplicative')
        """
        super(AttentionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Decoder RNN
        self.rnn = nn.LSTM(
            input_size=embedding_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        if attention_type == 'additive':
            from .attention import AdditiveAttention
            self.attention = AdditiveAttention(
                query_dim=hidden_dim,
                key_dim=encoder_dim,
                hidden_dim=hidden_dim
            )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
            
        # Output projection
        self.output_proj = nn.Linear(
            hidden_dim + encoder_dim,
            vocab_size
        )
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(
            f"Initialized AttentionDecoder: vocab_size={vocab_size}, "
            f"hidden_dim={hidden_dim}, attention_type={attention_type}"
        )
        
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_tokens: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            encoder_output: Encoder output (batch, enc_seq_len, encoder_dim)
            target_tokens: Target token IDs (batch, dec_seq_len)
            hidden: Initial hidden state (optional)
            
        Returns:
            Tuple of (output_logits, hidden_state)
        """
        batch_size = encoder_output.size(0)
        dec_seq_len = target_tokens.size(1)
        
        # Embed target tokens
        embedded = self.embedding(target_tokens)  # (batch, dec_seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        outputs = []
        
        for t in range(dec_seq_len):
            # Get current embedded token
            current_embedded = embedded[:, t:t+1, :]  # (batch, 1, embedding_dim)
            
            # Compute attention context
            if hasattr(self, 'encoder_proj'):
                # For multi-head attention
                query = hidden[0][-1:].transpose(0, 1) if hidden is not None else torch.zeros(
                    batch_size, 1, self.hidden_dim, device=encoder_output.device
                )
                context, _ = self.attention(
                    query, 
                    self.encoder_proj(encoder_output), 
                    encoder_output
                )
            else:
                # For additive attention
                query = hidden[0][-1] if hidden is not None else torch.zeros(
                    batch_size, self.hidden_dim, device=encoder_output.device
                )
                context, _ = self.attention(
                    query.unsqueeze(1),
                    encoder_output,
                    encoder_output
                )
                
            # Concatenate embedded token and context
            rnn_input = torch.cat([current_embedded, context], dim=-1)
            
            # Pass through RNN
            rnn_output, hidden = self.rnn(rnn_input, hidden)
            
            # Concatenate RNN output and context for final projection
            output = torch.cat([rnn_output, context], dim=-1)
            output = self.output_proj(output)
            
            outputs.append(output)
            
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)  # (batch, dec_seq_len, vocab_size)
        
        return outputs, hidden


class TransducerDecoder(nn.Module):
    """
    Transducer decoder (prediction network) for RNN-T/Transformer-T.
    
    Processes previously predicted tokens to generate prediction embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize transducer decoder.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Token embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super(TransducerDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        logger.info(
            f"Initialized TransducerDecoder: vocab_size={vocab_size}, "
            f"hidden_dim={hidden_dim}"
        )
        
    def forward(
        self,
        tokens: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            tokens: Token IDs (batch, seq_len)
            hidden: Hidden state (optional)
            
        Returns:
            Tuple of (prediction_embeddings, hidden_state)
        """
        # Embed tokens
        embedded = self.embedding(tokens)  # (batch, seq_len, embedding_dim)
        
        # Pass through LSTM
        output, hidden = self.lstm(embedded, hidden)
        
        return output, hidden


class JointNetwork(nn.Module):
    """
    Joint network for RNN-Transducer.
    
    Combines encoder output and prediction network output to produce
    final token probabilities.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joint_dim: int,
        vocab_size: int
    ):
        """
        Initialize joint network.
        
        Args:
            encoder_dim: Encoder output dimension
            decoder_dim: Decoder output dimension
            joint_dim: Joint hidden dimension
            vocab_size: Output vocabulary size
        """
        super(JointNetwork, self).__init__()
        
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
        self.decoder_proj = nn.Linear(decoder_dim, joint_dim)
        self.output_proj = nn.Linear(joint_dim, vocab_size)
        
        logger.info(
            f"Initialized JointNetwork: encoder_dim={encoder_dim}, "
            f"decoder_dim={decoder_dim}, joint_dim={joint_dim}, "
            f"vocab_size={vocab_size}"
        )
        
    def forward(
        self,
        encoder_output: torch.Tensor,
        decoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            encoder_output: Encoder output (batch, enc_len, encoder_dim)
            decoder_output: Decoder output (batch, dec_len, decoder_dim)
            
        Returns:
            Joint output (batch, enc_len, dec_len, vocab_size)
        """
        # Project encoder and decoder outputs
        enc_proj = self.encoder_proj(encoder_output)  # (batch, enc_len, joint_dim)
        dec_proj = self.decoder_proj(decoder_output)  # (batch, dec_len, joint_dim)
        
        # Add encoder and decoder projections with broadcasting
        # (batch, enc_len, 1, joint_dim) + (batch, 1, dec_len, joint_dim)
        joint = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)
        
        # Apply activation
        joint = torch.tanh(joint)
        
        # Project to vocabulary
        output = self.output_proj(joint)  # (batch, enc_len, dec_len, vocab_size)
        
        return output


def main():
    """Test decoder modules."""
    batch_size = 4
    enc_seq_len = 100
    dec_seq_len = 20
    encoder_dim = 128
    vocab_size = 50
    
    encoder_output = torch.randn(batch_size, enc_seq_len, encoder_dim)
    target_tokens = torch.randint(0, vocab_size, (batch_size, dec_seq_len))
    
    print("="*60)
    print("Testing Decoder Modules")
    print("="*60)
    
    # Test CTC Decoder
    print("\n1. Testing CTCDecoder...")
    model = CTCDecoder(input_dim=encoder_dim, vocab_size=vocab_size)
    log_probs = model(encoder_output)
    decoded = model.decode_greedy(log_probs)
    print(f"   Encoder output shape: {encoder_output.shape}")
    print(f"   Log probs shape: {log_probs.shape}")
    print(f"   Decoded sequences: {len(decoded)} sequences")
    
    # Test Attention Decoder
    print("\n2. Testing AttentionDecoder...")
    model = AttentionDecoder(
        vocab_size=vocab_size,
        embedding_dim=128,
        encoder_dim=encoder_dim,
        hidden_dim=256
    )
    output, hidden = model(encoder_output, target_tokens)
    print(f"   Output shape: {output.shape}")
    
    # Test Transducer Decoder + Joint Network
    print("\n3. Testing TransducerDecoder + JointNetwork...")
    decoder = TransducerDecoder(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256)
    joint = JointNetwork(encoder_dim=encoder_dim, decoder_dim=256, joint_dim=512, vocab_size=vocab_size)
    
    decoder_output, _ = decoder(target_tokens)
    joint_output = joint(encoder_output, decoder_output)
    
    print(f"   Decoder output shape: {decoder_output.shape}")
    print(f"   Joint output shape: {joint_output.shape}")
    
    print("\n" + "="*60)
    print("All decoder modules tested successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
