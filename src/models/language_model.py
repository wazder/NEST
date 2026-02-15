"""
Language Model Integration for NEST

This module provides integration with pre-trained language models:
- Shallow fusion with external LM
- Deep fusion with LM
- LM rescoring for beam search  
- GPT-2/BERT integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class LanguageModelFusion(nn.Module):
    """
    Base class for language model fusion.
    """
    
    def __init__(
        self,
        lm_model: nn.Module,
        fusion_type: str = 'shallow',
        lm_weight: float = 0.3
    ):
        """
        Initialize LM fusion.
        
        Args:
            lm_model: Pre-trained language model
            fusion_type: 'shallow' or 'deep'
            lm_weight: Weight for LM scores (0-1)
        """
        super().__init__()
        
        self.lm_model = lm_model
        self.fusion_type = fusion_type
        self.lm_weight = lm_weight
        
        # Freeze LM by default
        for param in self.lm_model.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        am_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse acoustic model and language model scores.
        
        Args:
            am_logits: Acoustic model logits (batch, seq_len, vocab_size)
            input_ids: Previous tokens for LM (batch, seq_len)
            
        Returns:
            Fused logits
        """
        raise NotImplementedError


class ShallowFusion(LanguageModelFusion):
    """
    Shallow fusion: combine AM and LM scores at output.
    
    P(y|x) ∝ P_AM(y|x) * P_LM(y)^λ
    """
    
    def forward(
        self,
        am_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Shallow fusion of AM and LM.
        
        Args:
            am_logits: AM logits (batch, seq_len, vocab_size)
            input_ids: Context for LM (batch, context_len)
            
        Returns:
            Fused logits
        """
        batch_size, seq_len, vocab_size = am_logits.shape
        
        # Get LM scores
        if input_ids is not None and len(input_ids.shape) > 0:
            with torch.no_grad():
                lm_output = self.lm_model(input_ids)
                
                # Handle different LM output formats
                if hasattr(lm_output, 'logits'):
                    lm_logits = lm_output.logits
                else:
                    lm_logits = lm_output
                    
                # Take last position if needed
                if lm_logits.size(1) > 1:
                    lm_logits = lm_logits[:, -1:, :]
                    
                # Expand to match AM sequence length
                lm_logits = lm_logits.expand(batch_size, seq_len, vocab_size)
        else:
            # Uniform LM if no context
            lm_logits = torch.zeros_like(am_logits)
            
        # Convert to log probabilities
        am_log_probs = F.log_softmax(am_logits, dim=-1)
        lm_log_probs = F.log_softmax(lm_logits, dim=-1)
        
        # Shallow fusion
        fused_log_probs = (1 - self.lm_weight) * am_log_probs + self.lm_weight * lm_log_probs
        
        return fused_log_probs


class DeepFusion(LanguageModelFusion):
    """
    Deep fusion: learn to combine AM and LM hidden states.
    
    Gulcehre et al. (2015): "On Using Monolingual Corpora in Neural Machine Translation"
    """
    
    def __init__(
        self,
        lm_model: nn.Module,
        am_hidden_dim: int,
        lm_hidden_dim: int,
        vocab_size: int,
        lm_weight: float = 0.3
    ):
        """
        Initialize deep fusion.
        
        Args:
            lm_model: Language model
            am_hidden_dim: AM hidden dimension
            lm_hidden_dim: LM hidden dimension
            vocab_size: Vocabulary size
            lm_weight: Initial LM weight
        """
        super().__init__(lm_model, 'deep', lm_weight)
        
        self.am_hidden_dim = am_hidden_dim
        self.lm_hidden_dim = lm_hidden_dim
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(am_hidden_dim + lm_hidden_dim, am_hidden_dim),
            nn.Sigmoid()
        )
        
        # Combined projection
        self.output_projection = nn.Linear(am_hidden_dim, vocab_size)
        
    def forward(
        self,
        am_hidden: torch.Tensor,
        lm_input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Deep fusion of AM and LM hidden states.
        
        Args:
            am_hidden: AM hidden states (batch, seq_len, am_hidden_dim)
            lm_input_ids: Input IDs for LM (batch, seq_len)
            
        Returns:
            Fused logits
        """
        batch_size, seq_len, _ = am_hidden.shape
        
        # Get LM hidden states
        if lm_input_ids is not None:
            with torch.no_grad():
                lm_output = self.lm_model(lm_input_ids, output_hidden_states=True)
                
                # Extract last hidden state
                if hasattr(lm_output, 'hidden_states'):
                    lm_hidden = lm_output.hidden_states[-1]
                else:
                    lm_hidden = lm_output[0]
                    
                # Match sequence length
                if lm_hidden.size(1) != seq_len:
                    # Repeat or truncate
                    if lm_hidden.size(1) < seq_len:
                        lm_hidden = lm_hidden.repeat(1, (seq_len // lm_hidden.size(1)) + 1, 1)
                    lm_hidden = lm_hidden[:, :seq_len, :]
        else:
            lm_hidden = torch.zeros(batch_size, seq_len, self.lm_hidden_dim).to(am_hidden.device)
            
        # Concatenate AM and LM hidden states
        combined = torch.cat([am_hidden, lm_hidden], dim=-1)
        
        # Compute gate
        gate = self.gate(combined)
        
        # Gated fusion
        fused_hidden = gate * am_hidden + (1 - gate) * am_hidden
        
        # Project to vocabulary
        logits = self.output_projection(fused_hidden)
        
        return logits


class LanguageModelRescorer:
    """
    Rescore beam search hypotheses with language model.
    """
    
    def __init__(
        self,
        lm_model: nn.Module,
        lm_weight: float = 0.3,
        length_penalty: float = 0.6
    ):
        """
        Initialize LM rescorer.
        
        Args:
            lm_model: Language model
            lm_weight: Weight for LM score
            length_penalty: Length normalization penalty
        """
        self.lm_model = lm_model
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty
        
        # Set to eval mode
        self.lm_model.eval()
        
    def score_hypothesis(
        self,
        token_ids: torch.Tensor,
        am_score: float
    ) -> float:
        """
        Score a hypothesis with AM + LM.
        
        Args:
            token_ids: Token IDs (seq_len,)
            am_score: Acoustic model score
            
        Returns:
            Combined score
        """
        with torch.no_grad():
            # Get LM score
            lm_output = self.lm_model(token_ids.unsqueeze(0))
            
            if hasattr(lm_output, 'logits'):
                lm_logits = lm_output.logits
            else:
                lm_logits = lm_output
                
            # Compute log probability
            lm_log_probs = F.log_softmax(lm_logits, dim=-1)
            
            # Get log prob of each token
            seq_len = token_ids.size(0)
            lm_score = 0.0
            
            for i in range(1, seq_len):
                token_log_prob = lm_log_probs[0, i-1, token_ids[i]]
                lm_score += token_log_prob.item()
                
        # Length normalization
        length_norm = math.pow(seq_len, self.length_penalty)
        
        # Combine scores
        combined_score = (am_score + self.lm_weight * lm_score) / length_norm
        
        return combined_score
        
    def rescore_hypotheses(
        self,
        hypotheses: List[Tuple[torch.Tensor, float]]
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Rescore list of hypotheses.
        
        Args:
            hypotheses: List of (token_ids, am_score) tuples
            
        Returns:
            Rescored and sorted hypotheses
        """
        rescored = []
        
        for token_ids, am_score in hypotheses:
            combined_score = self.score_hypothesis(token_ids, am_score)
            rescored.append((token_ids, combined_score))
            
        # Sort by score (descending)
        rescored.sort(key=lambda x: x[1], reverse=True)
        
        return rescored


class SimpleLSTMLM(nn.Module):
    """
    Simple LSTM language model for testing.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM LM.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_hidden_states: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len)
            hidden: Optional hidden state
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Logits: (batch, seq_len, vocab_size)
        """
        # Embed
        embedded = self.embedding(input_ids)
        
        # LSTM
        if hidden is not None:
            output, hidden_state = self.lstm(embedded, hidden)
        else:
            output, hidden_state = self.lstm(embedded)
            
        # Project
        logits = self.output(output)
        
        if output_hidden_states:
            # Return object with logits and hidden_states attributes
            class Output:
                pass
            result = Output()
            result.logits = logits
            result.hidden_states = (output,)
            return result
        
        return logits


def main():
    """Example usage."""
    print("="*60)
    print("Language Model Integration")
    print("="*60)
    
    batch_size = 2
    seq_len = 20
    vocab_size = 1000
    hidden_dim = 512
    
    # Create simple LM
    lm = SimpleLSTMLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
    
    # Test Shallow Fusion
    print("\n1. Shallow Fusion")
    shallow_fusion = ShallowFusion(lm, lm_weight=0.3)
    
    am_logits = torch.randn(batch_size, seq_len, vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    fused_logits = shallow_fusion(am_logits, input_ids)
    print(f"   AM logits: {am_logits.shape}")
    print(f"   Fused logits: {fused_logits.shape}")
    print(f"   LM weight: {shallow_fusion.lm_weight}")
    
    # Test Deep Fusion
    print("\n2. Deep Fusion")
    deep_fusion = DeepFusion(
        lm,
        am_hidden_dim=hidden_dim,
        lm_hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        lm_weight=0.3
    )
    
    am_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    fused_logits = deep_fusion(am_hidden, input_ids)
    
    print(f"   AM hidden: {am_hidden.shape}")
    print(f"   Fused logits: {fused_logits.shape}")
    
    # Test LM Rescorer
    print("\n3. Language Model Rescoring")
    rescorer = LanguageModelRescorer(lm, lm_weight=0.3, length_penalty=0.6)
    
    # Create dummy hypotheses
    hypotheses = [
        (torch.randint(0, vocab_size, (10,)), -5.0),
        (torch.randint(0, vocab_size, (12,)), -6.0),
        (torch.randint(0, vocab_size, (8,)), -4.5)
    ]
    
    rescored = rescorer.rescore_hypotheses(hypotheses)
    
    print(f"   Original hypotheses: {len(hypotheses)}")
    orig_scores = [h[1] for h in hypotheses]
    print(f"   Original scores: {orig_scores}")
    rescored_scores = [h[1] for h in rescored]
    print(f"   Rescored scores: {rescored_scores}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
