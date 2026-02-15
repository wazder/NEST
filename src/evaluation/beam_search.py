"""
Beam Search Decoder for NEST Models

Implements efficient beam search decoding with:
- Length normalization
- Coverage penalty
- N-best hypotheses
- Batch beam search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Hypothesis:
    """Single hypothesis in beam search."""
    tokens: List[int]
    score: float
    coverage: Optional[torch.Tensor] = None
    
    def __lt__(self, other):
        """For sorting."""
        return self.score < other.score


class BeamSearchDecoder:
    """
    Beam search decoder with various features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        sos_token: int,
        eos_token: int,
        beam_size: int = 5,
        max_length: int = 100,
        length_penalty: float = 0.6,
        coverage_penalty: float = 0.0,
        min_length: int = 1
    ):
        """
        Initialize beam search decoder.
        
        Args:
            model: NEST model
            vocab_size: Vocabulary size
            sos_token: Start-of-sequence token ID
            eos_token: End-of-sequence token ID
            beam_size: Beam width
            max_length: Maximum generation length
            length_penalty: Length normalization (α in Wu et al. 2016)
            coverage_penalty: Coverage penalty weight
            min_length: Minimum generation length
        """
        self.model = model
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
        self.min_length = min_length
        
    def decode(
        self,
        encoder_output: torch.Tensor,
        return_nbest: int = 1
    ) -> List[Hypothesis]:
        """
        Beam search decoding.
        
        Args:
            encoder_output: Encoded EEG features (1, seq_len, hidden_dim)
            return_nbest: Return top N hypotheses
            
        Returns:
            List of top N hypotheses
        """
        device = encoder_output.device
        batch_size = encoder_output.size(0)
        
        assert batch_size == 1, "Beam search only supports batch_size=1"
        
        # Initialize beam
        hypotheses = [Hypothesis(tokens=[self.sos_token], score=0.0)]
        completed = []
        
        # Beam search loop
        for step in range(self.max_length):
            all_candidates = []
            
            for hyp in hypotheses:
                # Don't expand completed hypotheses
                if hyp.tokens[-1] == self.eos_token:
                    completed.append(hyp)
                    continue
                    
                # Get next token probabilities
                input_tokens = torch.tensor([hyp.tokens], device=device)
                
                with torch.no_grad():
                    logits = self.model.decode_step(encoder_output, input_tokens)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                    
                # Get top K tokens
                top_scores, top_tokens = log_probs.topk(self.beam_size, dim=-1)
                
                # Create new hypotheses
                for score, token in zip(top_scores[0], top_tokens[0]):
                    new_tokens = hyp.tokens + [token.item()]
                    new_score = hyp.score + score.item()
                    
                    # Apply length normalization (during search, not just at end)
                    lp = self._length_penalty(len(new_tokens))
                    normalized_score = new_score / lp
                    
                    # Don't allow EOS too early
                    if token == self.eos_token and len(new_tokens) < self.min_length:
                        continue
                        
                    new_hyp = Hypothesis(
                        tokens=new_tokens,
                        score=normalized_score
                    )
                    all_candidates.append(new_hyp)
                    
            if not all_candidates:
                break
                
            # Select top beam_size hypotheses
            ordered = sorted(all_candidates, key=lambda x: x.score, reverse=True)
            hypotheses = ordered[:self.beam_size]
            
            # Check if all hypotheses are complete
            if all(h.tokens[-1] == self.eos_token for h in hypotheses):
                completed.extend(hypotheses)
                break
                
        # Add remaining hypotheses to completed
        completed.extend(hypotheses)
        
        # Sort by score and return top N
        completed = sorted(completed, key=lambda x: x.score, reverse=True)
        
        return completed[:return_nbest]
        
    def _length_penalty(self, length: int) -> float:
        """
        Length penalty (Wu et al. 2016).
        
        LP(Y) = ((5 + |Y|) / 6)^α
        """
        return ((5 + length) / 6) ** self.length_penalty
        
    def _coverage_penalty(self, coverage: torch.Tensor) -> float:
        """
        Coverage penalty to prevent over/under translation.
        
        Args:
            coverage: (seq_len,) - sum of attention weights
            
        Returns:
            Penalty value
        """
        penalty = torch.log(coverage.clamp(min=1.0)).sum()
        return self.coverage_penalty * penalty


class GreedyDecoder:
    """
    Simple greedy decoder (baseline).
    """
    
    def __init__(
        self,
        model: nn.Module,
        sos_token: int,
        eos_token: int,
        max_length: int = 100
    ):
        """
        Initialize greedy decoder.
        
        Args:
            model: NEST model
            sos_token: Start token ID
            eos_token: End token ID
            max_length: Maximum length
        """
        self.model = model
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length
        
    def decode(
        self,
        encoder_output: torch.Tensor
    ) -> List[int]:
        """
        Greedy decoding.
        
        Args:
            encoder_output: (batch, seq_len, hidden_dim)
            
        Returns:
            List of token IDs
        """
        device = encoder_output.device
        batch_size = encoder_output.size(0)
        
        # Initialize
        tokens = [[self.sos_token] for _ in range(batch_size)]
        finished = [False] * batch_size
        
        for _ in range(self.max_length):
            if all(finished):
                break
                
            # Current tokens
            input_tokens = torch.tensor(tokens, device=device)
            
            # Get next token
            with torch.no_grad():
                logits = self.model.decode_step(encoder_output, input_tokens)
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                
            # Update tokens
            for i in range(batch_size):
                if not finished[i]:
                    token = next_tokens[i].item()
                    tokens[i].append(token)
                    
                    if token == self.eos_token:
                        finished[i] = True
                        
        return tokens


class BatchBeamSearch:
    """
    Efficient batch beam search.
    
    Process multiple inputs simultaneously with beam search.
    """
    
    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        sos_token: int,
        eos_token: int,
        beam_size: int = 5,
        max_length: int = 100,
        length_penalty: float = 0.6
    ):
        """
        Initialize batch beam search.
        
        Args:
            model: NEST model
            vocab_size: Vocabulary size
            sos_token: SOS token
            eos_token: EOS token
            beam_size: Beam size
            max_length: Max length
            length_penalty: Length penalty
        """
        self.model = model
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        
    def decode(
        self,
        encoder_output: torch.Tensor
    ) -> List[List[int]]:
        """
        Batch beam search.
        
        Args:
            encoder_output: (batch, seq_len, hidden_dim)
            
        Returns:
            List of decoded sequences (batch_size, seq_len)
        """
        device = encoder_output.device
        batch_size = encoder_output.size(0)
        
        # Expand encoder output for beam
        # (batch * beam, seq_len, hidden_dim)
        encoder_output = encoder_output.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
        encoder_output = encoder_output.view(batch_size * self.beam_size, -1, encoder_output.size(-1))
        
        # Initialize beam
        beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        
        # Initial tokens
        input_tokens = torch.full(
            (batch_size * self.beam_size, 1),
            self.sos_token,
            dtype=torch.long,
            device=device
        )
        
        # Track finished beams
        finished = torch.zeros(batch_size, self.beam_size, dtype=torch.bool, device=device)
        
        for step in range(self.max_length):
            # Get next token probabilities
            with torch.no_grad():
                logits = self.model.decode_step(encoder_output, input_tokens)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
            # Shape: (batch * beam, vocab)
            log_probs = log_probs.view(batch_size, self.beam_size, -1)
            
            # Add to beam scores
            scores = beam_scores.unsqueeze(-1) + log_probs
            
            # Flatten to (batch, beam * vocab)
            scores = scores.view(batch_size, -1)
            
            # Get top beam_size scores
            top_scores, top_indices = scores.topk(self.beam_size, dim=-1)
            
            # Recover beam and token IDs
            beam_ids = top_indices // self.vocab_size
            token_ids = top_indices % self.vocab_size
            
            # Update beam scores
            beam_scores = top_scores
            
            # Gather previous tokens
            beam_offset = torch.arange(batch_size, device=device).unsqueeze(-1) * self.beam_size
            gather_indices = (beam_ids + beam_offset).view(-1)
            
            input_tokens = input_tokens[gather_indices]
            
            # Append new tokens
            input_tokens = torch.cat([input_tokens, token_ids.view(-1, 1)], dim=-1)
            
            # Check for EOS
            finished = finished | (token_ids == self.eos_token)
            
            if finished.all():
                break
                
        # Get best hypotheses (first beam)
        best_tokens = input_tokens.view(batch_size, self.beam_size, -1)[:, 0, :]
        
        return best_tokens.tolist()


def main():
    """Example usage."""
    print("="*60)
    print("Beam Search Decoder")
    print("="*60)
    
    # Dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self, vocab_size, hidden_dim):
            super().__init__()
            self.decoder = nn.Linear(hidden_dim, vocab_size)
            
        def decode_step(self, encoder_output, input_tokens):
            # Simple dummy decoder
            batch_size = encoder_output.size(0)
            seq_len = input_tokens.size(1)
            logits = torch.randn(batch_size, seq_len, 1000)
            return logits
            
    model = DummyModel(vocab_size=1000, hidden_dim=512)
    
    # Test beam search
    print("\n1. Beam Search Decoder")
    beam_decoder = BeamSearchDecoder(
        model,
        vocab_size=1000,
        sos_token=0,
        eos_token=1,
        beam_size=5,
        max_length=20,
        length_penalty=0.6
    )
    
    encoder_output = torch.randn(1, 100, 512)
    hypotheses = beam_decoder.decode(encoder_output, return_nbest=3)
    
    print(f"   Generated {len(hypotheses)} hypotheses")
    for i, hyp in enumerate(hypotheses):
        print(f"   Hypothesis {i+1}: length={len(hyp.tokens)}, score={hyp.score:.4f}")
        
    # Test greedy decoder
    print("\n2. Greedy Decoder")
    greedy_decoder = GreedyDecoder(model, sos_token=0, eos_token=1, max_length=20)
    
    tokens = greedy_decoder.decode(encoder_output)
    print(f"   Generated tokens: {len(tokens[0])} tokens")
    
    # Test batch beam search
    print("\n3. Batch Beam Search")
    batch_beam = BatchBeamSearch(
        model,
        vocab_size=1000,
        sos_token=0,
        eos_token=1,
        beam_size=5,
        max_length=20
    )
    
    batch_encoder_output = torch.randn(4, 100, 512)
    batch_tokens = batch_beam.decode(batch_encoder_output)
    
    print(f"   Batch size: {len(batch_tokens)}")
    print(f"   Sequence lengths: {[len(t) for t in batch_tokens]}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
