"""
Evaluation Metrics for EEG-to-Text Models

This module provides metrics for evaluating sequence transduction models:
- Word Error Rate (WER)
- Character Error Rate (CER)
- BLEU Score
- Perplexity
"""

import numpy as np
from typing import List, Union
import Levenshtein


def word_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (Substitutions + Deletions + Insertions) / Number of Words in Reference
    
    Args:
        references: List of reference sentences
        hypotheses: List of hypothesis sentences
        
    Returns:
        Word error rate (0-1 scale)
    """
    total_distance = 0
    total_words = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        distance = Levenshtein.distance(ref_words, hyp_words)
        total_distance += distance
        total_words += len(ref_words)
        
    if total_words == 0:
        return 0.0
        
    return total_distance / total_words


def character_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (Substitutions + Deletions + Insertions) / Number of Characters in Reference
    
    Args:
        references: List of reference sentences
        hypotheses: List of hypothesis sentences
        
    Returns:
        Character error rate (0-1 scale)
    """
    total_distance = 0
    total_chars = 0
    
    for ref, hyp in zip(references, hypotheses):
        distance = Levenshtein.distance(ref, hyp)
        total_distance += distance
        total_chars += len(ref)
        
    if total_chars == 0:
        return 0.0
        
    return total_distance / total_chars


def compute_bleu(
    references: List[List[str]],
    hypotheses: List[str],
    max_n: int = 4,
    weights: List[float] = None
) -> float:
    """
    Compute BLEU score.
    
    Args:
        references: List of reference lists (multiple references per hypothesis)
        hypotheses: List of hypothesis sentences
        max_n: Maximum n-gram order
        weights: Weights for each n-gram order (default: uniform)
        
    Returns:
        BLEU score (0-1 scale)
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
        
    # Calculate n-gram precisions
    precisions = []
    
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        
        for refs, hyp in zip(references, hypotheses):
            hyp_ngrams = get_ngrams(hyp.split(), n)
            
            # Find max match count with any reference
            max_ref_counts = {}
            for ref in refs:
                ref_ngrams = get_ngrams(ref.split(), n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(
                        max_ref_counts.get(ngram, 0),
                        count
                    )
                    
            # Count matches
            for ngram, count in hyp_ngrams.items():
                matches += min(count, max_ref_counts.get(ngram, 0))
                
            total += max(len(hyp.split()) - n + 1, 0)
            
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)
            
    # Calculate brevity penalty
    ref_length = sum(
        min(len(ref.split()) for ref in refs)
        for refs in references
    )
    hyp_length = sum(len(hyp.split()) for hyp in hypotheses)
    
    if hyp_length > ref_length:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_length / (hyp_length + 1e-10))
        
    # Calculate BLEU
    if min(precisions) == 0:
        return 0.0
        
    log_precisions = [w * np.log(p) for w, p in zip(weights, precisions)]
    bleu = bp * np.exp(sum(log_precisions))
    
    return bleu


def get_ngrams(tokens: List[str], n: int) -> dict:
    """
    Extract n-grams from tokens.
    
    Args:
        tokens: List of tokens
        n: N-gram order
        
    Returns:
        Dictionary mapping n-grams to counts
    """
    ngrams = {}
    
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
        
    return ngrams


def perplexity(log_probs: np.ndarray, lengths: np.ndarray) -> float:
    """
    Calculate perplexity.
    
    Perplexity = exp(-1/N * sum(log P(w_i)))
    
    Args:
        log_probs: Log probabilities (batch_size, seq_len)
        lengths: Actual sequence lengths (batch_size,)
        
    Returns:
        Perplexity score
    """
    total_log_prob = 0.0
    total_tokens = 0
    
    for log_prob, length in zip(log_probs, lengths):
        total_log_prob += log_prob[:length].sum()
        total_tokens += length
        
    if total_tokens == 0:
        return float('inf')
        
    avg_log_prob = total_log_prob / total_tokens
    ppl = np.exp(-avg_log_prob)
    
    return ppl


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate token-level accuracy.
    
    Args:
        predictions: Predicted token IDs (batch_size, seq_len)
        targets: Target token IDs (batch_size, seq_len)
        
    Returns:
        Accuracy (0-1 scale)
    """
    correct = (predictions == targets).sum()
    total = targets.size
    
    return correct / total if total > 0 else 0.0


class MetricsTracker:
    """Track and aggregate metrics during training/evaluation."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
        
    def update(self, metric_dict: dict):
        """
        Update metrics.
        
        Args:
            metric_dict: Dictionary of metric names to values
        """
        for name, value in metric_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
                
            self.metrics[name] += value
            self.counts[name] += 1
            
    def get_averages(self) -> dict:
        """
        Get average metrics.
        
        Returns:
            Dictionary of metric names to average values
        """
        averages = {}
        
        for name in self.metrics:
            if self.counts[name] > 0:
                averages[name] = self.metrics[name] / self.counts[name]
            else:
                averages[name] = 0.0
                
        return averages
        
    def __repr__(self) -> str:
        """String representation."""
        averages = self.get_averages()
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in averages.items())
        return f"MetricsTracker({metrics_str})"


def main():
    """Example usage."""
    print("="*60)
    print("Evaluation Metrics")
    print("="*60)
    
    # Example sentences
    references = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test"
    ]
    hypotheses = [
        "the quick brown fox jumps over a lazy dog",
        "hello world this is test"
    ]
    
    # WER
    wer = word_error_rate(references, hypotheses)
    print(f"\nWord Error Rate: {wer:.4f} ({wer*100:.2f}%)")
    
    # CER
    cer = character_error_rate(references, hypotheses)
    print(f"Character Error Rate: {cer:.4f} ({cer*100:.2f}%)")
    
    # BLEU (single reference)
    refs_multi = [[ref] for ref in references]
    bleu = compute_bleu(refs_multi, hypotheses)
    print(f"BLEU Score: {bleu:.4f}")
    
    # Perplexity
    log_probs = np.array([
        [-0.5, -0.3, -0.7, -0.2],
        [-0.4, -0.6, -0.3, -0.5]
    ])
    lengths = np.array([4, 3])
    ppl = perplexity(log_probs, lengths)
    print(f"Perplexity: {ppl:.4f}")
    
    # Accuracy
    predictions = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    targets = np.array([[1, 2, 4, 4], [5, 6, 7, 9]])
    acc = accuracy(predictions, targets)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # MetricsTracker
    print("\nMetrics Tracker:")
    tracker = MetricsTracker()
    tracker.update({'loss': 0.5, 'acc': 0.8})
    tracker.update({'loss': 0.4, 'acc': 0.85})
    tracker.update({'loss': 0.3, 'acc': 0.9})
    print(tracker)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
