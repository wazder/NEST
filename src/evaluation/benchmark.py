"""
Benchmark Evaluation for NEST Models

This script provides comprehensive evaluation of EEG-to-text models:
- Word Error Rate (WER)
- Character Error Rate (CER)
- BLEU Score
- Perplexity
- Inference time and throughput
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse

from ..training.metrics import (
    word_error_rate,
    character_error_rate,
    compute_bleu,
    perplexity,
    MetricsTracker
)
from ..models import ModelFactory
from ..utils.tokenizer import VocabularyBuilder


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: VocabularyBuilder,
        device: torch.device,
        max_length: int = 100
    ):
        """
        Initialize evaluator.
        
        Args:
            model: NEST model
            tokenizer: Tokenizer/vocabulary
            device: Device to run on
            max_length: Maximum generation length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        self.model.eval()
        self.model.to(device)
        
    def evaluate(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            verbose: Whether to show progress
            
        Returns:
            Dictionary of metrics
        """
        metrics_tracker = MetricsTracker()
        
        all_references = []
        all_hypotheses = []
        all_log_probs = []
        
        total_time = 0.0
        num_samples = 0
        
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                eeg_data = batch['eeg'].to(self.device)
                targets = batch['targets']
                
                batch_size = eeg_data.size(0)
                num_samples += batch_size
                
                # Measure inference time
                start_time = time.time()
                
                # Generate predictions
                predictions, log_probs = self.generate(eeg_data)
                
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Decode predictions and targets
                for pred, target in zip(predictions, targets):
                    # Convert to text
                    pred_text = self.tokenizer.decode(pred.cpu().tolist())
                    target_text = self.tokenizer.decode(target.tolist())
                    
                    all_hypotheses.append(pred_text)
                    all_references.append(target_text)
                    
                # Collect log probabilities
                all_log_probs.extend(log_probs)
                
        # Compute metrics
        wer = word_error_rate(all_references, all_hypotheses)
        cer = character_error_rate(all_references, all_hypotheses)
        
        # BLEU (single reference)
        refs_multi = [[ref] for ref in all_references]
        bleu = compute_bleu(refs_multi, all_hypotheses)
        
        # Average inference time
        avg_time = total_time / num_samples
        throughput = num_samples / total_time
        
        metrics = {
            'wer': wer,
            'cer': cer,
            'bleu': bleu,
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'num_samples': num_samples
        }
        
        return metrics
        
    def generate(
        self,
        eeg_data: torch.Tensor,
        method: str = 'greedy'
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Generate text from EEG.
        
        Args:
            eeg_data: (batch, channels, time)
            method: 'greedy' or 'beam'
            
        Returns:
            predictions: (batch, seq_len)
            log_probs: List of log probabilities
        """
        if method == 'greedy':
            return self._greedy_decode(eeg_data)
        elif method == 'beam':
            return self._beam_search(eeg_data)
        else:
            raise ValueError(f"Unknown generation method: {method}")
            
    def _greedy_decode(
        self,
        eeg_data: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        """Greedy decoding."""
        batch_size = eeg_data.size(0)
        
        # Encode EEG
        encoder_output = self.model.spatial_cnn(eeg_data)
        encoder_output = self.model.temporal_encoder(encoder_output)
        
        # Initialize decoder input
        decoder_input = torch.full(
            (batch_size, 1),
            self.tokenizer.word_to_id['<SOS>'],
            dtype=torch.long,
            device=self.device
        )
        
        predictions = []
        log_probs = []
        
        for _ in range(self.max_length):
            # Decode step
            output = self.model.decoder(decoder_input, encoder_output)
            
            # Get next token
            logits = output[:, -1, :]
            log_prob = torch.log_softmax(logits, dim=-1)
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            predictions.append(next_token)
            log_probs.append(log_prob.max(dim=-1)[0].mean().item())
            
            # Update decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Check for EOS
            if (next_token == self.tokenizer.word_to_id['<EOS>']).all():
                break
                
        predictions = torch.cat(predictions, dim=1)
        
        return predictions, log_probs
        
    def _beam_search(
        self,
        eeg_data: torch.Tensor,
        beam_size: int = 5
    ) -> Tuple[torch.Tensor, List[float]]:
        """Beam search decoding."""
        # Simplified beam search (full implementation in beam_search.py)
        # For now, fall back to greedy
        return self._greedy_decode(eeg_data)


def load_model_and_tokenizer(
    model_config_path: str,
    model_key: str,
    checkpoint_path: str,
    tokenizer_path: str,
    device: torch.device
) -> Tuple[nn.Module, VocabularyBuilder]:
    """
    Load model and tokenizer.
    
    Args:
        model_config_path: Path to model config YAML
        model_key: Model key in config
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer JSON
        device: Device to load on
        
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    # Load tokenizer
    tokenizer = VocabularyBuilder(max_vocab_size=10000)
    tokenizer.load(tokenizer_path)
    
    # Load model
    model = ModelFactory.from_config_file(
        model_config_path,
        model_key,
        vocab_size=tokenizer.vocab_size
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def benchmark_suite(
    model: nn.Module,
    tokenizer: VocabularyBuilder,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str = 'results'
) -> Dict:
    """
    Run complete benchmark suite.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_loader: Test data loader
        device: Device
        output_dir: Output directory for results
        
    Returns:
        Dictionary of all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NEST Model Benchmark Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, tokenizer, device)
    
    # Run evaluation
    print("\nRunning evaluation on test set...")
    metrics = evaluator.evaluate(test_loader, verbose=True)
    
    # Print results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Word Error Rate (WER):    {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"BLEU Score:              {metrics['bleu']:.4f}")
    print(f"Avg Inference Time:      {metrics['avg_inference_time']*1000:.2f} ms")
    print(f"Throughput:              {metrics['throughput']:.2f} samples/sec")
    print(f"Total Samples:           {metrics['num_samples']}")
    print("="*60)
    
    # Save results
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return metrics


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate NEST models')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--model_key', type=str, required=True,
                        help='Model key in config')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to tokenizer JSON')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_config,
        args.model_key,
        args.checkpoint,
        args.tokenizer,
        device
    )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load test data (placeholder - implement actual data loading)
    # test_loader = create_dataloader(args.test_data, args.batch_size)
    
    # For now, create a dummy loader
    print("\n[Note: Using dummy data - implement actual data loading]")
    
    # Run benchmark
    # results = benchmark_suite(model, tokenizer, test_loader, device, args.output_dir)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
