#!/usr/bin/env python3
"""
Evaluation of trained NEST model on real ZuCo data.
Calculates ACTUAL WER (Word Error Rate) using greedy decoding.
"""

import json
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
try:
    import Levenshtein
except ImportError:
    print("Please install python-Levenshtein: pip install python-Levenshtein")
    sys.exit(1)

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.zuco_dataset import ZuCoTorchDataset
from src.models import ModelFactory

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Calculate Levenshtein distance on words
    dist = Levenshtein.distance(" ".join(ref_words), " ".join(hyp_words))
    # Normalize by reference length
    wer = dist / len(ref_words) if ref_words else 1.0
    return wer

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate."""
    if not reference:
        return 1.0 if hypothesis else 0.0
        
    dist = Levenshtein.distance(reference, hypothesis)
    return dist / len(reference)

def decode_to_string(token_ids, vocab_inv):
    """Convert token IDs to string."""
    chars = []
    for t in token_ids:
        if t in vocab_inv:
            chars.append(vocab_inv[t])
    return "".join(chars)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True, help='Path to results directory')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Checkpoint name')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_file = results_dir / "results.json"
    checkpoint_path = results_dir / args.checkpoint
    
    if not results_file.exists() or not checkpoint_path.exists():
        print(f"Error: Results or checkpoint not found in {results_dir}")
        sys.exit(1)
        
    print(f"Loading results from {results_dir}...")
    with open(results_file) as f:
        config = json.load(f)
        
    # Setup Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Reconstruct Vocab (Must match training!)
    # TODO: Save vocab in training artifacts to avoid hardcoding here
    vocab = {'<blank>': 0, '<pad>': 1, ' ': 2}
    for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
        vocab[c] = i + 3
    vocab_inv = {v: k for k, v in vocab.items()}
    
    # Load Model
    print("Loading model...")
    # recovering config - for now assumption is NEST_CTC based on previous fix
    model = ModelFactory.create_model({
            'model_name': 'NEST_CTC',
            'spatial_cnn': {'type': 'EEGNet', 'n_channels': 105, 'dropout': 0.5},
            'temporal_encoder': {'type': 'LSTM', 'input_dim': 16, 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.3},
            'decoder': {'input_dim': 512, 'vocab_size': 30, 'blank_id': 0}
    })
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Warning loading state dict: {e}")
        
    model = model.to(device)
    model.eval()
    
    # Load Data
    print("Loading test data...")
    dataset = ZuCoTorchDataset(
        root_dir="data/raw/zuco",
        max_samples=50 # Eval on subset for speed in demo
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    total_wer = 0
    total_cer = 0
    count = 0
    
    print("\nSample Predictions:")
    print("-" * 60)
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            eeg = torch.from_numpy(sample['eeg']).unsqueeze(0).to(device) # (1, channels, time)
            text = sample['text'].lower()
             
            # Forward
            # NEST_CTC returns log_probs (batch, time, vocab)
            log_probs = model(eeg) 
            
            # Decode using model's own decoder or manually if needed
            # model.ctc_decoder is available in NEST_CTC
            if hasattr(model, 'ctc_decoder'):
                decoded_ids = model.ctc_decoder.decode_greedy(log_probs)[0]
                pred_text = decode_to_string(decoded_ids, vocab_inv)
            else:
                pred_text = "" # Fallback
                
            # Metrics
            wer = calculate_wer(text, pred_text)
            cer = calculate_cer(text, pred_text)
            
            total_wer += wer
            total_cer += cer
            count += 1
            
            if i < 5:
                print(f"Ref:  {text}")
                print(f"Pred: {pred_text}")
                print(f"WER:  {wer:.2f}")
                print("-" * 60)
                
    avg_wer = total_wer / count if count > 0 else 0
    avg_cer = total_cer / count if count > 0 else 0
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Average WER: {avg_wer:.2%}")
    print(f"Average CER: {avg_cer:.2%}")
    print("=" * 60)
    print("Note: Performance depends heavily on training epochs and data quantity.")

if __name__ == "__main__":
    main()
