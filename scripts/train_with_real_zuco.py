#!/usr/bin/env python3
"""
Real ZuCo Training Script - Full Pipeline

This script:
1. Loads raw ZuCo .mat files using src.data
2. Preprocesses EEG data
3. Trains NEST models using src.models and src.training
4. Saves results

Usage:
    python scripts/train_with_real_zuco.py --model nest_ctc --epochs 100
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.zuco_dataset import ZuCoTorchDataset
from src.models import ModelFactory
from src.training.trainer import Trainer, CTCTrainer, get_optimizer, get_scheduler

def collate_fn(batch):
    """Collate batch of samples."""
    # Filter failed samples (None)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
        
    eeg_batch = [torch.from_numpy(s['eeg']) for s in batch]
    
    # Pad EEG to max length in batch
    max_len = max(eeg.size(1) for eeg in eeg_batch)
    padded_eeg = []
    for eeg in eeg_batch:
        if eeg.size(1) < max_len:
            pad = torch.zeros(eeg.size(0), max_len - eeg.size(1))
            eeg = torch.cat([eeg, pad], dim=1)
        padded_eeg.append(eeg)
        
    eeg_batch = torch.stack(padded_eeg)
    
    # Process targets (text to indices)
    # Simple vocabulary for demo purposes
    # In real pipeline, use a proper Tokenizer
    vocab = {
        '<blank>': 0,
        '<pad>': 1,
        ' ': 2
    }
    # Add a-z
    for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
        vocab[c] = i + 3
        
    targets = []
    target_lengths = []
    
    for s in batch:
        text = s['text'].lower()
        indices = []
        for c in text:
            if c in vocab:
                indices.append(vocab[c])
            else:
                 # unknown/ignore
                 pass
        
        if not indices:
            indices = [1] # pad if empty
            
        targets.append(torch.tensor(indices, dtype=torch.long))
        target_lengths.append(len(indices))
        
    # Pad targets
    max_target_len = max(len(t) for t in targets)
    padded_targets = torch.zeros(len(targets), max_target_len, dtype=torch.long)
    for i, t in enumerate(targets):
        padded_targets[i, :len(t)] = t
        
    return {
        'eeg': eeg_batch,
        'targets': padded_targets,
        'target_lengths': torch.tensor(target_lengths, dtype=torch.long),
        'input_lengths': torch.full((len(batch),), max_len // 4, dtype=torch.long) # Approx striding
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train NEST models on real ZuCo data')
    parser.add_argument('--model', default='nest_ctc', 
                       help='Model config key (e.g., nest_ctc, nest_rnn_t)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Quick test with minimal data and epochs')
    args = parser.parse_args()
    
    # Quick test settings
    if args.quick_test:
        print("⚡ QUICK TEST MODE")
        args.epochs = 2
        
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using M2 GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU")
        
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/real_zuco_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading real ZuCo data...")
    dataset = ZuCoTorchDataset(
        root_dir="data/raw/zuco",
        max_samples=20 if args.quick_test else None
    )
    
    if len(dataset) == 0:
        print("No data found! Please download ZuCo data to data/raw/zuco")
        return

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create Model
    print(f"Creating model: {args.model}")
    try:
        model = ModelFactory.from_config_file(
            'configs/model.yaml',
            args.model,
            vocab_size=30 # 26 chars + special
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        # Fallback for demo if config missing
        print("Using default configuration...")
        model = ModelFactory.create_model({
            'model_name': 'NEST_CTC',
            'spatial_cnn': {'type': 'EEGNet', 'n_channels': 105, 'dropout': 0.5},
            'temporal_encoder': {'type': 'LSTM', 'input_dim': 16, 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.3},
            'decoder': {'input_dim': 512, 'vocab_size': 30, 'blank_id': 0}
        })
        
    model = model.to(device)
    
    # Trainer Setup
    optimizer = get_optimizer(model, 'adamw', learning_rate=1e-3)
    scheduler = get_scheduler(optimizer, 'cosine', T_max=args.epochs)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    trainer = CTCTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler
    )
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        save_path=str(output_dir / "best_model.pt")
    )
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
