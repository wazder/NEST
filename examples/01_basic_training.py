#!/usr/bin/env python3
"""
Basic Example: Train a simple NEST model on ZuCo dataset

This example demonstrates the complete workflow:
1. Load and preprocess data
2. Create a model
3. Train the model
4. Evaluate on test set
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from src.preprocessing import PreprocessingPipeline
from src.data.zuco_dataset import ZucoDataset
from src.models import ModelFactory
from src.training import Trainer, get_optimizer, get_scheduler
from src.utils.tokenizer import CharTokenizer
from src.training.metrics import compute_wer, compute_cer

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    data_dir = Path('data/processed/zuco')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 1. Create tokenizer
    # For simplicity, using character-level tokenizer
    chars = list('abcdefghijklmnopqrstuvwxyz .,!?\'')
    tokenizer = CharTokenizer(vocab=chars)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # 2. Load datasets
    print("\nLoading datasets...")
    train_dataset = ZucoDataset(
        data_dir=data_dir / 'train',
        tokenizer=tokenizer,
        max_seq_len=256
    )
    
    val_dataset = ZucoDataset(
        data_dir=data_dir / 'val',
        tokenizer=tokenizer,
        max_seq_len=256
    )
    
    test_dataset = ZucoDataset(
        data_dir=data_dir / 'test',
        tokenizer=tokenizer,
        max_seq_len=256
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 3. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=val_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.collate_fn
    )
    
    # 4. Create model
    print("\nCreating model...")
    model = ModelFactory.from_config_file(
        config_path='configs/model.yaml',
        model_key='nest_attention',
        vocab_size=vocab_size
    )
    model = model.to(device)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # 5. Setup training
    print("\nSetting up training...")
    optimizer = get_optimizer(
        model,
        optimizer_type='adamw',
        learning_rate=1e-4,
        weight_decay=0.01
    )
    
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        T_max=50,
        eta_min=1e-6
    )
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # 6. Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        clip_grad_norm=1.0
    )
    
    # 7. Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        save_path=checkpoint_dir / 'best_model.pt',
        early_stopping_patience=10,
        log_interval=10
    )
    
    # 8. Print training results
    print("\n" + "="*50)
    print("Training Results:")
    print("="*50)
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation WER: {min(history.get('val_wer', [float('inf')])):.2f}%")
    
    # 9. Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test WER: {test_metrics.get('wer', 0):.2f}%")
    print(f"Test CER: {test_metrics.get('cer', 0):.2f}%")
    
    # 10. Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'test_metrics': test_metrics,
        'config': {
            'vocab_size': vocab_size,
            'tokenizer_vocab': chars
        }
    }, final_path)
    print(f"\nFinal model saved to {final_path}")

if __name__ == '__main__':
    main()
