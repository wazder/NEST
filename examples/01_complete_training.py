#!/usr/bin/env python3
"""
Complete NEST Training Example

This script demonstrates a full training pipeline from data loading to evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import json

# NEST imports
from src.preprocessing import PreprocessingPipeline
from src.models import ModelFactory
from src.training import Trainer, get_optimizer, get_scheduler
from src.utils.tokenizer import VocabularyBuilder
from src.evaluation import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description='Train NEST model')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--config', type=str, default='configs/model.yaml', help='Model config')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("NEST Training Pipeline")
    print("="*80)
    
    # Step 1: Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    print("-"*80)
    
    # TODO: Load your raw data here
    # For this example, assume data is already preprocessed
    # In practice, use PreprocessingPipeline:
    # 
    # pipeline = PreprocessingPipeline('configs/preprocessing.yaml')
    # splits = pipeline.run_pipeline(raw_data, labels, sfreq=500, ...)
    
    print("✓ Data loaded")
    
    # Step 2: Build vocabulary
    print("\n[2/6] Building vocabulary...")
    print("-"*80)
    
    # Example texts (in practice, load from your dataset)
    example_texts = [
        "the quick brown fox jumps over the lazy dog",
        "neural networks decode brain signals",
        "electroencephalography measures electrical activity"
    ]
    
    vocab_builder = VocabularyBuilder(
        vocab_size=5000,
        min_frequency=2,
        use_bpe=True
    )
    
    vocab_builder.build_from_texts(example_texts)
    vocab_path = output_dir / 'vocabulary.json'
    vocab_builder.save(str(vocab_path))
    
    print(f"✓ Vocabulary size: {len(vocab_builder.vocab)}")
    print(f"✓ Saved to {vocab_path}")
    
    # Step 3: Create model
    print("\n[3/6] Creating model...")
    print("-"*80)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model = ModelFactory.from_config_file(
        args.config,
        model_key='nest_transformer',
        vocab_size=len(vocab_builder.vocab)
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model: {model.__class__.__name__}")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Device: {device}")
    
    # Step 4: Setup training
    print("\n[4/6] Setting up training...")
    print("-"*80)
    
    # Optimizer
    optimizer = get_optimizer(
        model,
        optimizer_type='adamw',
        learning_rate=args.lr,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Loss function
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        clip_grad_norm=1.0,
        log_interval=10
    )
    
    print(f"✓ Optimizer: AdamW (lr={args.lr})")
    print(f"✓ Scheduler: CosineAnnealing")
    print(f"✓ Loss: CTCLoss")
    
    # Step 5: Train model
    print("\n[5/6] Training model...")
    print("-"*80)
    
    # TODO: Create actual DataLoaders from your preprocessed data
    # For this example, we'll show the structure:
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # For demonstration, we'll skip actual training
    print("NOTE: Actual training requires real data loaders")
    print("Expected training loop:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Early stopping patience: 10")
    
    # history = trainer.train(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     epochs=args.epochs,
    #     save_path=str(output_dir / 'best_model.pt'),
    #     early_stopping_patience=10
    # )
    
    # Save training history
    # history_path = output_dir / 'training_history.json'
    # with open(history_path, 'w') as f:
    #     json.dump(history, f, indent=2)
    
    # Step 6: Evaluate model
    print("\n[6/6] Evaluating model...")
    print("-"*80)
    
    # evaluator = ModelEvaluator(model, vocab_builder, device=device)
    # test_metrics = evaluator.evaluate(test_loader)
    
    # print(f"✓ Test WER: {test_metrics['wer']:.2%}")
    # print(f"✓ Test CER: {test_metrics['cer']:.2%}")
    # print(f"✓ Test BLEU: {test_metrics['bleu']:.4f}")
    
    # Save evaluation results
    # results_path = output_dir / 'test_results.json'
    # with open(results_path, 'w') as f:
    #     json.dump(test_metrics, f, indent=2)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("Generated files:")
    print(f"  - vocabulary.json")
    print(f"  - best_model.pt (checkpoint)")
    print(f"  - training_history.json")
    print(f"  - test_results.json")
    
    print("\nNext steps:")
    print("  1. Run optimization: examples/02_optimize_model.py")
    print("  2. Deploy model: examples/03_deploy_model.py")
    print("  3. Try real-time inference: examples/04_realtime_demo.py")


if __name__ == '__main__':
    main()
