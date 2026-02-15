#!/usr/bin/env python3
"""
Advanced Example: Subject Adaptation and Domain Adversarial Training

This example demonstrates:
1. Cross-subject generalization
2. Subject-specific adaptation
3. Domain adversarial training (DANN)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from src.data.zuco_dataset import ZucoDataset
from src.models import ModelFactory
from src.models.adaptation import SubjectAdapter, DomainAdversarialNetwork
from src.training import Trainer, get_optimizer
from src.utils.tokenizer import CharTokenizer

def train_subject_adaptation():
    """Train model with subject embeddings"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*50)
    print("Subject Adaptation Training")
    print("="*50)
    
    # Create tokenizer
    chars = list('abcdefghijklmnopqrstuvwxyz .,!?\'')
    tokenizer = CharTokenizer(vocab=chars)
    
    # Load data
    train_dataset = ZucoDataset(
        data_dir='data/processed/zuco/train',
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    # Create base model
    base_model = ModelFactory.from_config_file(
        'configs/model.yaml',
        model_key='nest_attention',
        vocab_size=len(tokenizer)
    )
    
    # Wrap with subject adapter
    n_subjects = len(set(train_dataset.subject_ids))
    model = SubjectAdapter(
        base_model=base_model,
        n_subjects=n_subjects,
        embedding_dim=64
    ).to(device)
    
    print(f"Training with {n_subjects} subjects")
    
    # Setup training
    optimizer = get_optimizer(model, 'adamw', learning_rate=1e-4)
    criterion = nn.CTCLoss(blank=0)
    
    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            tokens = batch['tokens'].to(device)
            subject_ids = batch['subject_id'].to(device)
            
            # Forward pass with subject adaptation
            output = model(eeg, subject_ids)
            loss = criterion(output, tokens)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'checkpoints/subject_adapted.pt')
    print("\nModel saved to checkpoints/subject_adapted.pt")

def train_domain_adversarial():
    """Train with domain adversarial network (DANN)"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*50)
    print("Domain Adversarial Training (DANN)")
    print("="*50)
    
    # Create tokenizer
    chars = list('abcdefghijklmnopqrstuvwxyz .,!?\'')
    tokenizer = CharTokenizer(vocab=chars)
    
    # Load data
    train_dataset = ZucoDataset(
        data_dir='data/processed/zuco/train',
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    # Create model
    base_model = ModelFactory.from_config_file(
        'configs/model.yaml',
        model_key='nest_attention',
        vocab_size=len(tokenizer)
    )
    
    # Wrap with DANN
    n_domains = len(set(train_dataset.subject_ids))
    model = DomainAdversarialNetwork(
        feature_extractor=base_model.encoder,
        classifier=base_model.decoder,
        n_domains=n_domains
    ).to(device)
    
    print(f"Training DANN with {n_domains} domains")
    
    # Setup training
    optimizer = get_optimizer(model, 'adamw', learning_rate=1e-4)
    
    # Training loop with gradient reversal
    model.train()
    for epoch in range(10):
        total_class_loss = 0
        total_domain_loss = 0
        
        # Annealing parameter for gradient reversal
        alpha = 2.0 / (1.0 + np.exp(-10 * epoch / 10)) - 1.0
        
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            tokens = batch['tokens'].to(device)
            domain_ids = batch['subject_id'].to(device)
            
            # Forward pass
            class_loss, domain_loss = model(eeg, tokens, domain_ids, alpha=alpha)
            
            # Total loss (minimize class loss, maximize domain confusion)
            loss = class_loss + 0.1 * domain_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
        
        avg_class = total_class_loss / len(train_loader)
        avg_domain = total_domain_loss / len(train_loader)
        print(f"Epoch {epoch+1}/10 - Class Loss: {avg_class:.4f}, "
              f"Domain Loss: {avg_domain:.4f}, Alpha: {alpha:.3f}")
    
    # Save model
    torch.save(model.state_dict(), 'checkpoints/dann_model.pt')
    print("\nModel saved to checkpoints/dann_model.pt")

def evaluate_cross_subject():
    """Evaluate cross-subject generalization"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*50)
    print("Cross-Subject Evaluation")
    print("="*50)
    
    # Load models and compare
    # This would compare baseline vs subject-adapted vs DANN models
    # on held-out subjects
    
    print("Evaluating models on unseen subjects...")
    print("(Implementation would compare WER across different adaptation methods)")

def main():
    # Train with subject adaptation
    train_subject_adaptation()
    
    # Train with domain adversarial network
    train_domain_adversarial()
    
    # Evaluate cross-subject performance
    evaluate_cross_subject()
    
    print("\n" + "="*50)
    print("Advanced training complete!")
    print("="*50)

if __name__ == '__main__':
    main()
