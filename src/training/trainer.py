"""
Training Utilities for NEST Models

This module provides training loops, metrics, and utilities for
training EEG-to-text models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable, List
import logging
from tqdm import tqdm
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Generic trainer for NEST models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        clip_grad_norm: Optional[float] = None,
        log_interval: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            clip_grad_norm: Gradient clipping threshold (optional)
            log_interval: Logging interval in batches
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        
        self.train_losses = []
        self.val_losses = []
        
        logger.info(
            f"Initialized Trainer: device={device}, "
            f"clip_grad={clip_grad_norm}"
        )
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            eeg_data = batch['eeg'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(eeg_data, targets)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )
                
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate metrics
            batch_size = eeg_data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / total_samples
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
        avg_loss = total_loss / total_samples
        self.train_losses.append(avg_loss)
        
        metrics = {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
        
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                eeg_data = batch['eeg'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                outputs = self.model(eeg_data, targets)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Accumulate metrics
                batch_size = eeg_data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Update progress bar
                avg_loss = total_loss / total_samples
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
        avg_loss = total_loss / total_samples
        self.val_losses.append(avg_loss)
        
        metrics = {
            'val_loss': avg_loss
        }
        
        return metrics
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: Optional[str] = None,
        early_stopping_patience: Optional[int] = None
    ) -> Dict:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_path: Path to save best model (optional)
            early_stopping_patience: Early stopping patience (optional)
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Log metrics
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"LR: {train_metrics['learning_rate']:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Save best model
            if save_path and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, save_path)
                logger.info(f"Saved best model to {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience: {early_stopping_patience})"
                )
                break
                
        logger.info("Training complete!")
        
        return history


class CTCTrainer(Trainer):
    """Specialized trainer for CTC models."""
    
    def __init__(self, *args, **kwargs):
        """Initialize CTC trainer."""
        super().__init__(*args, **kwargs)
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train epoch for CTC."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            eeg_data = batch['eeg'].to(self.device)
            targets = batch['targets'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            log_probs = self.model(eeg_data)  # (batch, time, vocab)
            log_probs = log_probs.transpose(0, 1)  # (time, batch, vocab) for CTC
            
            # CTC loss
            loss = self.criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )
            
            loss.backward()
            
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )
                
            self.optimizer.step()
            
            batch_size = eeg_data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({'loss': f'{total_loss / total_samples:.4f}'})
                
        avg_loss = total_loss / total_samples
        self.train_losses.append(avg_loss)
        
        return {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.should_stop


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer by name.
    
    Args:
        model: PyTorch model
        optimizer_name: Optimizer name ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'step',
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: Scheduler name ('step', 'cosine', 'plateau')
        **kwargs: Scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10)
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
    return scheduler


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Example usage."""
    print("="*60)
    print("Training Utilities")
    print("="*60)
    
    # Create dummy model
    model = nn.Linear(10, 5)
    
    # Count parameters
    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,}")
    
    # Get optimizer
    optimizer = get_optimizer(model, 'adam', learning_rate=1e-3)
    print(f"Optimizer: {optimizer.__class__.__name__}")
    
    # Get scheduler
    scheduler = get_scheduler(optimizer, 'step', step_size=10)
    print(f"Scheduler: {scheduler.__class__.__name__}")
    
    # Early stopping
    early_stop = EarlyStopping(patience=5)
    print(f"Early stopping initialized with patience={early_stop.patience}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
