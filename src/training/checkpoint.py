"""
Checkpoint Management for NEST Models

This module provides utilities for saving/loading model checkpoints
and managing training state.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = 'checkpoints',
        max_checkpoints: int = 5
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            scheduler: Learning rate scheduler (optional)
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CheckpointManager initialized at {self.checkpoint_dir}")
        
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None
    ) -> str:
        """
        Save checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            filename: Checkpoint filename (auto-generated if None)
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
            
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
        
    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {checkpoint_path}")
        
        # Load optimizer state
        if load_optimizer and self.optimizer is not None:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state")
                
        # Load scheduler state
        if load_scheduler and self.scheduler is not None:
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state")
                
        return checkpoint
        
    def save_best_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        metric_name: str = 'val_loss',
        mode: str = 'min'
    ) -> bool:
        """
        Save checkpoint if it's the best so far.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            metric_name: Metric to track for best model
            mode: 'min' or 'max'
            
        Returns:
            True if checkpoint was saved, False otherwise
        """
        best_checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        
        # Check if this is the best model
        is_best = False
        
        if not best_checkpoint_path.exists():
            is_best = True
        else:
            prev_checkpoint = torch.load(best_checkpoint_path)
            prev_metric = prev_checkpoint['metrics'].get(metric_name)
            curr_metric = metrics.get(metric_name)
            
            if prev_metric is None or curr_metric is None:
                logger.warning(f"Metric '{metric_name}' not found in metrics")
                return False
                
            if mode == 'min':
                is_best = curr_metric < prev_metric
            elif mode == 'max':
                is_best = curr_metric > prev_metric
            else:
                raise ValueError(f"Invalid mode: {mode}")
                
        if is_best:
            self.save_checkpoint(epoch, metrics, filename='best_model.pt')
            logger.info(
                f"Saved best model with {metric_name}={metrics[metric_name]:.4f}"
            )
            return True
            
        return False
        
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        # Get all checkpoint files (excluding best_model.pt)
        checkpoints = sorted(
            [
                f for f in self.checkpoint_dir.glob('checkpoint_epoch_*.pt')
            ],
            key=lambda x: x.stat().st_mtime
        )
        
        # Remove oldest checkpoints
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.info(f"Removed old checkpoint: {oldest}")
            
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        return checkpoints[0] if checkpoints else None


def save_config(config: Dict, filepath: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Saved config to {filepath}")


def load_config(filepath: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config not found: {filepath}")
        
    with open(filepath, 'r') as f:
        config = json.load(f)
        
    logger.info(f"Loaded config from {filepath}")
    
    return config


def main():
    """Example usage."""
    print("="*60)
    print("Checkpoint Management")
    print("="*60)
    
    # Create dummy model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    manager = CheckpointManager(
        model,
        optimizer,
        checkpoint_dir='test_checkpoints',
        max_checkpoints=3
    )
    print(f"\nCheckpoint directory: {manager.checkpoint_dir}")
    
    # Save checkpoint
    metrics = {'loss': 0.5, 'accuracy': 0.85}
    checkpoint_path = manager.save_checkpoint(
        epoch=1,
        metrics=metrics
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best checkpoint
    is_best = manager.save_best_checkpoint(
        epoch=1,
        metrics=metrics,
        metric_name='loss',
        mode='min'
    )
    print(f"Is best model: {is_best}")
    
    # Save config
    config = {
        'model': 'NEST_RNN_T',
        'learning_rate': 0.001,
        'batch_size': 32
    }
    save_config(config, 'test_checkpoints/config.json')
    print("Saved configuration")
    
    # Load config
    loaded_config = load_config('test_checkpoints/config.json')
    print(f"Loaded config: {loaded_config}")
    
    # Cleanup
    import shutil
    shutil.rmtree('test_checkpoints')
    print("\nCleaned up test checkpoints")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
