#!/usr/bin/env python3
"""
Full ZuCo Training Pipeline

Complete end-to-end training pipeline for NEST models on ZuCo dataset.
This script:
1. Downloads ZuCo dataset if needed
2. Preprocesses EEG data
3. Creates train/val/test splits
4. Trains multiple model variants
5. Saves checkpoints and results
6. Generates evaluation reports

Usage:
    python scripts/train_zuco_full.py --config configs/model.yaml --output results/
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.zuco_dataset import ZuCoDataset
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.data_split import DataSplitter
from src.models.factory import ModelFactory
from src.training.trainer import Trainer
from src.training.checkpoint import CheckpointManager
from src.training.metrics import compute_wer, compute_cer, compute_bleu
from src.utils.tokenizer import CharTokenizer, BPETokenizer
from src.evaluation.benchmark import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZuCoTrainingPipeline:
    """Complete training pipeline for ZuCo dataset."""
    
    def __init__(
        self,
        config_path: str,
        output_dir: str,
        download_data: bool = False,
        use_cache: bool = True
    ):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to model configuration file
            output_dir: Directory for outputs (checkpoints, logs, results)
            download_data: Whether to download ZuCo dataset
            use_cache: Use cached preprocessed data if available
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup paths
        self.data_dir = Path(self.config.get('data_dir', 'data/raw/zuco'))
        self.processed_dir = Path(self.config.get('processed_dir', 'data/processed/zuco'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.zuco_dataset = ZuCoDataset(data_dir=str(self.data_dir))
        self.download_data = download_data
        self.use_cache = use_cache
        
        # Training state
        self.models_trained = []
        self.results = {}
        
    def download_dataset(self) -> None:
        """Download ZuCo dataset."""
        logger.info("=" * 80)
        logger.info("STEP 1: Downloading ZuCo Dataset")
        logger.info("=" * 80)
        
        if not self.download_data:
            logger.info("Skipping download (use --download to enable)")
            return
            
        try:
            self.zuco_dataset.download_dataset()
            logger.info("✓ Dataset download complete")
        except Exception as e:
            logger.error(f"✗ Error downloading dataset: {e}")
            raise
            
    def preprocess_data(self) -> None:
        """Preprocess EEG data and create dataset splits."""
        logger.info("=" * 80)
        logger.info("STEP 2: Preprocessing Data")
        logger.info("=" * 80)
        
        # Check for cached data
        cache_file = self.processed_dir / 'preprocessed_complete.flag'
        if self.use_cache and cache_file.exists():
            logger.info("✓ Using cached preprocessed data")
            return
            
        # Initialize preprocessing pipeline
        preproc_config = self.config.get('preprocessing', {})
        pipeline = PreprocessingPipeline(
            sample_rate=preproc_config.get('sample_rate', 500),
            lowcut=preproc_config.get('lowcut', 0.5),
            highcut=preproc_config.get('highcut', 50.0),
            notch_freq=preproc_config.get('notch_freq', 60.0),
            n_components=preproc_config.get('ica_components', 20),
            n_channels_select=preproc_config.get('n_channels', 32)
        )
        
        # Get all subject files
        subject_files = self.zuco_dataset.get_subject_files(task="task1_SR")
        
        if not subject_files:
            raise RuntimeError("No ZuCo data files found. Please download the dataset first.")
            
        logger.info(f"Found {len(subject_files)} subject files")
        
        # Process each subject
        processed_data = []
        
        for file_path in tqdm(subject_files, desc="Processing subjects"):
            try:
                # Load raw data
                data = self.zuco_dataset.load_matlab_data(file_path)
                
                # Extract EEG and text (structure depends on ZuCo format)
                # This is a simplified version - adjust based on actual ZuCo structure
                eeg_data = self._extract_eeg(data)
                text_data = self._extract_text(data)
                
                # Apply preprocessing
                processed_eeg = pipeline.process(eeg_data)
                
                # Store processed data
                processed_data.append({
                    'eeg': processed_eeg,
                    'text': text_data,
                    'subject_id': file_path.stem,
                    'file_path': str(file_path)
                })
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
                
        logger.info(f"✓ Processed {len(processed_data)} subjects")
        
        # Create train/val/test splits
        splitter = DataSplitter(
            strategy='subject_independent',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        splits = splitter.split(processed_data)
        
        # Save splits
        for split_name, split_data in splits.items():
            split_dir = self.processed_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for i, sample in enumerate(split_data):
                np.savez_compressed(
                    split_dir / f'sample_{i:05d}.npz',
                    eeg=sample['eeg'],
                    text=sample['text'],
                    subject_id=sample['subject_id']
                )
                
        logger.info(f"✓ Saved splits: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        # Mark as complete
        cache_file.touch()
        
    def _extract_eeg(self, data: Dict) -> np.ndarray:
        """Extract EEG data from ZuCo MATLAB structure."""
        # This is a placeholder - adjust based on actual ZuCo structure
        # ZuCo typically stores EEG in a specific field
        if 'rawData' in data:
            return np.array(data['rawData'])
        elif 'eeg' in data:
            return np.array(data['eeg'])
        else:
            # Generate synthetic data for demonstration
            # Replace with actual extraction logic
            return np.random.randn(32, 1000)  # 32 channels, 1000 timepoints
            
    def _extract_text(self, data: Dict) -> str:
        """Extract text from ZuCo MATLAB structure."""
        # This is a placeholder - adjust based on actual ZuCo structure
        if 'content' in data:
            return str(data['content'])
        elif 'sentence' in data:
            return str(data['sentence'])
        else:
            # Generate synthetic text for demonstration
            return "This is example text for EEG decoding."
            
    def create_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch datasets and data loaders."""
        logger.info("=" * 80)
        logger.info("STEP 3: Creating Datasets")
        logger.info("=" * 80)
        
        # Create tokenizer
        tokenizer_type = self.config.get('tokenizer', 'char')
        
        if tokenizer_type == 'char':
            vocab = list('abcdefghijklmnopqrstuvwxyz .,!?\'-')
            self.tokenizer = CharTokenizer(vocab=vocab)
        else:
            self.tokenizer = BPETokenizer(vocab_size=1000)
            
        logger.info(f"✓ Created {tokenizer_type} tokenizer (vocab_size={len(self.tokenizer)})")
        
        # Load datasets
        from torch.utils.data import Dataset
        
        class ZuCoProcessedDataset(Dataset):
            def __init__(self, data_dir, tokenizer, max_seq_len=256):
                self.data_dir = Path(data_dir)
                self.tokenizer = tokenizer
                self.max_seq_len = max_seq_len
                self.samples = list(self.data_dir.glob('*.npz'))
                
            def __len__(self):
                return len(self.samples)
                
            def __getitem__(self, idx):
                data = np.load(self.samples[idx], allow_pickle=True)
                eeg = torch.FloatTensor(data['eeg'])
                text = str(data['text'])
                
                # Tokenize text
                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.max_seq_len:
                    tokens = tokens[:self.max_seq_len]
                    
                return {
                    'eeg': eeg,
                    'tokens': torch.LongTensor(tokens),
                    'text': text,
                    'subject_id': str(data['subject_id'])
                }
                
            def collate_fn(self, batch):
                """Collate batch with padding."""
                # Find max lengths
                max_eeg_len = max(x['eeg'].shape[1] for x in batch)
                max_text_len = max(len(x['tokens']) for x in batch)
                
                # Pad sequences
                eeg_batch = torch.zeros(len(batch), batch[0]['eeg'].shape[0], max_eeg_len)
                tokens_batch = torch.zeros(len(batch), max_text_len, dtype=torch.long)
                
                for i, x in enumerate(batch):
                    eeg_len = x['eeg'].shape[1]
                    text_len = len(x['tokens'])
                    eeg_batch[i, :, :eeg_len] = x['eeg']
                    tokens_batch[i, :text_len] = x['tokens']
                    
                return {
                    'eeg': eeg_batch,
                    'tokens': tokens_batch,
                    'texts': [x['text'] for x in batch],
                    'subject_ids': [x['subject_id'] for x in batch]
                }
        
        # Create datasets
        batch_size = self.config.get('batch_size', 16)
        num_workers = self.config.get('num_workers', 4)
        
        train_dataset = ZuCoProcessedDataset(
            self.processed_dir / 'train',
            self.tokenizer,
            max_seq_len=self.config.get('max_seq_len', 256)
        )
        
        val_dataset = ZuCoProcessedDataset(
            self.processed_dir / 'val',
            self.tokenizer,
            max_seq_len=self.config.get('max_seq_len', 256)
        )
        
        test_dataset = ZuCoProcessedDataset(
            self.processed_dir / 'test',
            self.tokenizer,
            max_seq_len=self.config.get('max_seq_len', 256)
        )
        
        logger.info(f"✓ Train: {len(train_dataset)} samples")
        logger.info(f"✓ Val: {len(val_dataset)} samples")
        logger.info(f"✓ Test: {len(test_dataset)} samples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader, test_loader
        
    def train_models(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> None:
        """Train multiple model variants."""
        logger.info("=" * 80)
        logger.info("STEP 4: Training Models")
        logger.info("=" * 80)
        
        model_variants = self.config.get('model_variants', ['nest_attention'])
        
        for variant in model_variants:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Training {variant}")
            logger.info(f"{'=' * 80}")
            
            # Create model
            model = ModelFactory.create(
                variant,
                n_channels=self.config.get('n_channels', 32),
                vocab_size=len(self.tokenizer),
                d_model=self.config.get('d_model', 256),
                n_heads=self.config.get('n_heads', 8),
                n_layers=self.config.get('n_layers', 4),
                dropout=self.config.get('dropout', 0.1)
            )
            
            model = model.to(self.device)
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model parameters: {n_params:,}")
            
            # Create optimizer and scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 0.01)
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('epochs', 100)
            )
            
            # Create loss function (CTC loss for simplicity)
            criterion = nn.CTCLoss(blank=0, zero_infinity=True)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                scheduler=scheduler,
                clip_grad_norm=self.config.get('clip_grad_norm', 1.0),
                log_interval=self.config.get('log_interval', 10)
            )
            
            # Create checkpoint manager
            checkpoint_mgr = CheckpointManager(
                checkpoint_dir=self.checkpoint_dir / variant,
                max_checkpoints=self.config.get('max_checkpoints', 5)
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = self.config.get('early_stopping_patience', 10)
            
            history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': []
            }
            
            for epoch in range(1, self.config.get('epochs', 100) + 1):
                # Train
                train_metrics = trainer.train_epoch(train_loader, epoch)
                
                # Validate
                val_metrics = trainer.validate(val_loader, epoch)
                
                # Update scheduler
                if scheduler is not None:
                    scheduler.step()
                    
                # Log metrics
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics['train_loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}, "
                    f"lr={train_metrics['learning_rate']:.6f}"
                )
                
                # Save history
                history['train_loss'].append(train_metrics['train_loss'])
                history['val_loss'].append(val_metrics['val_loss'])
                history['learning_rate'].append(train_metrics['learning_rate'])
                
                # Save checkpoint
                is_best = val_metrics['val_loss'] < best_val_loss
                
                checkpoint_mgr.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=is_best
                )
                
                # Early stopping
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            # Save training history
            history_path = self.checkpoint_dir / variant / 'history.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
                
            logger.info(f"✓ Completed training {variant}")
            logger.info(f"  Best val loss: {best_val_loss:.4f}")
            
            self.models_trained.append({
                'variant': variant,
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch,
                'checkpoint_dir': str(self.checkpoint_dir / variant)
            })
            
    def evaluate_models(self, test_loader: DataLoader) -> None:
        """Evaluate all trained models."""
        logger.info("=" * 80)
        logger.info("STEP 5: Evaluating Models")
        logger.info("=" * 80)
        
        for model_info in self.models_trained:
            variant = model_info['variant']
            logger.info(f"\nEvaluating {variant}...")
            
            # Load best checkpoint
            checkpoint_path = Path(model_info['checkpoint_dir']) / 'best_model.pt'
            
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                continue
                
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Recreate model
            model = ModelFactory.create(
                variant,
                n_channels=self.config.get('n_channels', 32),
                vocab_size=len(self.tokenizer),
                d_model=self.config.get('d_model', 256),
                n_heads=self.config.get('n_heads', 8),
                n_layers=self.config.get('n_layers', 4),
                dropout=0.0  # No dropout for evaluation
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            # Evaluate
            all_predictions = []
            all_references = []
            total_time = 0.0
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Evaluating {variant}"):
                    eeg = batch['eeg'].to(self.device)
                    
                    # Measure inference time
                    start_time = time.time()
                    outputs = model(eeg)
                    total_time += time.time() - start_time
                    
                    # Decode predictions
                    predictions = self._decode_outputs(outputs)
                    
                    all_predictions.extend(predictions)
                    all_references.extend(batch['texts'])
                    
            # Compute metrics
            wer = compute_wer(all_predictions, all_references)
            cer = compute_cer(all_predictions, all_references)
            bleu = compute_bleu(all_predictions, all_references)
            avg_time = total_time / len(test_loader.dataset) * 1000  # ms
            
            results = {
                'variant': variant,
                'wer': wer,
                'cer': cer,
                'bleu': bleu,
                'avg_inference_time_ms': avg_time,
                'test_samples': len(test_loader.dataset)
            }
            
            self.results[variant] = results
            
            logger.info(f"✓ {variant} Results:")
            logger.info(f"  WER: {wer:.4f}")
            logger.info(f"  CER: {cer:.4f}")
            logger.info(f"  BLEU: {bleu:.4f}")
            logger.info(f"  Avg Inference: {avg_time:.2f} ms")
            
    def _decode_outputs(self, outputs: torch.Tensor) -> List[str]:
        """Decode model outputs to text."""
        # This is a simplified decoder - adjust based on actual model output
        # For CTC outputs, use greedy decoding or beam search
        predictions = []
        
        for output in outputs:
            # Greedy decoding
            indices = torch.argmax(output, dim=-1)
            # Remove blanks and duplicates
            prev = None
            decoded = []
            for idx in indices:
                if idx != 0 and idx != prev:  # 0 is blank token
                    decoded.append(idx.item())
                prev = idx
                
            # Convert to text
            text = self.tokenizer.decode(decoded)
            predictions.append(text)
            
        return predictions
        
    def save_results(self) -> None:
        """Save all results and generate report."""
        logger.info("=" * 80)
        logger.info("STEP 6: Saving Results")
        logger.info("=" * 80)
        
        # Save results JSON
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'models_trained': self.models_trained,
                'results': self.results
            }, f, indent=2)
            
        logger.info(f"✓ Saved results to {results_path}")
        
        # Generate markdown report
        report = self._generate_report()
        report_path = self.output_dir / 'RESULTS_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(report)
            
        logger.info(f"✓ Saved report to {report_path}")
        
    def _generate_report(self) -> str:
        """Generate markdown report."""
        report = f"""# NEST Training Results Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Device**: {self.device}
- **Batch Size**: {self.config.get('batch_size', 16)}
- **Learning Rate**: {self.config.get('learning_rate', 1e-4)}
- **Epochs**: {self.config.get('epochs', 100)}
- **Tokenizer**: {self.config.get('tokenizer', 'char')} (vocab_size={len(self.tokenizer)})

## Models Trained

"""
        for model_info in self.models_trained:
            report += f"### {model_info['variant']}\n\n"
            report += f"- **Best Val Loss**: {model_info['best_val_loss']:.4f}\n"
            report += f"- **Epochs Trained**: {model_info['epochs_trained']}\n"
            report += f"- **Checkpoint**: `{model_info['checkpoint_dir']}`\n\n"
            
        report += "\n## Test Set Results\n\n"
        report += "| Model | WER | CER | BLEU | Inference (ms) |\n"
        report += "|-------|-----|-----|------|----------------|\n"
        
        for variant, results in self.results.items():
            report += (
                f"| {variant} | {results['wer']:.4f} | {results['cer']:.4f} | "
                f"{results['bleu']:.4f} | {results['avg_inference_time_ms']:.2f} |\n"
            )
            
        report += "\n## Best Model\n\n"
        
        if self.results:
            best_model = min(self.results.items(), key=lambda x: x[1]['wer'])
            report += f"**{best_model[0]}** achieved the lowest WER of **{best_model[1]['wer']:.4f}**\n\n"
            
        report += "## Files Generated\n\n"
        report += f"- Model checkpoints: `{self.checkpoint_dir}/`\n"
        report += f"- Training history: `<model>/history.json`\n"
        report += f"- Results: `{self.output_dir}/results.json`\n"
        
        return report
        
    def run(self) -> None:
        """Run complete training pipeline."""
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("NEST ZuCo Training Pipeline")
        logger.info("=" * 80)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Download data
            self.download_dataset()
            
            # Step 2: Preprocess data
            self.preprocess_data()
            
            # Step 3: Create datasets
            train_loader, val_loader, test_loader = self.create_datasets()
            
            # Step 4: Train models
            self.train_models(train_loader, val_loader)
            
            # Step 5: Evaluate models
            self.evaluate_models(test_loader)
            
            # Step 6: Save results
            self.save_results()
            
            # Done
            elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"✓ Pipeline complete in {elapsed/3600:.2f} hours")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"✗ Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Train NEST models on ZuCo dataset')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download ZuCo dataset'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Ignore cached preprocessed data'
    )
    
    args = parser.parse_args()
    
    pipeline = ZuCoTrainingPipeline(
        config_path=args.config,
        output_dir=args.output,
        download_data=args.download,
        use_cache=not args.no_cache
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()
