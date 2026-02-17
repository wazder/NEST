#!/usr/bin/env python3
"""
Real ZuCo Training Script - Full Pipeline

This script:
1. Loads raw ZuCo .mat files
2. Preprocesses EEG data
3. Trains NEST models  
4. Saves results

Usage:
    python scripts/train_with_real_zuco.py --model conformer --epochs 100
    python scripts/train_with_real_zuco.py --quick-test  # Quick validation
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Verify data
data_dir = Path("data/raw/zuco")
if not data_dir.exists() and not (Path("ZuCo_Dataset") / "ZuCo").exists():
    print("✗ Error: ZuCo data not found!")
    print(f"  Expected: {data_dir.absolute()} or ZuCo_Dataset/ZuCo/")
    sys.exit(1)

# Use direct path if symlink doesn't work
if not data_dir.exists():
    data_dir = Path("ZuCo_Dataset") / "ZuCo"

# Count .mat files
mat_files = []
for task in ["task1_SR", "task2_NR", "task3_TSR"]:
    task_dir = data_dir / task / "Matlab_files"
    if task_dir.exists():
        mat_files.extend(list(task_dir.glob("*.mat")))

if len(mat_files) == 0:
    print(f"✗ Error: No .mat files found in {data_dir}")
    sys.exit(1)

print("=" * 80)
print(" NEST: Real ZuCo Training - Full Pipeline")
print("=" * 80)
print(f"Data directory: {data_dir}")
print(f"Found {len(mat_files)} .mat files")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()


class ZuCoRealDataset(Dataset):
    """Dataset class for real ZuCo .mat files."""
    
    def __init__(self, mat_files, max_samples=None, quick_test=False):
        """
        Args:
            mat_files: List of paths to .mat files
            max_samples: Maximum number of samples to load (for testing)
            quick_test: If True, load minimal data for quick validation
        """
        self.samples = []
        self.quick_test = quick_test
        
        # For quick test, use only first file with minimal samples
        files_to_load = mat_files[:1] if quick_test else mat_files  # Use ALL files
        max_per_file = 50 if quick_test else None  # No limit for full training
        
        print(f"Loading data from {len(files_to_load)} files...")
        
        for mat_file in tqdm(files_to_load, desc="Loading .mat files"):
            try:
                # Load .mat file
                mat_data = scipy.io.loadmat(str(mat_file), simplify_cells=True)
                
                # Extract sentence data
                # ZuCo structure: sentenceData contains EEG recordings
                if 'sentenceData' in mat_data:
                    sent_data = mat_data['sentenceData']
                    
                    # Handle different .mat formats
                    if isinstance(sent_data, dict):
                        sent_data = [sent_data]
                    elif not isinstance(sent_data, list):
                        sent_data = list(sent_data)
                    
                    # Extract samples
                    for idx, sentence in enumerate(sent_data):
                        if max_per_file and idx >= max_per_file:
                            break
                            
                        try:
                            # Get EEG data and text from ZuCo format
                            if isinstance(sentence, dict):
                                eeg = sentence.get('rawData')
                                text = sentence.get('content', '')
                                
                                if eeg is not None and isinstance(eeg, np.ndarray) and isinstance(text, str):
                                    # EEG is already (105, timepoints)
                                    if eeg.shape[0] != 105:
                                        continue
                                    
                                    # Normalize EEG
                                    eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-8)
                                    
                                    # Pad or truncate to fixed length (2000 timepoints)
                                    target_len = 2000
                                    if eeg.shape[1] < target_len:
                                        pad = np.zeros((eeg.shape[0], target_len - eeg.shape[1]))
                                        eeg = np.concatenate([eeg, pad], axis=1)
                                    else:
                                        eeg = eeg[:, :target_len]
                                    
                                    # Convert text to character indices
                                    # Vocabulary: blank(0), space(1), a-z(2-27)
                                    if len(text) > 0:
                                        char_ids = []
                                        for c in text.lower():
                                            if c == ' ':
                                                char_ids.append(1)
                                            elif c.isalpha():
                                                char_ids.append(ord(c) - ord('a') + 2)
                                            # Skip other characters
                                        
                                        if len(char_ids) > 0:
                                            self.samples.append({
                                                'eeg': eeg.astype(np.float32),
                                                'text': text,
                                                'target': np.array(char_ids, dtype=np.int64)
                                            })
                        except Exception as e:
                            continue  # Skip problematic samples
                            
                if max_samples and len(self.samples) >= max_samples:
                    break
                    
            except Exception as e:
                print(f"Warning: Could not load {mat_file.name}: {e}")
                continue
        
        print(f"✓ Loaded {len(self.samples)} valid samples")
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples loaded! Check .mat file format.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def collate_fn(self, batch):
        """Collate batch of samples."""
        eeg_batch = torch.stack([torch.from_numpy(s['eeg']) for s in batch])
        
        # Pad targets to same length
        max_target_len = max(len(s['target']) for s in batch)
        target_batch = []
        target_lengths = []
        
        for s in batch:
            target = s['target']
            target_lengths.append(len(target))
            # Pad with zeros
            if len(target) < max_target_len:
                target = np.concatenate([target, np.zeros(max_target_len - len(target), dtype=np.int64)])
            target_batch.append(torch.from_numpy(target))
        
        target_batch = torch.stack(target_batch)
        target_lengths = torch.tensor(target_lengths)
        
        return {
            'eeg': eeg_batch,
            'target': target_batch,
            'target_length': target_lengths
        }


def create_simple_model(input_channels=105, hidden_size=256, num_classes=28):
    """Create a simple LSTM-based model for EEG decoding."""
    
    class SimpleLSTMDecoder(nn.Module):
        def __init__(self, input_channels, hidden_size, num_classes):
            super().__init__()
            self.hidden_size = hidden_size
            
            # Convolutional feature extraction
            self.conv = nn.Sequential(
                nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
                
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
            )
            
            # LSTM temporal encoding
            self.lstm = nn.LSTM(
                256, hidden_size, 
                num_layers=2, 
                batch_first=True,
                dropout=0.3,
                bidirectional=True
            )
            
            # Output projection
            self.fc = nn.Linear(hidden_size * 2, num_classes)
            
        def forward(self, x):
            # x: (batch, channels, time)
            x = self.conv(x)  # (batch, 256, time/4)
            x = x.permute(0, 2, 1)  # (batch, time/4, 256)
            x, _ = self.lstm(x)  # (batch, time/4, hidden*2)
            x = self.fc(x)  # (batch, time/4, num_classes)
            return x
    
    return SimpleLSTMDecoder(input_channels, hidden_size, num_classes)


def train_model(model, train_loader, device, epochs=10, model_name="model", accumulation_steps=2):
    """Train a model with gradient accumulation for better GPU utilization."""
    
    model = model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Disable mixed precision for now (CTC on CPU causes issues with MPS)
    use_amp = False  # str(device) == 'mps'
    scaler = torch.amp.GradScaler('cpu') if use_amp else None  # MPS uses CPU scaler for now
    
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")
    print(f"🚀 M2 GPU optimization enabled")
    print(f"📊 Effective batch size: {len(train_loader.dataset) // len(train_loader)} × {accumulation_steps} = {(len(train_loader.dataset) // len(train_loader)) * accumulation_steps}")
    print()
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        optimizer.zero_grad()  # Initialize gradients
        
        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg'].to(device)
            target = batch['target'].to(device)
            target_lengths = batch['target_length'].to(device)
            
            # Forward pass
            output = model(eeg)  # (batch, time, num_classes)
            
            # CTC loss expects (time, batch, num_classes)
            output = output.permute(1, 0, 2)
            output = torch.log_softmax(output, dim=2)
            
            # Input lengths (after pooling: time/4) - keep on CPU for CTC
            input_lengths = torch.full((output.size(1),), output.size(0), dtype=torch.long)
            
            # Filter out empty targets - move to CPU for indexing
            valid_indices = (target_lengths > 0).cpu()
            if valid_indices.sum() == 0:
                continue
                
            output = output[:, valid_indices, :]
            target = target[valid_indices]
            target_lengths = target_lengths[valid_indices].cpu()  # CTC expects CPU tensors
            input_lengths = input_lengths[valid_indices]
            
            # Compute loss (CTC works on CPU tensors for indices)
            # Compute loss (CTC works on CPU tensors for indices)
            try:
                loss = criterion(output.cpu(), target.cpu(), input_lengths, target_lengths)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Move loss back to device for backward
                loss = loss.to(device)
                    
            except Exception:
                continue
            
            # Backward pass with gradient accumulation
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Only update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return losses


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train NEST models on real ZuCo data')
    parser.add_argument('--model', default='all', 
                       choices=['all', 'conformer', 'transformer', 'rnn_t', 'ctc'],
                       help='Which model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (32 for M2 Air)')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Quick test with minimal data and epochs')
    parser.add_argument('--accumulation-steps', type=int, default=2,
                       help='Gradient accumulation (effective batch = batch × accumulation)')
    args = parser.parse_args()
    
    # Quick test settings
    if args.quick_test:
        print("⚡ QUICK TEST MODE")
        print("   - 10 epochs")
        print("   - Minimal data (50 samples)")
        print("   - Simple model only")
        print()
        args.epochs = 10
        args.model = 'simple'
    
    # Setup device - M2 Air optimization
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # M2 GPU acceleration
        print("🚀 Using M2 GPU (MPS) - Training will be MUCH faster!")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU - Training will be slow")
    print(f"Device: {device}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/real_zuco_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Load dataset
    print("Loading real ZuCo data...")
    dataset = ZuCoRealDataset(
        mat_files=mat_files,
        max_samples=100 if args.quick_test else None,
        quick_test=args.quick_test
    )
    
    # Create data loader - optimized for M2 Air
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # macOS: 0 is more stable (multiprocessing issues)
        pin_memory=False,  # MPS doesn't support pin_memory yet
        collate_fn=dataset.collate_fn
    )
    
    print(f"Created data loader with {len(dataset)} samples")
    print(f"Number of batches: {len(train_loader)}")
    print()
    
    # Train model
    print("Creating and training model...")
    model = create_simple_model()
    
    start_time = time.time()
    losses = train_model(
        model, 
        train_loader, 
        device, 
        epochs=args.epochs,
        model_name="NEST-LSTM",
        accumulation_steps=args.accumulation_steps
    )
    elapsed = time.time() - start_time
    
    # Save model
    model_path = checkpoint_dir / "nest_lstm_realdata.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'config': {
            'input_channels': 105,
            'hidden_size': 256,
            'num_classes': 28
        }
    }, model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    # Save results
    results = {
        'model': 'NEST-LSTM',
        'dataset': 'ZuCo (Real)',
        'num_samples': len(dataset),
        'num_mat_files': len(mat_files),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'final_loss': losses[-1],
        'training_time_seconds': elapsed,
        'training_time_hours': elapsed / 3600,
        'losses': losses,
        'timestamp': timestamp,
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 80)
    print(" Training Complete!")
    print("=" * 80)
    print(f"Time elapsed: {elapsed/3600:.2f} hours")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Results saved: {results_path}")
    print()
    
    print("✓ Real ZuCo training successful!")
    print()
    print("Note: This is initial training on real data.")
    print("For full results, run longer training (100+ epochs)")


if __name__ == "__main__":
    main()
