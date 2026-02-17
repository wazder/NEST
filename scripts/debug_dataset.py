import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.zuco_dataset import ZuCoTorchDataset

def debug_dataset():
    print("Initializing dataset...")
    try:
        dataset = ZuCoTorchDataset(
            root_dir="ZuCo_Dataset/ZuCo",
            task="task1-SR",
            max_samples=None
        )
    except Exception as e:
        print(f"Failed to init dataset: {e}")
        # Try with underscore
        dataset = ZuCoTorchDataset(
            root_dir="ZuCo_Dataset/ZuCo",
            task="task1_SR",
            max_samples=None
        )

    print(f"Dataset size: {len(dataset)}")
    
    print("Iterating through samples...")
    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]
            if sample['eeg'] is None or sample['eeg'].size == 0:
                print(f"\n❌ FAIL at index {i}: EEG is empty/None")
        except Exception as e:
            print(f"\n❌ ERROR at index {i}: {e}")
            # Get the file info if possible
            file_idx, sent_idx = dataset.indices[i]
            print(f"   File index: {file_idx}, Sentence index: {sent_idx}")
            print(f"   File path: {dataset.files[file_idx]}")
            # Determine which sentence
            break

if __name__ == "__main__":
    debug_dataset()
