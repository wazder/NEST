"""
ZuCo Dataset Downloader and Handler

The ZuCo (Zurich Cognitive Language Processing Corpus) dataset contains
EEG and eye-tracking data recorded while subjects read English text.

References:
- ZuCo 1.0: Hollenstein et al. (2018)
- ZuCo 2.0: Hollenstein et al. (2020)
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from urllib.request import urlretrieve
import zipfile
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZuCoDataset:
    """Handler for ZuCo EEG dataset."""
    
    # ZuCo dataset URLs (public OSF repository)
    ZUCO_URLS = {
        "task1_SR": "https://osf.io/q3zws/download",  # Sentence Reading
        "task2_NR": "https://osf.io/sjezn/download",  # Normal Reading
        "task3_TSR": "https://osf.io/2urht/download",  # Task-specific Reading
    }
    
    def __init__(self, data_dir: str = "data/raw/zuco"):
        """
        Initialize ZuCo dataset handler.
        
        Args:
            data_dir: Directory to store raw ZuCo data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, tasks: Optional[List[str]] = None) -> None:
        """
        Download ZuCo dataset files.
        
        Args:
            tasks: List of tasks to download. If None, downloads all tasks.
                  Options: ["task1_SR", "task2_NR", "task3_TSR"]
        """
        if tasks is None:
            tasks = list(self.ZUCO_URLS.keys())
            
        logger.info(f"Downloading ZuCo dataset tasks: {tasks}")
        
        for task in tasks:
            if task not in self.ZUCO_URLS:
                logger.warning(f"Unknown task: {task}. Skipping.")
                continue
                
            url = self.ZUCO_URLS[task]
            output_path = self.data_dir / f"{task}.zip"
            
            if output_path.exists():
                logger.info(f"Task {task} already downloaded. Skipping.")
                continue
                
            logger.info(f"Downloading {task} from {url}")
            try:
                urlretrieve(url, output_path, reporthook=self._download_progress)
                logger.info(f"Successfully downloaded {task}")
                
                # Extract the zip file
                self._extract_zip(output_path, self.data_dir / task)
                
            except Exception as e:
                logger.error(f"Error downloading {task}: {e}")
                
    def _download_progress(self, block_num, block_size, total_size):
        """Progress bar for download."""
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f'\rDownload progress: {percent:.1f}%', end='')
        
    def _extract_zip(self, zip_path: Path, extract_to: Path) -> None:
        """Extract zip file."""
        logger.info(f"Extracting {zip_path.name}")
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
        logger.info(f"Extracted to {extract_to}")
        
    def load_matlab_data(self, file_path: Path) -> Dict:
        """
        Load MATLAB .mat file from ZuCo dataset.
        
        Args:
            file_path: Path to .mat file
            
        Returns:
            Dictionary containing EEG data and metadata
        """
        mat_data = scipy.io.loadmat(str(file_path))
        return mat_data
        
    def get_subject_files(self, task: str = "task1_SR") -> List[Path]:
        """
        Get all subject data files for a given task.
        
        Args:
            task: Task name (e.g., "task1_SR")
            
        Returns:
            List of paths to subject data files
        """
        task_dir = self.data_dir / task
        if not task_dir.exists():
            logger.warning(f"Task directory {task_dir} does not exist.")
            return []
            
        # ZuCo data is typically stored as .mat files
        mat_files = list(task_dir.rglob("*.mat"))
        logger.info(f"Found {len(mat_files)} subject files for {task}")
        
        return mat_files
        
    def explore_dataset(self, task: str = "task1_SR") -> pd.DataFrame:
        """
        Perform exploratory data analysis on ZuCo dataset.
        
        Args:
            task: Task name to explore
            
        Returns:
            DataFrame with dataset statistics
        """
        subject_files = self.get_subject_files(task)
        
        if not subject_files:
            logger.warning("No subject files found for exploration.")
            return pd.DataFrame()
            
        exploration_data = []
        
        for file_path in tqdm(subject_files[:5], desc="Exploring dataset"):  # Limit to 5 for quick exploration
            try:
                data = self.load_matlab_data(file_path)
                
                # Extract basic information
                info = {
                    'file': file_path.name,
                    'task': task,
                }
                
                # Try to extract common fields (ZuCo structure may vary)
                for key in data.keys():
                    if not key.startswith('__'):
                        info[f'{key}_shape'] = str(np.array(data[key]).shape)
                        
                exploration_data.append(info)
                
            except Exception as e:
                logger.error(f"Error exploring {file_path}: {e}")
                
        df = pd.DataFrame(exploration_data)
        return df
        
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about downloaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'data_directory': str(self.data_dir),
            'downloaded_tasks': [],
            'total_files': 0
        }
        
        for task in self.ZUCO_URLS.keys():
            task_dir = self.data_dir / task
            if task_dir.exists():
                files = list(task_dir.rglob("*.mat"))
                info['downloaded_tasks'].append({
                    'task': task,
                    'num_files': len(files),
                    'path': str(task_dir)
                })
                info['total_files'] += len(files)
                
        return info


def main():
    """Example usage of ZuCoDataset."""
    # Initialize dataset handler
    dataset = ZuCoDataset(data_dir="data/raw/zuco")
    
    # Download dataset (this will take time and bandwidth!)
    # Uncomment to actually download:
    # dataset.download_dataset()
    
    # Get dataset info
    info = dataset.get_dataset_info()
    logger.info(f"Dataset info: {info}")
    
    # Explore dataset
    if info['total_files'] > 0:
        exploration = dataset.explore_dataset()
        print("\nDataset Exploration:")
        print(exploration)
    else:
        logger.info("No dataset files found. Please download the dataset first.")


if __name__ == "__main__":
    main()


import torch
from torch.utils.data import Dataset

class ZuCoTorchDataset(Dataset):
    """
    PyTorch Dataset for ZuCo data.
    
    Implements lazy loading and proper preprocessing integration.
    """
    
    def __init__(
        self, 
        root_dir: str, 
        task: str = "task1_SR", 
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing task folders
            task: Task name (e.g., 'task1_SR')
            transform: Optional transform to apply to samples
            max_samples: Limit number of samples (for testing)
        """
        self.root_dir = Path(root_dir)
        self.task = task
        self.transform = transform
        
        # Get list of files
        self.files = list((self.root_dir / task / "Matlab_files").glob("*.mat"))
        
        if not self.files:
            # Try alternative path structure
            self.files = list((self.root_dir / task).glob("*.mat"))
            
        if not self.files:
            logger.warning(f"No .mat files found in {self.root_dir}/{task}")
            
        if max_samples:
            self.files = self.files[:max_samples]
            
        # Build index of (file_idx, sentence_idx) to allow random access
        # minimal loading to get counts
        self.indices = []
        self._build_index()
        
    def _build_index(self):
        """Build index of all samples without loading full data."""
        logger.info(f"Indexing {len(self.files)} files...")
        for file_idx, fpath in enumerate(self.files):
            try:
                # Load only headers/structure if possible, 
                # but scipy.io.loadmat loads everything. 
                # For optimization, we might cache this index.
                # For now, we load and count.
                mat = scipy.io.loadmat(str(fpath), variable_names=['sentenceData'], simplify_cells=True)
                if 'sentenceData' in mat:
                    data = mat['sentenceData']
                    count = len(data) if isinstance(data, (list, np.ndarray)) else 1
                    for i in range(count):
                        self.indices.append((file_idx, i))
            except Exception as e:
                logger.warning(f"Failed to index {fpath.name}: {e}")
                
        logger.info(f"Indexed {len(self.indices)} total samples")

    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        file_idx, sample_idx = self.indices[idx]
        fpath = self.files[file_idx]
        
        # Load specific file
        try:
            mat = scipy.io.loadmat(str(fpath), simplify_cells=True)
            sent_data = mat['sentenceData']
            
            # Handle different structures
            if isinstance(sent_data, dict):
                 sample = sent_data
            else:
                 sample = sent_data[sample_idx]
                 
            # Extract content
            eeg = sample['rawData']
            text = sample['content']
            
            # Basic validation
            if eeg is None or not isinstance(eeg, np.ndarray):
                raise ValueError("Invalid EEG data")
                
            # Convert to float32
            eeg = eeg.astype(np.float32)
            
            data_dict = {
                'eeg': eeg, 
                'text': text,
                'file': fpath.name
            }
            
            if self.transform:
                data_dict = self.transform(data_dict)
                
            return data_dict
            
        except Exception as e:
            # Return None or handle error
            logger.error(f"Error loading sample {idx}: {e}")
            raise e

