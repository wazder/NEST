"""
Main EEG Preprocessing Pipeline

This script orchestrates the complete preprocessing workflow:
1. Dataset loading
2. Band-pass filtering (0.5-50 Hz)
3. ICA artifact removal
4. Electrode selection
5. Data augmentation
6. Train/val/test splitting
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional, List
import logging
from tqdm import tqdm

from src.data.zuco_dataset import ZuCoDataset
from src.preprocessing.filtering import EEGFilter
from src.preprocessing.artifact_removal import ICArtifactRemoval
from src.preprocessing.electrode_selection import ElectrodeSelector
from src.preprocessing.augmentation import EEGAugmentation
from src.preprocessing.data_split import EEGDataSplitter, DatasetOrganizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Complete EEG preprocessing pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.dataset = None
        self.filter = None
        self.ica = None
        self.selector = None
        self.augmenter = None
        self.splitter = None
        
        logger.info("Initialized preprocessing pipeline")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            # Default configuration
            config = {
                'dataset': {
                    'name': 'zuco',
                    'data_dir': 'data/raw/zuco',
                    'tasks': ['task1_SR']
                },
                'filtering': {
                    'l_freq': 0.5,
                    'h_freq': 50.0,
                    'notch_freq': 50.0,
                    'sfreq': 500.0
                },
                'ica': {
                    'n_components': 20,
                    'method': 'fastica',
                    'threshold': 3.0
                },
                'electrode_selection': {
                    'method': 'variance',  # or 'region', 'names'
                    'n_channels': 32
                },
                'augmentation': {
                    'enabled': True,
                    'n_augmentations': 3,
                    'noise_level': 0.05,
                    'scale_range': [0.9, 1.1]
                },
                'splitting': {
                    'method': 'subject_independent',
                    'train_ratio': 0.7,
                    'val_ratio': 0.15,
                    'test_ratio': 0.15
                },
                'output': {
                    'processed_dir': 'data/processed',
                    'save_intermediate': False
                }
            }
            logger.info("Using default configuration")
            
        return config
        
    def load_dataset(self) -> None:
        """Load and prepare dataset."""
        logger.info("Loading dataset...")
        
        dataset_config = self.config['dataset']
        
        if dataset_config['name'] == 'zuco':
            self.dataset = ZuCoDataset(data_dir=dataset_config['data_dir'])
            
            # Get dataset info
            info = self.dataset.get_dataset_info()
            logger.info(f"Dataset info: {info}")
            
        else:
            raise NotImplementedError(
                f"Dataset {dataset_config['name']} not implemented"
            )
            
    def apply_filtering(self, data: np.ndarray) -> np.ndarray:
        """
        Apply band-pass filtering to EEG data.
        
        Args:
            data: Raw EEG data (channels × time_points)
            
        Returns:
            Filtered EEG data
        """
        logger.info("Applying band-pass filtering...")
        
        filter_config = self.config['filtering']
        
        self.filter = EEGFilter(
            l_freq=filter_config['l_freq'],
            h_freq=filter_config['h_freq'],
            notch_freq=filter_config.get('notch_freq'),
            sfreq=filter_config['sfreq']
        )
        
        filtered = self.filter.apply_all_filters(
            data,
            apply_notch=filter_config.get('notch_freq') is not None
        )
        
        return filtered
        
    def apply_ica(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply ICA for artifact removal.
        
        Args:
            data: Filtered EEG data (channels × time_points)
            sfreq: Sampling frequency
            ch_names: Channel names
            
        Returns:
            Cleaned EEG data
        """
        logger.info("Applying ICA artifact removal...")
        
        ica_config = self.config['ica']
        
        self.ica = ICArtifactRemoval(
            n_components=ica_config['n_components'],
            method=ica_config['method']
        )
        
        # Fit ICA
        self.ica.fit(data, sfreq=sfreq, ch_names=ch_names)
        
        # Find artifact components
        eog_components = self.ica.find_bads_eog(
            data, sfreq, ch_names,
            threshold=ica_config['threshold']
        )
        
        # Exclude components
        self.ica.exclude_components(eog_components)
        
        # Apply ICA
        cleaned = self.ica.apply(data, sfreq, ch_names)
        
        return cleaned
        
    def apply_electrode_selection(
        self,
        data: np.ndarray,
        ch_names: List[str]
    ) -> tuple:
        """
        Apply electrode selection.
        
        Args:
            data: EEG data (channels × time_points)
            ch_names: Channel names
            
        Returns:
            Tuple of (selected_data, selected_channel_names)
        """
        logger.info("Applying electrode selection...")
        
        selection_config = self.config['electrode_selection']
        
        self.selector = ElectrodeSelector(ch_names)
        
        method = selection_config['method']
        
        if method == 'variance':
            selected_indices, _ = self.selector.select_by_variance(
                data,
                n_channels=selection_config['n_channels']
            )
        elif method == 'region':
            selected_indices, _ = self.selector.select_by_region(
                region=selection_config['region']
            )
        elif method == 'names':
            selected_indices, _ = self.selector.select_by_names(
                selected_names=selection_config['channel_names']
            )
        else:
            logger.warning(f"Unknown selection method: {method}. Skipping.")
            return data, ch_names
            
        selected_data, selected_names = self.selector.apply_selection(
            data, selected_indices
        )
        
        return selected_data, selected_names
        
    def apply_augmentation(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> tuple:
        """
        Apply data augmentation.
        
        Args:
            data: EEG data (n_samples, channels, time_points)
            labels: Labels (n_samples,)
            
        Returns:
            Tuple of (augmented_data, augmented_labels)
        """
        aug_config = self.config['augmentation']
        
        if not aug_config['enabled']:
            logger.info("Data augmentation disabled")
            return data, labels
            
        logger.info("Applying data augmentation...")
        
        self.augmenter = EEGAugmentation()
        
        augmented_data = []
        augmented_labels = []
        
        for i in tqdm(range(len(data)), desc="Augmenting samples"):
            # Keep original
            augmented_data.append(data[i])
            augmented_labels.append(labels[i])
            
            # Generate augmented samples
            aug_samples = self.augmenter.random_augment(
                data[i],
                n_augmentations=aug_config['n_augmentations'],
                noise={'noise_level': aug_config['noise_level']},
                scaling={'scale_range': tuple(aug_config['scale_range'])}
            )
            
            for aug_sample in aug_samples:
                augmented_data.append(aug_sample)
                augmented_labels.append(labels[i])
                
        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)
        
        logger.info(
            f"Augmentation complete: {len(data)} → {len(augmented_data)} samples"
        )
        
        return augmented_data, augmented_labels
        
    def split_data(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Split data into train/val/test sets.
        
        Args:
            data: EEG data
            labels: Labels
            subject_ids: Subject IDs (required for subject-independent split)
            
        Returns:
            Dictionary with splits
        """
        logger.info("Splitting data...")
        
        split_config = self.config['splitting']
        
        self.splitter = EEGDataSplitter()
        
        method = split_config['method']
        
        if method == 'subject_independent':
            if subject_ids is None:
                raise ValueError(
                    "subject_ids required for subject-independent split"
                )
            splits = self.splitter.subject_independent_split(
                data, labels, subject_ids,
                train_ratio=split_config['train_ratio'],
                val_ratio=split_config['val_ratio'],
                test_ratio=split_config['test_ratio']
            )
        elif method == 'random':
            splits = self.splitter.random_split(
                data, labels,
                train_ratio=split_config['train_ratio'],
                val_ratio=split_config['val_ratio'],
                test_ratio=split_config['test_ratio'],
                stratify=split_config.get('stratify', False)
            )
        elif method == 'temporal':
            splits = self.splitter.temporal_split(
                data, labels,
                train_ratio=split_config['train_ratio'],
                val_ratio=split_config['val_ratio'],
                test_ratio=split_config['test_ratio']
            )
        else:
            raise ValueError(f"Unknown split method: {method}")
            
        # Get statistics
        stats = DatasetOrganizer.get_split_statistics(splits)
        logger.info(f"Split statistics: {stats}")
        
        return splits
        
    def save_processed_data(self, splits: Dict) -> None:
        """
        Save processed data to disk.
        
        Args:
            splits: Dictionary with train/val/test splits
        """
        output_config = self.config['output']
        output_dir = output_config['processed_dir']
        
        logger.info(f"Saving processed data to {output_dir}")
        
        DatasetOrganizer.save_splits(splits, output_dir)
        
        # Save configuration
        config_path = Path(output_dir) / 'preprocessing_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        logger.info(f"Saved configuration to {config_path}")
        
    def run_pipeline(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        sfreq: float,
        ch_names: List[str],
        subject_ids: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run complete preprocessing pipeline.
        
        Args:
            data: Raw EEG data (n_samples, channels, time_points)
            labels: Labels (n_samples,)
            sfreq: Sampling frequency
            ch_names: Channel names
            subject_ids: Subject IDs (optional)
            
        Returns:
            Dictionary with processed splits
        """
        logger.info("="*60)
        logger.info("Starting preprocessing pipeline")
        logger.info("="*60)
        
        processed_samples = []
        processed_labels = []
        
        # Process each sample
        for i in tqdm(range(len(data)), desc="Processing samples"):
            sample = data[i]
            
            # 1. Filtering
            filtered = self.apply_filtering(sample)
            
            # 2. ICA (optional, can be expensive)
            if self.config['ica'].get('enabled', True):
                cleaned = self.apply_ica(filtered, sfreq, ch_names)
            else:
                cleaned = filtered
                
            # 3. Electrode selection
            if self.config['electrode_selection'].get('enabled', True):
                selected, selected_names = self.apply_electrode_selection(
                    cleaned, ch_names
                )
                ch_names = selected_names  # Update for next iteration
            else:
                selected = cleaned
                
            processed_samples.append(selected)
            processed_labels.append(labels[i])
            
        processed_data = np.array(processed_samples)
        processed_labels = np.array(processed_labels)
        
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        # 4. Data augmentation
        augmented_data, augmented_labels = self.apply_augmentation(
            processed_data, processed_labels
        )
        
        # 5. Split data
        splits = self.split_data(augmented_data, augmented_labels, subject_ids)
        
        # 6. Save processed data
        self.save_processed_data(splits)
        
        logger.info("="*60)
        logger.info("Preprocessing pipeline complete")
        logger.info("="*60)
        
        return splits


def main():
    """Example usage of preprocessing pipeline."""
    # Initialize pipeline
    pipeline = PreprocessingPipeline(config_path='configs/preprocessing.yaml')
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 100
    n_channels = 64
    n_timepoints = 1000
    sfreq = 500.0
    
    data = np.random.randn(n_samples, n_channels, n_timepoints) * 10e-6
    labels = np.random.randint(0, 2, n_samples)
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    subject_ids = np.repeat(np.arange(10), 10)  # 10 subjects, 10 samples each
    
    # Run pipeline
    splits = pipeline.run_pipeline(
        data, labels, sfreq, ch_names, subject_ids
    )
    
    logger.info("Pipeline execution complete!")
    logger.info(f"Train samples: {splits['train'][0].shape[0]}")
    logger.info(f"Val samples: {splits['val'][0].shape[0]}")
    logger.info(f"Test samples: {splits['test'][0].shape[0]}")


if __name__ == "__main__":
    main()
