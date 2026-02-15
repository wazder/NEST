"""
Data Splitting for EEG Datasets

Implements train/validation/test splitting strategies with
subject-independent evaluation protocol for EEG data.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGDataSplitter:
    """Data splitting utilities for EEG datasets."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize data splitter.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        logger.info(f"Initialized EEG Data Splitter with seed {random_state}")
        
    def subject_independent_split(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data by subjects (subject-independent evaluation).
        
        Args:
            data: EEG data (n_samples, ...)
            labels: Labels (n_samples,)
            subject_ids: Subject ID for each sample (n_samples,)
            train_ratio: Proportion of subjects for training
            val_ratio: Proportion of subjects for validation
            test_ratio: Proportion of subjects for testing
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        # Validate ratios
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
            "Ratios must sum to 1.0"
            
        # Get unique subjects
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        logger.info(
            f"Splitting {n_subjects} subjects: "
            f"train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}"
        )
        
        # Shuffle subjects
        np.random.seed(self.random_state)
        shuffled_subjects = np.random.permutation(unique_subjects)
        
        # Calculate split indices
        train_end = int(n_subjects * train_ratio)
        val_end = train_end + int(n_subjects * val_ratio)
        
        # Split subjects
        train_subjects = shuffled_subjects[:train_end]
        val_subjects = shuffled_subjects[train_end:val_end]
        test_subjects = shuffled_subjects[val_end:]
        
        # Split data by subjects
        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)
        test_mask = np.isin(subject_ids, test_subjects)
        
        splits = {
            'train': (data[train_mask], labels[train_mask]),
            'val': (data[val_mask], labels[val_mask]),
            'test': (data[test_mask], labels[test_mask])
        }
        
        logger.info(
            f"Split sizes - Train: {np.sum(train_mask)}, "
            f"Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}"
        )
        logger.info(
            f"Train subjects: {len(train_subjects)}, "
            f"Val subjects: {len(val_subjects)}, "
            f"Test subjects: {len(test_subjects)}"
        )
        
        return splits
        
    def temporal_split(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data temporally (earlier samples for train, later for test).
        
        Args:
            data: EEG data (n_samples, ...)
            labels: Labels (n_samples,)
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
            "Ratios must sum to 1.0"
            
        n_samples = len(data)
        
        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split data
        splits = {
            'train': (data[:train_end], labels[:train_end]),
            'val': (data[train_end:val_end], labels[train_end:val_end]),
            'test': (data[val_end:], labels[val_end:])
        }
        
        logger.info(
            f"Temporal split - Train: {train_end}, "
            f"Val: {val_end - train_end}, Test: {n_samples - val_end}"
        )
        
        return splits
        
    def random_split(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = False
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Random split of data samples.
        
        Args:
            data: EEG data (n_samples, ...)
            labels: Labels (n_samples,)
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            stratify: Whether to maintain class distribution
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
            "Ratios must sum to 1.0"
            
        # First split: train vs (val + test)
        stratify_arg = labels if stratify else None
        
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data, labels,
            test_size=(val_ratio + test_ratio),
            random_state=self.random_state,
            stratify=stratify_arg
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        stratify_arg = temp_labels if stratify else None
        
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.random_state,
            stratify=stratify_arg
        )
        
        splits = {
            'train': (train_data, train_labels),
            'val': (val_data, val_labels),
            'test': (test_data, test_labels)
        }
        
        logger.info(
            f"Random split - Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}"
        )
        
        return splits
        
    def cross_subject_kfold(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
        n_folds: int = 5
    ) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        K-fold cross-validation with subject-independent folds.
        
        Args:
            data: EEG data (n_samples, ...)
            labels: Labels (n_samples,)
            subject_ids: Subject ID for each sample
            n_folds: Number of folds
            
        Returns:
            List of fold dictionaries with 'train' and 'test' splits
        """
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        logger.info(
            f"Creating {n_folds}-fold cross-validation "
            f"with {n_subjects} subjects"
        )
        
        # Create K-fold splitter for subjects
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        folds = []
        
        for fold_idx, (train_subject_idx, test_subject_idx) in enumerate(
            kf.split(unique_subjects)
        ):
            train_subjects = unique_subjects[train_subject_idx]
            test_subjects = unique_subjects[test_subject_idx]
            
            # Split data by subjects
            train_mask = np.isin(subject_ids, train_subjects)
            test_mask = np.isin(subject_ids, test_subjects)
            
            fold_split = {
                'train': (data[train_mask], labels[train_mask]),
                'test': (data[test_mask], labels[test_mask])
            }
            
            folds.append(fold_split)
            
            logger.info(
                f"Fold {fold_idx + 1}/{n_folds} - "
                f"Train: {np.sum(train_mask)} samples ({len(train_subjects)} subjects), "
                f"Test: {np.sum(test_mask)} samples ({len(test_subjects)} subjects)"
            )
            
        return folds
        
    def leave_one_subject_out(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray
    ) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Leave-one-subject-out cross-validation.
        
        Args:
            data: EEG data (n_samples, ...)
            labels: Labels (n_samples,)
            subject_ids: Subject ID for each sample
            
        Returns:
            List of splits (one per subject)
        """
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        logger.info(
            f"Creating leave-one-subject-out CV with {n_subjects} subjects"
        )
        
        splits = []
        
        for test_subject in unique_subjects:
            train_mask = subject_ids != test_subject
            test_mask = subject_ids == test_subject
            
            split = {
                'train': (data[train_mask], labels[train_mask]),
                'test': (data[test_mask], labels[test_mask])
            }
            
            splits.append(split)
            
            logger.debug(
                f"Subject {test_subject} - "
                f"Train: {np.sum(train_mask)}, Test: {np.sum(test_mask)}"
            )
            
        logger.info(f"Created {len(splits)} LOSO splits")
        
        return splits


class DatasetOrganizer:
    """Organize and save dataset splits."""
    
    @staticmethod
    def save_splits(
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
        save_dir: str,
        prefix: str = ''
    ) -> None:
        """
        Save dataset splits to disk.
        
        Args:
            splits: Dictionary with 'train', 'val', 'test' splits
            save_dir: Directory to save splits
            prefix: Prefix for filenames
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for split_name, (split_data, split_labels) in splits.items():
            data_filename = os.path.join(
                save_dir, f'{prefix}{split_name}_data.npy'
            )
            labels_filename = os.path.join(
                save_dir, f'{prefix}{split_name}_labels.npy'
            )
            
            np.save(data_filename, split_data)
            np.save(labels_filename, split_labels)
            
            logger.info(
                f"Saved {split_name} split: "
                f"{split_data.shape[0]} samples to {save_dir}"
            )
            
    @staticmethod
    def load_splits(
        load_dir: str,
        prefix: str = ''
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load dataset splits from disk.
        
        Args:
            load_dir: Directory containing splits
            prefix: Prefix for filenames
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        import os
        
        splits = {}
        
        for split_name in ['train', 'val', 'test']:
            data_filename = os.path.join(
                load_dir, f'{prefix}{split_name}_data.npy'
            )
            labels_filename = os.path.join(
                load_dir, f'{prefix}{split_name}_labels.npy'
            )
            
            if os.path.exists(data_filename) and os.path.exists(labels_filename):
                split_data = np.load(data_filename)
                split_labels = np.load(labels_filename)
                splits[split_name] = (split_data, split_labels)
                
                logger.info(
                    f"Loaded {split_name} split: {split_data.shape[0]} samples"
                )
            else:
                logger.warning(f"Split {split_name} not found in {load_dir}")
                
        return splits
        
    @staticmethod
    def get_split_statistics(
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Dict]:
        """
        Get statistics for each split.
        
        Args:
            splits: Dictionary with splits
            
        Returns:
            Dictionary with statistics for each split
        """
        stats = {}
        
        for split_name, (split_data, split_labels) in splits.items():
            stats[split_name] = {
                'n_samples': len(split_data),
                'data_shape': split_data.shape,
                'n_unique_labels': len(np.unique(split_labels)),
                'label_distribution': {
                    str(label): np.sum(split_labels == label)
                    for label in np.unique(split_labels)
                }
            }
            
        return stats


def main():
    """Example usage of data splitting."""
    # Generate synthetic EEG data
    np.random.seed(42)
    n_subjects = 10
    samples_per_subject = 100
    n_channels = 32
    n_timepoints = 500
    
    # Create data
    all_data = []
    all_labels = []
    all_subjects = []
    
    for subject_id in range(n_subjects):
        subject_data = np.random.randn(
            samples_per_subject, n_channels, n_timepoints
        )
        subject_labels = np.random.randint(0, 2, samples_per_subject)
        subject_ids = np.full(samples_per_subject, subject_id)
        
        all_data.append(subject_data)
        all_labels.append(subject_labels)
        all_subjects.append(subject_ids)
        
    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)
    subject_ids = np.concatenate(all_subjects)
    
    # Initialize splitter
    splitter = EEGDataSplitter(random_state=42)
    
    # Subject-independent split
    splits = splitter.subject_independent_split(
        data, labels, subject_ids,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # Get statistics
    stats = DatasetOrganizer.get_split_statistics(splits)
    
    logger.info("Split Statistics:")
    for split_name, split_stats in stats.items():
        logger.info(f"{split_name}: {split_stats}")
        
    # Cross-validation
    cv_folds = splitter.cross_subject_kfold(data, labels, subject_ids, n_folds=5)
    logger.info(f"Created {len(cv_folds)} CV folds")


if __name__ == "__main__":
    main()
