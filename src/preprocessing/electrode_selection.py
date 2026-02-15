"""
Electrode Selection and Channel Optimization

This module implements strategies for selecting optimal EEG electrodes/channels
for specific tasks, reducing dimensionality while preserving signal quality.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElectrodeSelector:
    """Electrode/channel selection for EEG signals."""
    
    # Standard 10-20 electrode positions for common montages
    STANDARD_10_20 = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]
    
    # Region-specific electrodes
    REGIONS = {
        'frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
        'central': ['C3', 'Cz', 'C4'],
        'temporal': ['T3', 'T4', 'T5', 'T6'],
        'parietal': ['P3', 'Pz', 'P4'],
        'occipital': ['O1', 'O2']
    }
    
    def __init__(self, ch_names: List[str]):
        """
        Initialize electrode selector.
        
        Args:
            ch_names: List of available channel names
        """
        self.ch_names = ch_names
        self.n_channels = len(ch_names)
        logger.info(f"Initialized ElectrodeSelector with {self.n_channels} channels")
        
    def select_by_names(self, selected_names: List[str]) -> Tuple[List[int], List[str]]:
        """
        Select electrodes by channel names.
        
        Args:
            selected_names: List of channel names to select
            
        Returns:
            Tuple of (selected_indices, selected_names)
        """
        selected_indices = []
        found_names = []
        
        for name in selected_names:
            if name in self.ch_names:
                idx = self.ch_names.index(name)
                selected_indices.append(idx)
                found_names.append(name)
            else:
                logger.warning(f"Channel {name} not found in available channels")
                
        logger.info(f"Selected {len(selected_indices)} channels by name")
        return selected_indices, found_names
        
    def select_by_region(self, region: str) -> Tuple[List[int], List[str]]:
        """
        Select electrodes by brain region.
        
        Args:
            region: Brain region ('frontal', 'central', 'temporal', 'parietal', 'occipital')
            
        Returns:
            Tuple of (selected_indices, selected_names)
        """
        if region not in self.REGIONS:
            raise ValueError(f"Unknown region: {region}. Available: {list(self.REGIONS.keys())}")
            
        region_channels = self.REGIONS[region]
        return self.select_by_names(region_channels)
        
    def select_by_variance(
        self,
        data: np.ndarray,
        n_channels: int,
        method: str = 'highest'
    ) -> Tuple[List[int], np.ndarray]:
        """
        Select channels based on signal variance.
        
        Args:
            data: EEG data (channels × time_points)
            n_channels: Number of channels to select
            method: 'highest' or 'lowest' variance
            
        Returns:
            Tuple of (selected_indices, variance_scores)
        """
        logger.info(f"Selecting {n_channels} channels by {method} variance")
        
        # Compute variance for each channel
        variance = np.var(data, axis=1)
        
        # Select channels
        if method == 'highest':
            selected_indices = np.argsort(variance)[-n_channels:].tolist()
        elif method == 'lowest':
            selected_indices = np.argsort(variance)[:n_channels].tolist()
        else:
            raise ValueError(f"Unknown method: {method}")
            
        logger.info(f"Selected channels: {[self.ch_names[i] for i in selected_indices]}")
        
        return selected_indices, variance
        
    def select_by_mutual_information(
        self,
        data: np.ndarray,
        target: np.ndarray,
        n_channels: int
    ) -> Tuple[List[int], np.ndarray]:
        """
        Select channels based on mutual information with target.
        
        Args:
            data: EEG data (channels × samples)
            target: Target variable for each sample
            n_channels: Number of channels to select
            
        Returns:
            Tuple of (selected_indices, MI_scores)
        """
        logger.info(f"Selecting {n_channels} channels by mutual information")
        
        # Compute MI for each channel
        mi_scores = np.zeros(data.shape[0])
        
        for ch_idx in range(data.shape[0]):
            mi_scores[ch_idx] = mutual_info_regression(
                data[ch_idx, :].reshape(-1, 1),
                target,
                random_state=42
            )[0]
            
        # Select top channels
        selected_indices = np.argsort(mi_scores)[-n_channels:].tolist()
        
        logger.info(
            f"Selected channels: {[self.ch_names[i] for i in selected_indices]} "
            f"with MI scores: {mi_scores[selected_indices]}"
        )
        
        return selected_indices, mi_scores
        
    def select_by_correlation(
        self,
        data: np.ndarray,
        n_channels: int,
        threshold: float = 0.9
    ) -> Tuple[List[int], np.ndarray]:
        """
        Select channels while avoiding highly correlated ones.
        
        Args:
            data: EEG data (channels × time_points)
            n_channels: Number of channels to select
            threshold: Correlation threshold
            
        Returns:
            Tuple of (selected_indices, correlation_matrix)
        """
        logger.info(f"Selecting {n_channels} low-correlation channels (threshold={threshold})")
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data)
        
        selected_indices = []
        remaining_indices = list(range(data.shape[0]))
        
        while len(selected_indices) < n_channels and remaining_indices:
            # Select channel with highest variance among remaining
            variances = np.var(data[remaining_indices], axis=1)
            best_idx_in_remaining = np.argmax(variances)
            best_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_idx)
            
            # Remove highly correlated channels
            correlated = np.where(np.abs(corr_matrix[best_idx]) > threshold)[0]
            remaining_indices = [idx for idx in remaining_indices if idx not in correlated]
            
        logger.info(f"Selected {len(selected_indices)} low-correlation channels")
        
        return selected_indices, corr_matrix
        
    def apply_selection(
        self,
        data: np.ndarray,
        selected_indices: List[int]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply channel selection to data.
        
        Args:
            data: EEG data (channels × time_points)
            selected_indices: Indices of channels to keep
            
        Returns:
            Tuple of (selected_data, selected_channel_names)
        """
        selected_data = data[selected_indices, :]
        selected_names = [self.ch_names[i] for i in selected_indices]
        
        logger.info(
            f"Applied selection: {data.shape[0]} → {selected_data.shape[0]} channels"
        )
        
        return selected_data, selected_names


class ChannelOptimizer:
    """Advanced channel optimization techniques."""
    
    @staticmethod
    def optimize_by_pca(
        data: np.ndarray,
        n_components: int,
        explained_variance_threshold: float = 0.95
    ) -> Tuple[np.ndarray, PCA]:
        """
        Optimize channels using PCA dimensionality reduction.
        
        Args:
            data: EEG data (samples × channels)
            n_components: Number of components to keep
            explained_variance_threshold: Minimum explained variance ratio
            
        Returns:
            Tuple of (transformed_data, fitted_pca)
        """
        logger.info(f"Applying PCA with {n_components} components")
        
        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(data)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_var:.3f}")
        
        if explained_var < explained_variance_threshold:
            logger.warning(
                f"Explained variance ({explained_var:.3f}) below threshold "
                f"({explained_variance_threshold})"
            )
            
        return transformed, pca
        
    @staticmethod
    def rank_channels_by_importance(
        data: np.ndarray,
        target: np.ndarray,
        ch_names: List[str],
        method: str = 'mutual_info'
    ) -> List[Tuple[str, float]]:
        """
        Rank all channels by importance.
        
        Args:
            data: EEG data (channels × samples)
            target: Target variable
            ch_names: Channel names
            method: Ranking method ('mutual_info', 'variance', 'correlation')
            
        Returns:
            List of (channel_name, importance_score) tuples, sorted by importance
        """
        logger.info(f"Ranking channels by {method}")
        
        if method == 'mutual_info':
            scores = np.zeros(data.shape[0])
            for ch_idx in range(data.shape[0]):
                scores[ch_idx] = mutual_info_regression(
                    data[ch_idx, :].reshape(-1, 1),
                    target,
                    random_state=42
                )[0]
                
        elif method == 'variance':
            scores = np.var(data, axis=1)
            
        elif method == 'correlation':
            scores = np.array([
                np.abs(np.corrcoef(data[i], target)[0, 1])
                for i in range(data.shape[0])
            ])
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Create ranked list
        ranked = sorted(
            zip(ch_names, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info(f"Top 5 channels: {ranked[:5]}")
        
        return ranked
        
    @staticmethod
    def create_montage_subset(
        full_montage: List[str],
        n_channels: int,
        strategy: str = 'distributed'
    ) -> List[str]:
        """
        Create a subset of channels from a full montage.
        
        Args:
            full_montage: Full list of channel names
            n_channels: Number of channels to select
            strategy: Selection strategy ('distributed', 'frontal', 'posterior')
            
        Returns:
            List of selected channel names
        """
        logger.info(f"Creating montage subset: {n_channels} channels, strategy={strategy}")
        
        if strategy == 'distributed':
            # Select evenly distributed channels
            step = len(full_montage) // n_channels
            subset = full_montage[::step][:n_channels]
            
        elif strategy == 'frontal':
            # Prioritize frontal channels
            frontal_priority = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC1', 'FC2']
            subset = [ch for ch in frontal_priority if ch in full_montage][:n_channels]
            
        elif strategy == 'posterior':
            # Prioritize posterior channels
            posterior_priority = ['O1', 'O2', 'P3', 'Pz', 'P4', 'PO3', 'PO4', 'T5', 'T6']
            subset = [ch for ch in posterior_priority if ch in full_montage][:n_channels]
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        logger.info(f"Selected montage subset: {subset}")
        
        return subset


def main():
    """Example usage of electrode selection."""
    # Generate synthetic EEG data
    np.random.seed(42)
    n_channels = 32
    n_samples = 1000
    
    # Simulate channel names
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    
    # Simulate EEG data
    data = np.random.randn(n_channels, n_samples) * 10e-6
    
    # Initialize selector
    selector = ElectrodeSelector(ch_names)
    
    # Select by variance
    selected_indices, variance = selector.select_by_variance(data, n_channels=16)
    selected_data, selected_names = selector.apply_selection(data, selected_indices)
    
    logger.info(f"Original data: {data.shape}")
    logger.info(f"Selected data: {selected_data.shape}")
    logger.info(f"Selected channels: {selected_names}")
    
    # Test correlation-based selection
    selected_indices_corr, corr_matrix = selector.select_by_correlation(
        data, n_channels=16, threshold=0.8
    )
    logger.info(f"Correlation-based selection: {len(selected_indices_corr)} channels")


if __name__ == "__main__":
    main()
