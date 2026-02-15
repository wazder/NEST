"""
Independent Component Analysis (ICA) for Artifact Removal

This module implements ICA-based artifact removal for EEG signals,
specifically targeting eye-blink, eye-movement, and muscle artifacts.
"""

import numpy as np
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from typing import Optional, List, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICArtifactRemoval:
    """ICA-based artifact removal for EEG signals."""
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        method: str = 'fastica',
        random_state: int = 42,
        max_iter: int = 200
    ):
        """
        Initialize ICA artifact removal.
        
        Args:
            n_components: Number of ICA components (None = use all channels)
            method: ICA algorithm ('fastica', 'infomax', 'picard')
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations
        """
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self.max_iter = max_iter
        self.ica = None
        
        logger.info(
            f"Initialized ICA Artifact Removal: "
            f"n_components={n_components}, method={method}"
        )
        
    def fit(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: Optional[List[str]] = None,
        decim: Optional[int] = None,
        reject: Optional[dict] = None
    ) -> 'ICArtifactRemoval':
        """
        Fit ICA on EEG data.
        
        Args:
            data: EEG data array (channels × time_points)
            sfreq: Sampling frequency (Hz)
            ch_names: Channel names (optional)
            decim: Decimation factor for faster computation
            reject: Rejection criteria for bad segments
            
        Returns:
            Self (fitted)
        """
        logger.info("Fitting ICA on EEG data")
        
        # Create channel names if not provided
        if ch_names is None:
            ch_names = [f'EEG{i:03d}' for i in range(data.shape[0])]
            
        # Create MNE Info object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types='eeg'
        )
        
        # Create RawArray
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Initialize ICA
        self.ica = ICA(
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state,
            max_iter=self.max_iter,
            verbose=False
        )
        
        # Fit ICA
        self.ica.fit(raw, decim=decim, reject=reject, verbose=False)
        
        logger.info(
            f"ICA fitted with {self.ica.n_components_} components "
            f"from {len(ch_names)} channels"
        )
        
        return self
        
    def find_bads_eog(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: Optional[List[str]] = None,
        threshold: float = 3.0,
        eog_channels: Optional[List[str]] = None
    ) -> List[int]:
        """
        Automatically identify ICA components related to eye artifacts.
        
        Args:
            data: EEG data array
            sfreq: Sampling frequency
            ch_names: Channel names
            threshold: Threshold for component selection (z-score)
            eog_channels: Names of EOG channels (if available)
            
        Returns:
            List of component indices to exclude
        """
        if self.ica is None:
            raise ValueError("ICA must be fitted first. Call fit() before find_bads_eog()")
            
        logger.info("Detecting eye artifact components")
        
        # Create channel names if not provided
        if ch_names is None:
            ch_names = [f'EEG{i:03d}' for i in range(data.shape[0])]
            
        # Create MNE Info object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types='eeg'
        )
        
        # Create RawArray
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Find EOG artifacts
        if eog_channels:
            eog_indices, eog_scores = self.ica.find_bads_eog(
                raw,
                ch_name=eog_channels,
                threshold=threshold,
                verbose=False
            )
        else:
            # Use frontal channels as proxy for EOG
            eog_indices, eog_scores = self.ica.find_bads_eog(
                raw,
                threshold=threshold,
                verbose=False
            )
            
        logger.info(f"Found {len(eog_indices)} EOG-related components: {eog_indices}")
        
        return eog_indices
        
    def find_bads_correlation(
        self,
        data: np.ndarray,
        sfreq: float,
        reference_signal: np.ndarray,
        ch_names: Optional[List[str]] = None,
        threshold: float = 0.8
    ) -> List[int]:
        """
        Find bad components by correlation with reference signal.
        
        Args:
            data: EEG data array
            sfreq: Sampling frequency
            reference_signal: Reference artifact signal (e.g., EOG, EMG)
            ch_names: Channel names
            threshold: Correlation threshold
            
        Returns:
            List of component indices to exclude
        """
        if self.ica is None:
            raise ValueError("ICA must be fitted first.")
            
        logger.info("Finding components correlated with reference signal")
        
        # Get ICA sources
        if ch_names is None:
            ch_names = [f'EEG{i:03d}' for i in range(data.shape[0])]
            
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        sources = self.ica.get_sources(raw).get_data()
        
        # Compute correlation with reference
        correlations = np.array([
            np.abs(np.corrcoef(sources[i], reference_signal)[0, 1])
            for i in range(sources.shape[0])
        ])
        
        # Find components above threshold
        bad_components = np.where(correlations > threshold)[0].tolist()
        
        logger.info(
            f"Found {len(bad_components)} components with correlation > {threshold}: "
            f"{bad_components}"
        )
        
        return bad_components
        
    def exclude_components(self, component_indices: List[int]) -> None:
        """
        Mark components for exclusion.
        
        Args:
            component_indices: List of component indices to exclude
        """
        if self.ica is None:
            raise ValueError("ICA must be fitted first.")
            
        self.ica.exclude = component_indices
        logger.info(f"Excluding components: {component_indices}")
        
    def apply(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply ICA cleaning (remove excluded components).
        
        Args:
            data: EEG data array (channels × time_points)
            sfreq: Sampling frequency
            ch_names: Channel names
            
        Returns:
            Cleaned EEG data
        """
        if self.ica is None:
            raise ValueError("ICA must be fitted first.")
            
        logger.info(f"Applying ICA artifact removal (excluding {len(self.ica.exclude)} components)")
        
        # Create channel names if not provided
        if ch_names is None:
            ch_names = [f'EEG{i:03d}' for i in range(data.shape[0])]
            
        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Apply ICA
        raw_clean = self.ica.apply(raw.copy(), verbose=False)
        
        cleaned_data = raw_clean.get_data()
        
        logger.info(f"ICA artifact removal complete. Output shape: {cleaned_data.shape}")
        
        return cleaned_data
        
    def plot_components(
        self,
        n_components: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ICA components for visual inspection.
        
        Args:
            n_components: Number of components to plot
            save_path: Path to save plot (optional)
        """
        if self.ica is None:
            raise ValueError("ICA must be fitted first.")
            
        logger.info(f"Plotting {n_components} ICA components")
        
        # Plot components
        fig = self.ica.plot_components(
            picks=range(min(n_components, self.ica.n_components_)),
            show=False
        )
        
        if save_path:
            fig.savefig(save_path)
            logger.info(f"Saved component plot to {save_path}")
            
    def get_component_properties(self, component_idx: int) -> dict:
        """
        Get properties of a specific ICA component.
        
        Args:
            component_idx: Component index
            
        Returns:
            Dictionary with component properties
        """
        if self.ica is None:
            raise ValueError("ICA must be fitted first.")
            
        properties = {
            'index': component_idx,
            'mixing_matrix': self.ica.mixing_matrix_[:, component_idx],
            'unmixing_matrix': self.ica.unmixing_matrix_[component_idx, :],
        }
        
        return properties


class AutomaticArtifactRejection:
    """Automatic artifact rejection using various methods."""
    
    @staticmethod
    def reject_by_amplitude(
        data: np.ndarray,
        threshold: float = 100e-6,
        mode: str = 'any'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reject segments with amplitude exceeding threshold.
        
        Args:
            data: EEG data (channels × time_points)
            threshold: Amplitude threshold in volts
            mode: 'any' or 'all' (reject if any/all channels exceed threshold)
            
        Returns:
            Tuple of (cleaned_data, bad_segments_mask)
        """
        logger.info(f"Rejecting segments with amplitude > {threshold*1e6:.1f} µV")
        
        # Find bad segments
        if mode == 'any':
            bad_segments = np.any(np.abs(data) > threshold, axis=0)
        else:  # 'all'
            bad_segments = np.all(np.abs(data) > threshold, axis=0)
            
        # Keep only good segments
        good_segments = ~bad_segments
        cleaned_data = data[:, good_segments]
        
        n_rejected = np.sum(bad_segments)
        logger.info(f"Rejected {n_rejected} / {len(bad_segments)} segments")
        
        return cleaned_data, bad_segments
        
    @staticmethod
    def reject_by_variance(
        data: np.ndarray,
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reject channels with abnormal variance.
        
        Args:
            data: EEG data (channels × time_points)
            threshold: Z-score threshold for variance
            
        Returns:
            Tuple of (cleaned_data, bad_channels_mask)
        """
        logger.info(f"Rejecting channels with abnormal variance (z > {threshold})")
        
        # Compute variance for each channel
        channel_var = np.var(data, axis=1)
        
        # Compute z-scores
        z_scores = (channel_var - np.mean(channel_var)) / np.std(channel_var)
        
        # Find bad channels
        bad_channels = np.abs(z_scores) > threshold
        
        # Keep only good channels
        cleaned_data = data[~bad_channels, :]
        
        n_rejected = np.sum(bad_channels)
        logger.info(f"Rejected {n_rejected} / {len(bad_channels)} channels")
        
        return cleaned_data, bad_channels


def main():
    """Example usage of ICA artifact removal."""
    # Generate synthetic EEG data with artifacts
    np.random.seed(42)
    n_channels = 32
    n_samples = 10000
    sfreq = 500.0
    
    # Simulate clean EEG
    time = np.arange(n_samples) / sfreq
    eeg_signal = np.random.randn(n_channels, n_samples) * 10e-6
    
    # Add eye-blink artifact (affects frontal channels more)
    blink_times = [1.0, 3.0, 5.0, 7.0, 9.0]
    for t in blink_times:
        idx = int(t * sfreq)
        if idx < n_samples - 100:
            amplitude = 50e-6
            blink = amplitude * np.exp(-((np.arange(100) - 50) ** 2) / 200)
            eeg_signal[:4, idx:idx+100] += blink  # Affect frontal channels
            
    # Initialize ICA
    ica_cleaner = ICArtifactRemoval(n_components=20, method='fastica')
    
    # Fit ICA
    ica_cleaner.fit(eeg_signal, sfreq=sfreq)
    
    # Find eye artifact components
    eog_components = ica_cleaner.find_bads_eog(eeg_signal, sfreq=sfreq, threshold=2.0)
    
    # Exclude components
    ica_cleaner.exclude_components(eog_components)
    
    # Apply ICA
    cleaned_data = ica_cleaner.apply(eeg_signal, sfreq=sfreq)
    
    logger.info(f"Original data shape: {eeg_signal.shape}")
    logger.info(f"Cleaned data shape: {cleaned_data.shape}")
    logger.info(f"Removed {len(eog_components)} artifact components")


if __name__ == "__main__":
    main()
