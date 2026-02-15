"""
EEG Signal Filtering Module

Implements various filtering techniques for EEG signal preprocessing,
including band-pass filtering for artifact removal.
"""

import numpy as np
import mne
from typing import Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGFilter:
    """EEG signal filtering utilities."""
    
    def __init__(
        self,
        l_freq: float = 0.5,
        h_freq: float = 50.0,
        notch_freq: Optional[float] = None,
        sfreq: float = 500.0
    ):
        """
        Initialize EEG filter.
        
        Args:
            l_freq: Low cutoff frequency for high-pass filter (Hz)
            h_freq: High cutoff frequency for low-pass filter (Hz)
            notch_freq: Frequency for notch filter (e.g., 50 or 60 Hz for powerline)
            sfreq: Sampling frequency (Hz)
        """
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.sfreq = sfreq
        
        logger.info(
            f"Initialized EEG Filter: "
            f"band-pass={l_freq}-{h_freq} Hz, "
            f"notch={notch_freq} Hz, "
            f"sampling_rate={sfreq} Hz"
        )
        
    def bandpass_filter(
        self,
        data: np.ndarray,
        l_freq: Optional[float] = None,
        h_freq: Optional[float] = None,
        method: str = 'fir',
        phase: str = 'zero'
    ) -> np.ndarray:
        """
        Apply band-pass filter to EEG data.
        
        Args:
            data: EEG data array (channels × time_points)
            l_freq: Low cutoff frequency (uses default if None)
            h_freq: High cutoff frequency (uses default if None)
            method: Filter method ('fir' or 'iir')
            phase: Phase of the filter ('zero', 'zero-double', or 'minimum')
            
        Returns:
            Filtered EEG data
        """
        l_freq = l_freq if l_freq is not None else self.l_freq
        h_freq = h_freq if h_freq is not None else self.h_freq
        
        logger.info(f"Applying band-pass filter: {l_freq}-{h_freq} Hz")
        
        # Create MNE Info object for filtering
        info = mne.create_info(
            ch_names=[f'ch_{i}' for i in range(data.shape[0])],
            sfreq=self.sfreq,
            ch_types='eeg'
        )
        
        # Create RawArray
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Apply band-pass filter
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            phase=phase,
            verbose=False
        )
        
        filtered_data = raw.get_data()
        
        logger.info(f"Filtering complete. Output shape: {filtered_data.shape}")
        return filtered_data
        
    def notch_filter(
        self,
        data: np.ndarray,
        freqs: Optional[Union[float, list]] = None,
        notch_widths: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply notch filter to remove powerline noise.
        
        Args:
            data: EEG data array (channels × time_points)
            freqs: Frequency or frequencies to notch filter
            notch_widths: Width of notch filter
            
        Returns:
            Filtered EEG data
        """
        if freqs is None:
            if self.notch_freq is None:
                logger.warning("No notch frequency specified. Skipping notch filter.")
                return data
            freqs = self.notch_freq
            
        logger.info(f"Applying notch filter at {freqs} Hz")
        
        # Create MNE Info object
        info = mne.create_info(
            ch_names=[f'ch_{i}' for i in range(data.shape[0])],
            sfreq=self.sfreq,
            ch_types='eeg'
        )
        
        # Create RawArray
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Apply notch filter
        raw.notch_filter(
            freqs=freqs,
            notch_widths=notch_widths,
            verbose=False
        )
        
        filtered_data = raw.get_data()
        
        logger.info(f"Notch filtering complete.")
        return filtered_data
        
    def highpass_filter(
        self,
        data: np.ndarray,
        l_freq: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply high-pass filter (remove slow drifts).
        
        Args:
            data: EEG data array
            l_freq: Low cutoff frequency
            
        Returns:
            Filtered EEG data
        """
        l_freq = l_freq if l_freq is not None else self.l_freq
        return self.bandpass_filter(data, l_freq=l_freq, h_freq=None)
        
    def lowpass_filter(
        self,
        data: np.ndarray,
        h_freq: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply low-pass filter (remove high-frequency noise).
        
        Args:
            data: EEG data array
            h_freq: High cutoff frequency
            
        Returns:
            Filtered EEG data
        """
        h_freq = h_freq if h_freq is not None else self.h_freq
        return self.bandpass_filter(data, l_freq=None, h_freq=h_freq)
        
    def apply_all_filters(
        self,
        data: np.ndarray,
        apply_notch: bool = True
    ) -> np.ndarray:
        """
        Apply complete filtering pipeline.
        
        Args:
            data: EEG data array (channels × time_points)
            apply_notch: Whether to apply notch filter
            
        Returns:
            Filtered EEG data
        """
        logger.info("Applying complete filtering pipeline")
        
        # Apply band-pass filter
        filtered = self.bandpass_filter(data)
        
        # Apply notch filter if requested
        if apply_notch and self.notch_freq is not None:
            filtered = self.notch_filter(filtered)
            
        logger.info("Filtering pipeline complete")
        return filtered
        
    def get_frequency_bands(
        self,
        data: np.ndarray,
        bands: Optional[dict] = None
    ) -> dict:
        """
        Extract specific frequency bands from EEG data.
        
        Args:
            data: EEG data array (channels × time_points)
            bands: Dictionary of frequency bands
                  e.g., {'delta': (0.5, 4), 'theta': (4, 8), ...}
                  
        Returns:
            Dictionary with filtered data for each band
        """
        if bands is None:
            # Standard EEG frequency bands
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
            
        band_data = {}
        
        for band_name, (l_freq, h_freq) in bands.items():
            logger.info(f"Extracting {band_name} band: {l_freq}-{h_freq} Hz")
            band_data[band_name] = self.bandpass_filter(
                data, l_freq=l_freq, h_freq=h_freq
            )
            
        return band_data


class AdaptiveFilter:
    """Adaptive filtering techniques for EEG."""
    
    @staticmethod
    def apply_car(data: np.ndarray) -> np.ndarray:
        """
        Apply Common Average Reference (CAR) filtering.
        
        Args:
            data: EEG data array (channels × time_points)
            
        Returns:
            CAR-filtered data
        """
        logger.info("Applying Common Average Reference (CAR)")
        
        # Compute average across all channels
        avg_signal = np.mean(data, axis=0, keepdims=True)
        
        # Subtract average from each channel
        car_data = data - avg_signal
        
        return car_data
        
    @staticmethod
    def apply_laplacian(data: np.ndarray, neighbors: dict) -> np.ndarray:
        """
        Apply Laplacian (surface) filtering.
        
        Args:
            data: EEG data array (channels × time_points)
            neighbors: Dictionary mapping channel index to list of neighbor indices
            
        Returns:
            Laplacian-filtered data
        """
        logger.info("Applying Laplacian spatial filter")
        
        filtered = np.zeros_like(data)
        
        for ch_idx, neighbor_idx in neighbors.items():
            if neighbor_idx:
                # V_filtered = V_center - mean(V_neighbors)
                neighbor_mean = np.mean(data[neighbor_idx, :], axis=0)
                filtered[ch_idx, :] = data[ch_idx, :] - neighbor_mean
            else:
                # If no neighbors, keep original
                filtered[ch_idx, :] = data[ch_idx, :]
                
        return filtered


def main():
    """Example usage of EEG filtering."""
    # Generate synthetic EEG data
    np.random.seed(42)
    n_channels = 64
    n_samples = 5000
    sfreq = 500.0
    
    # Simulate EEG with noise
    time = np.arange(n_samples) / sfreq
    signal = np.sin(2 * np.pi * 10 * time)  # 10 Hz signal
    noise = np.random.randn(n_channels, n_samples) * 0.1
    powerline = np.sin(2 * np.pi * 50 * time) * 0.05  # 50 Hz powerline noise
    
    data = signal + noise + powerline
    
    # Initialize filter
    eeg_filter = EEGFilter(l_freq=0.5, h_freq=50.0, notch_freq=50.0, sfreq=sfreq)
    
    # Apply filtering
    filtered_data = eeg_filter.apply_all_filters(data)
    
    # Extract frequency bands
    bands = eeg_filter.get_frequency_bands(data)
    
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Filtered data shape: {filtered_data.shape}")
    logger.info(f"Extracted bands: {list(bands.keys())}")
    
    # Apply CAR
    car_data = AdaptiveFilter.apply_car(filtered_data)
    logger.info(f"CAR-filtered data shape: {car_data.shape}")


if __name__ == "__main__":
    main()
