"""
Data Augmentation for EEG Signals

This module implements various data augmentation techniques specifically
designed for EEG signals to address limited sample sizes.
"""

import numpy as np
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGAugmentation:
    """Data augmentation techniques for EEG signals."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize EEG augmentation.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        logger.info(f"Initialized EEG Augmentation with seed {random_state}")
        
    def add_gaussian_noise(
        self,
        data: np.ndarray,
        noise_level: float = 0.1,
        snr_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise to EEG signal.
        
        Args:
            data: EEG data (channels × time_points)
            noise_level: Noise standard deviation as fraction of signal std
            snr_db: Signal-to-noise ratio in dB (overrides noise_level if provided)
            
        Returns:
            Augmented EEG data
        """
        if snr_db is not None:
            # Calculate noise level from SNR
            signal_power = np.mean(data ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power)
        else:
            noise_std = noise_level * np.std(data)
            
        noise = np.random.normal(0, noise_std, data.shape)
        augmented = data + noise
        
        logger.debug(f"Added Gaussian noise with std={noise_std:.2e}")
        return augmented
        
    def time_shift(
        self,
        data: np.ndarray,
        shift_range: Tuple[int, int] = (-50, 50)
    ) -> np.ndarray:
        """
        Apply random time shift to EEG signal.
        
        Args:
            data: EEG data (channels × time_points)
            shift_range: Range of shift in samples (min, max)
            
        Returns:
            Time-shifted EEG data
        """
        shift = np.random.randint(shift_range[0], shift_range[1])
        
        if shift > 0:
            augmented = np.pad(data, ((0, 0), (shift, 0)), mode='edge')[:, :-shift]
        elif shift < 0:
            augmented = np.pad(data, ((0, 0), (0, -shift)), mode='edge')[:, -shift:]
        else:
            augmented = data.copy()
            
        logger.debug(f"Applied time shift: {shift} samples")
        return augmented
        
    def amplitude_scaling(
        self,
        data: np.ndarray,
        scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Apply random amplitude scaling to EEG signal.
        
        Args:
            data: EEG data (channels × time_points)
            scale_range: Range of scaling factors (min, max)
            
        Returns:
            Amplitude-scaled EEG data
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        augmented = data * scale
        
        logger.debug(f"Applied amplitude scaling: {scale:.3f}")
        return augmented
        
    def channel_dropout(
        self,
        data: np.ndarray,
        dropout_prob: float = 0.1
    ) -> np.ndarray:
        """
        Randomly dropout (zero out) channels.
        
        Args:
            data: EEG data (channels × time_points)
            dropout_prob: Probability of dropping each channel
            
        Returns:
            Augmented EEG data with dropped channels
        """
        mask = np.random.random(data.shape[0]) > dropout_prob
        augmented = data.copy()
        augmented[~mask, :] = 0
        
        n_dropped = np.sum(~mask)
        logger.debug(f"Dropped {n_dropped} / {data.shape[0]} channels")
        return augmented
        
    def time_warping(
        self,
        data: np.ndarray,
        warp_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """
        Apply time warping (speed up or slow down) to EEG signal.
        
        Args:
            data: EEG data (channels × time_points)
            warp_range: Range of warping factors (min, max)
            
        Returns:
            Time-warped EEG data
        """
        warp_factor = np.random.uniform(warp_range[0], warp_range[1])
        n_samples = data.shape[1]
        
        # Create warped time indices
        original_indices = np.arange(n_samples)
        warped_indices = np.linspace(0, n_samples - 1, int(n_samples / warp_factor))
        
        # Interpolate
        augmented = np.zeros((data.shape[0], n_samples))
        for ch in range(data.shape[0]):
            augmented[ch, :] = np.interp(original_indices, warped_indices, data[ch, :])
            
        logger.debug(f"Applied time warping: factor={warp_factor:.3f}")
        return augmented
        
    def segment_shuffle(
        self,
        data: np.ndarray,
        n_segments: int = 4
    ) -> np.ndarray:
        """
        Shuffle segments of the EEG signal.
        
        Args:
            data: EEG data (channels × time_points)
            n_segments: Number of segments to divide signal into
            
        Returns:
            Augmented EEG data with shuffled segments
        """
        n_samples = data.shape[1]
        segment_length = n_samples // n_segments
        
        # Create segments
        segments = []
        for i in range(n_segments):
            start = i * segment_length
            end = start + segment_length if i < n_segments - 1 else n_samples
            segments.append(data[:, start:end])
            
        # Shuffle segments
        np.random.shuffle(segments)
        
        # Concatenate
        augmented = np.concatenate(segments, axis=1)
        
        logger.debug(f"Shuffled {n_segments} segments")
        return augmented
        
    def mixup(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        alpha: float = 0.2
    ) -> np.ndarray:
        """
        Apply mixup augmentation between two EEG samples.
        
        Args:
            data1: First EEG sample (channels × time_points)
            data2: Second EEG sample (channels × time_points)
            alpha: Beta distribution parameter for mixing
            
        Returns:
            Mixed EEG data
        """
        lam = np.random.beta(alpha, alpha)
        augmented = lam * data1 + (1 - lam) * data2
        
        logger.debug(f"Applied mixup with lambda={lam:.3f}")
        return augmented
        
    def frequency_shift(
        self,
        data: np.ndarray,
        shift_hz: float = 1.0,
        sfreq: float = 500.0
    ) -> np.ndarray:
        """
        Apply frequency shift using FFT.
        
        Args:
            data: EEG data (channels × time_points)
            shift_hz: Frequency shift in Hz
            sfreq: Sampling frequency
            
        Returns:
            Frequency-shifted EEG data
        """
        # FFT
        fft_data = np.fft.fft(data, axis=1)
        freqs = np.fft.fftfreq(data.shape[1], 1/sfreq)
        
        # Shift in frequency domain
        shift_bins = int(shift_hz * data.shape[1] / sfreq)
        shifted_fft = np.roll(fft_data, shift_bins, axis=1)
        
        # IFFT
        augmented = np.real(np.fft.ifft(shifted_fft, axis=1))
        
        logger.debug(f"Applied frequency shift: {shift_hz} Hz")
        return augmented
        
    def add_band_noise(
        self,
        data: np.ndarray,
        band: Tuple[float, float],
        sfreq: float,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        Add noise in specific frequency band.
        
        Args:
            data: EEG data (channels × time_points)
            band: Frequency band (low_freq, high_freq) in Hz
            sfreq: Sampling frequency
            noise_level: Noise amplitude
            
        Returns:
            Augmented EEG data
        """
        n_samples = data.shape[1]
        time = np.arange(n_samples) / sfreq
        
        # Generate band-limited noise
        noise = np.zeros_like(data)
        for _ in range(10):  # Sum several frequencies in the band
            freq = np.random.uniform(band[0], band[1])
            phase = np.random.uniform(0, 2 * np.pi)
            noise += np.sin(2 * np.pi * freq * time + phase)
            
        noise = noise * noise_level / 10
        augmented = data + noise
        
        logger.debug(f"Added band noise in {band[0]}-{band[1]} Hz")
        return augmented
        
    def random_augment(
        self,
        data: np.ndarray,
        n_augmentations: int = 1,
        augmentation_prob: float = 0.5,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Apply random combination of augmentations.
        
        Args:
            data: EEG data (channels × time_points)
            n_augmentations: Number of augmented samples to generate
            augmentation_prob: Probability of applying each augmentation
            **kwargs: Additional arguments for specific augmentations
            
        Returns:
            List of augmented EEG samples
        """
        augmented_samples = []
        
        augmentation_funcs = [
            lambda x: self.add_gaussian_noise(x, **kwargs.get('noise', {})),
            lambda x: self.amplitude_scaling(x, **kwargs.get('scaling', {})),
            lambda x: self.time_shift(x, **kwargs.get('shift', {})),
        ]
        
        for _ in range(n_augmentations):
            aug_data = data.copy()
            
            for aug_func in augmentation_funcs:
                if np.random.random() < augmentation_prob:
                    aug_data = aug_func(aug_data)
                    
            augmented_samples.append(aug_data)
            
        logger.info(f"Generated {n_augmentations} augmented samples")
        return augmented_samples


class AdvancedAugmentation:
    """Advanced augmentation techniques."""
    
    @staticmethod
    def sliding_window(
        data: np.ndarray,
        window_size: int,
        stride: int
    ) -> List[np.ndarray]:
        """
        Create multiple samples using sliding window.
        
        Args:
            data: EEG data (channels × time_points)
            window_size: Size of each window
            stride: Stride between windows
            
        Returns:
            List of windowed samples
        """
        n_samples = data.shape[1]
        windows = []
        
        for start in range(0, n_samples - window_size + 1, stride):
            end = start + window_size
            windows.append(data[:, start:end])
            
        logger.info(f"Created {len(windows)} windows from sliding window")
        return windows
        
    @staticmethod
    def smote_eeg(
        data: np.ndarray,
        n_synthetic: int = 1,
        k_neighbors: int = 5
    ) -> List[np.ndarray]:
        """
        SMOTE-like synthetic sample generation for EEG.
        
        Args:
            data: Multiple EEG samples (n_samples, channels, time_points)
            n_synthetic: Number of synthetic samples per original
            k_neighbors: Number of neighbors to consider
            
        Returns:
            List of synthetic EEG samples
        """
        synthetic_samples = []
        n_samples = data.shape[0]
        
        for i in range(n_samples):
            for _ in range(n_synthetic):
                # Choose random neighbor
                neighbor_idx = np.random.choice(
                    [j for j in range(n_samples) if j != i]
                )
                
                # Interpolate
                alpha = np.random.random()
                synthetic = alpha * data[i] + (1 - alpha) * data[neighbor_idx]
                synthetic_samples.append(synthetic)
                
        logger.info(f"Generated {len(synthetic_samples)} SMOTE-like synthetic samples")
        return synthetic_samples


def main():
    """Example usage of EEG augmentation."""
    # Generate synthetic EEG data
    np.random.seed(42)
    n_channels = 32
    n_samples = 1000
    sfreq = 500.0
    
    # Simulate EEG signal
    time = np.arange(n_samples) / sfreq
    signal = np.sin(2 * np.pi * 10 * time)  # 10 Hz signal
    data = np.tile(signal, (n_channels, 1)) + np.random.randn(n_channels, n_samples) * 0.1
    
    # Initialize augmentation
    augmenter = EEGAugmentation(random_state=42)
    
    # Apply various augmentations
    noisy = augmenter.add_gaussian_noise(data, noise_level=0.1)
    scaled = augmenter.amplitude_scaling(data, scale_range=(0.8, 1.2))
    shifted = augmenter.time_shift(data, shift_range=(-50, 50))
    warped = augmenter.time_warping(data, warp_range=(0.9, 1.1))
    
    # Random augmentation
    aug_samples = augmenter.random_augment(data, n_augmentations=5)
    
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Generated {len(aug_samples)} augmented samples")
    logger.info(f"Each augmented sample shape: {aug_samples[0].shape}")


if __name__ == "__main__":
    main()
