"""Common test fixtures and utilities for NEST tests."""
import numpy as np
import pytest
import torch
from pathlib import Path
from typing import Dict, Tuple


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def sample_eeg_data() -> Tuple[np.ndarray, int]:
    """
    Generate synthetic EEG data for testing.
    
    Returns:
        Tuple of (eeg_data, sampling_rate)
        eeg_data shape: (n_channels, n_samples)
    """
    n_channels = 105  # ZuCo standard
    duration = 5.0  # seconds
    sampling_rate = 500  # Hz
    n_samples = int(duration * sampling_rate)
    
    # Generate synthetic EEG with multiple frequency bands
    t = np.linspace(0, duration, n_samples)
    eeg = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Delta (0.5-4 Hz)
        eeg[ch] += 10 * np.sin(2 * np.pi * 2 * t)
        # Theta (4-8 Hz)
        eeg[ch] += 5 * np.sin(2 * np.pi * 6 * t)
        # Alpha (8-13 Hz)
        eeg[ch] += 8 * np.sin(2 * np.pi * 10 * t)
        # Beta (13-30 Hz)
        eeg[ch] += 3 * np.sin(2 * np.pi * 20 * t)
        # Gamma (30-50 Hz)
        eeg[ch] += 2 * np.sin(2 * np.pi * 40 * t)
        # Add noise
        eeg[ch] += np.random.normal(0, 0.5, n_samples)
    
    return eeg, sampling_rate


@pytest.fixture
def sample_batch_eeg() -> torch.Tensor:
    """
    Generate batch of EEG tensors for model testing.
    
    Returns:
        torch.Tensor of shape (batch_size, n_channels, seq_len)
    """
    batch_size = 4
    n_channels = 105
    seq_len = 1000
    
    return torch.randn(batch_size, n_channels, seq_len)


@pytest.fixture
def sample_text_data() -> Dict[str, list]:
    """
    Generate sample text data for testing.
    
    Returns:
        Dictionary with texts and word-level timestamps
    """
    return {
        "texts": [
            "The quick brown fox jumps over the lazy dog",
            "Hello world this is a test sentence",
            "Neural networks process information efficiently",
        ],
        "word_timestamps": [
            [(0, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.5), (1.5, 1.9), 
             (1.9, 2.2), (2.2, 2.4), (2.4, 2.9), (2.9, 3.3)],
            [(0, 0.4), (0.4, 0.9), (0.9, 1.2), (1.2, 1.4), (1.4, 1.5), 
             (1.5, 1.9), (1.9, 2.5)],
            [(0, 0.5), (0.5, 1.1), (1.1, 1.7), (1.7, 2.5), (2.5, 3.4)],
        ],
    }


@pytest.fixture
def model_config() -> Dict:
    """
    Generate minimal model configuration for testing.
    
    Returns:
        Configuration dictionary
    """
    return {
        "model": {
            "name": "nest",
            "encoder": {
                "spatial": {
                    "type": "cnn",
                    "channels": [105, 64, 128, 256],
                    "kernel_size": 3,
                },
                "temporal": {
                    "type": "lstm",
                    "hidden_size": 512,
                    "num_layers": 2,
                    "bidirectional": True,
                },
            },
            "decoder": {
                "type": "attention",
                "vocab_size": 5000,
                "embedding_dim": 256,
                "hidden_size": 512,
                "num_layers": 2,
            },
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "epochs": 100,
            "optimizer": "adam",
        },
    }


@pytest.fixture
def preprocessing_config() -> Dict:
    """
    Generate preprocessing configuration for testing.
    
    Returns:
        Preprocessing configuration dictionary
    """
    return {
        "filtering": {
            "bandpass": {
                "low": 0.5,
                "high": 50.0,
            },
            "notch": {
                "freq": 50.0,
                "quality": 30,
            },
        },
        "artifact_removal": {
            "method": "ica",
            "n_components": 20,
        },
        "electrode_selection": {
            "method": "variance",
            "n_channels": 64,
        },
        "augmentation": {
            "noise_std": 0.1,
            "time_shift_max": 0.1,
        },
    }


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """
    Create temporary data directory structure.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    return data_dir


@pytest.fixture
def temp_checkpoint_dir(tmp_path) -> Path:
    """
    Create temporary checkpoint directory.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Path to temporary checkpoint directory
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def device() -> torch.device:
    """
    Get available device (CUDA if available, else CPU).
    
    Returns:
        torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            self.pad_token_id = 0
            self.sos_token_id = 1
            self.eos_token_id = 2
            self.unk_token_id = 3
            
        def encode(self, text: str) -> list:
            """Mock encoding."""
            return [i % self.vocab_size for i in range(len(text.split()))]
            
        def decode(self, ids: list) -> str:
            """Mock decoding."""
            return " ".join([f"word_{i}" for i in ids])
            
        def batch_encode(self, texts: list) -> list:
            """Mock batch encoding."""
            return [self.encode(text) for text in texts]
    
    return MockTokenizer()


def assert_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]):
    """
    Assert tensor has expected shape.
    
    Args:
        tensor: Input tensor
        expected_shape: Expected shape (can use None for dynamic dimensions)
    """
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None:
            assert actual == expected, (
                f"Dimension {i}: expected {expected}, got {actual}. "
                f"Full shape: {tensor.shape}"
            )


def assert_valid_probability_distribution(tensor: torch.Tensor, dim: int = -1):
    """
    Assert tensor represents valid probability distribution.
    
    Args:
        tensor: Input tensor
        dim: Dimension to sum over
    """
    assert torch.all(tensor >= 0), "Probabilities must be non-negative"
    assert torch.all(tensor <= 1), "Probabilities must be <= 1"
    sums = tensor.sum(dim=dim)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        "Probabilities must sum to 1"
    )
