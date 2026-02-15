"""
Real-Time Inference Pipeline

Implements streaming EEG processing for online BCI:
- Circular buffer for continuous data
- Sliding window processing
- Online preprocessing
- Low-latency decoding
- Real-time metrics
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Callable
import numpy as np
from collections import deque
import time
from dataclasses import dataclass


@dataclass
class StreamConfig:
    """Configuration for streaming inference."""
    sample_rate: int = 500
    window_size: int = 500
    hop_size: int = 100
    num_channels: int = 128
    max_latency_ms: float = 100.0
    buffer_size: int = 5000


class CircularBuffer:
    """
    Circular buffer for streaming data.
    """
    
    def __init__(
        self,
        num_channels: int,
        buffer_size: int
    ):
        """
        Initialize circular buffer.
        
        Args:
            num_channels: Number of EEG channels
            buffer_size: Buffer capacity (samples)
        """
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        
        # Initialize buffer
        self.buffer = np.zeros((num_channels, buffer_size), dtype=np.float32)
        self.write_pos = 0
        self.samples_written = 0
        
    def add_samples(
        self,
        samples: np.ndarray
    ):
        """
        Add samples to buffer.
        
        Args:
            samples: (num_channels, num_samples)
        """
        num_samples = samples.shape[1]
        
        # Write to buffer (handle wrap-around)
        for i in range(num_samples):
            self.buffer[:, self.write_pos] = samples[:, i]
            self.write_pos = (self.write_pos + 1) % self.buffer_size
            self.samples_written += 1
            
    def get_window(
        self,
        window_size: int,
        offset: int = 0
    ) -> Optional[np.ndarray]:
        """
        Get latest window of data.
        
        Args:
            window_size: Window size (samples)
            offset: Offset from current position
            
        Returns:
            Window data (num_channels, window_size) or None if not enough data
        """
        if self.samples_written < window_size:
            return None
            
        # Calculate start position
        end_pos = (self.write_pos - 1 - offset) % self.buffer_size
        start_pos = (end_pos - window_size + 1) % self.buffer_size
        
        # Extract window (handle wrap-around)
        if start_pos < end_pos:
            window = self.buffer[:, start_pos:end_pos+1]
        else:
            # Wrapped around
            window = np.concatenate([
                self.buffer[:, start_pos:],
                self.buffer[:, :end_pos+1]
            ], axis=1)
            
        return window
        
    def clear(self):
        """Clear buffer."""
        self.buffer[:] = 0
        self.write_pos = 0
        self.samples_written = 0


class OnlinePreprocessor:
    """
    Online preprocessing for streaming data.
    """
    
    def __init__(
        self,
        sample_rate: int = 500,
        bandpass: Tuple[float, float] = (0.5, 40.0),
        notch_freq: Optional[float] = 50.0
    ):
        """
        Initialize online preprocessor.
        
        Args:
            sample_rate: Sampling rate
            bandpass: Bandpass filter range (Hz)
            notch_freq: Notch filter frequency (Hz)
        """
        self.sample_rate = sample_rate
        self.bandpass = bandpass
        self.notch_freq = notch_freq
        
        # Initialize filters
        self._init_filters()
        
        # Filter states (for continuous filtering)
        self.filter_states = None
        
    def _init_filters(self):
        """Initialize digital filters."""
        from scipy import signal
        
        # Bandpass filter
        sos_bandpass = signal.butter(
            4,
            self.bandpass,
            btype='bandpass',
            fs=self.sample_rate,
            output='sos'
        )
        self.sos_bandpass = sos_bandpass
        
        # Notch filter
        if self.notch_freq is not None:
            b_notch, a_notch = signal.iirnotch(
                self.notch_freq,
                Q=30,
                fs=self.sample_rate
            )
            self.b_notch = b_notch
            self.a_notch = a_notch
        else:
            self.b_notch = None
            self.a_notch = None
            
    def process(
        self,
        data: np.ndarray,
        stateful: bool = True
    ) -> np.ndarray:
        """
        Process data online.
        
        Args:
            data: (num_channels, num_samples)
            stateful: Use stateful filtering (continuous)
            
        Returns:
            Processed data
        """
        from scipy import signal
        
        # Apply bandpass filter
        if stateful and self.filter_states is not None:
            filtered, self.filter_states = signal.sosfilt(
                self.sos_bandpass,
                data,
                zi=self.filter_states
            )
        else:
            filtered = signal.sosfilt(self.sos_bandpass, data)
            
            if stateful:
                # Initialize filter state
                self.filter_states = signal.sosfilt_zi(self.sos_bandpass)
                self.filter_states = self.filter_states[:, None] * filtered[:, :1]
                
        # Apply notch filter
        if self.b_notch is not None:
            filtered = signal.filtfilt(self.b_notch, self.a_notch, filtered)
            
        return filtered
        
    def reset(self):
        """Reset filter states."""
        self.filter_states = None


class StreamingInference:
    """
    Real-time streaming inference pipeline.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: StreamConfig,
        preprocessor: Optional[OnlinePreprocessor] = None,
        decoder: Optional[Callable] = None,
        device: str = 'cpu'
    ):
        """
        Initialize streaming inference.
        
        Args:
            model: NEST model
            config: Stream configuration
            preprocessor: Online preprocessor
            decoder: Decoding function
            device: Device
        """
        self.model = model.to(device).eval()
        self.config = config
        self.preprocessor = preprocessor or OnlinePreprocessor(config.sample_rate)
        self.decoder = decoder
        self.device = device
        
        # Initialize buffer
        self.buffer = CircularBuffer(
            config.num_channels,
            config.buffer_size
        )
        
        # Metrics
        self.num_processed = 0
        self.total_latency = 0.0
        self.latencies = deque(maxlen=100)
        
    def add_samples(
        self,
        samples: np.ndarray
    ):
        """
        Add new samples to buffer.
        
        Args:
            samples: (num_channels, num_samples)
        """
        self.buffer.add_samples(samples)
        
    def process_window(
        self,
        return_latency: bool = False
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """
        Process latest window.
        
        Args:
            return_latency: Return processing latency
            
        Returns:
            Model output and optional latency
        """
        start_time = time.time() if return_latency else None
        
        # Get window
        window = self.buffer.get_window(self.config.window_size)
        
        if window is None:
            return None, None
            
        # Preprocess
        preprocessed = self.preprocessor.process(window, stateful=True)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(preprocessed).unsqueeze(0).float()
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model.encode(input_tensor)
            
        # Measure latency
        latency = None
        if return_latency:
            latency = (time.time() - start_time) * 1000  # ms
            self.latencies.append(latency)
            self.total_latency += latency
            
        self.num_processed += 1
        
        return output, latency
        
    def decode_output(
        self,
        encoder_output: torch.Tensor
    ) -> str:
        """
        Decode output to text.
        
        Args:
            encoder_output: Encoder output
            
        Returns:
            Decoded text
        """
        if self.decoder is None:
            return ""
            
        return self.decoder(encoder_output)
        
    def get_metrics(self) -> dict:
        """
        Get performance metrics.
        
        Returns:
            Metrics dictionary
        """
        if not self.latencies:
            return {}
            
        latencies = list(self.latencies)
        
        return {
            'num_processed': self.num_processed,
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_hz': 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        }
        
    def reset(self):
        """Reset pipeline."""
        self.buffer.clear()
        self.preprocessor.reset()
        self.num_processed = 0
        self.total_latency = 0.0
        self.latencies.clear()


class SlidingWindowProcessor:
    """
    Process data with sliding window.
    """
    
    def __init__(
        self,
        window_size: int,
        hop_size: int
    ):
        """
        Initialize sliding window processor.
        
        Args:
            window_size: Window size (samples)
            hop_size: Hop between windows (samples)
        """
        self.window_size = window_size
        self.hop_size = hop_size
        
    def process_stream(
        self,
        data: np.ndarray,
        process_fn: Callable[[np.ndarray], any]
    ) -> List[any]:
        """
        Process data stream with sliding window.
        
        Args:
            data: (num_channels, total_samples)
            process_fn: Function to process each window
            
        Returns:
            List of processing results
        """
        results = []
        
        num_samples = data.shape[1]
        num_windows = (num_samples - self.window_size) // self.hop_size + 1
        
        for i in range(num_windows):
            start = i * self.hop_size
            end = start + self.window_size
            
            window = data[:, start:end]
            result = process_fn(window)
            results.append(result)
            
        return results


class LatencyMonitor:
    """
    Monitor and analyze system latency.
    """
    
    def __init__(self, max_samples: int = 1000):
        """
        Initialize latency monitor.
        
        Args:
            max_samples: Maximum samples to keep
        """
        self.latencies = deque(maxlen=max_samples)
        self.timestamps = deque(maxlen=max_samples)
        
    def record(self, latency_ms: float):
        """
        Record latency measurement.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self.latencies.append(latency_ms)
        self.timestamps.append(time.time())
        
    def get_statistics(self) -> dict:
        """
        Get latency statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.latencies:
            return {}
            
        latencies = np.array(self.latencies)
        
        return {
            'count': len(latencies),
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'median': float(np.median(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'p999': float(np.percentile(latencies, 99.9))
        }
        
    def check_sla(self, threshold_ms: float) -> Tuple[bool, float]:
        """
        Check if latency meets SLA.
        
        Args:
            threshold_ms: Latency threshold (ms)
            
        Returns:
            (meets_sla, violation_rate)
        """
        if not self.latencies:
            return True, 0.0
            
        violations = sum(1 for lat in self.latencies if lat > threshold_ms)
        violation_rate = violations / len(self.latencies)
        meets_sla = violation_rate < 0.01  # 99% must meet threshold
        
        return meets_sla, violation_rate


def main():
    """Example usage."""
    print("="*60)
    print("Real-Time Inference Pipeline")
    print("="*60)
    
    # Configuration
    config = StreamConfig(
        sample_rate=500,
        window_size=500,
        hop_size=100,
        num_channels=128,
        max_latency_ms=100.0
    )
    
    # Test 1: Circular buffer
    print("\n1. Circular Buffer")
    buffer = CircularBuffer(num_channels=128, buffer_size=1000)
    
    # Add samples
    samples = np.random.randn(128, 100)
    buffer.add_samples(samples)
    
    # Get window
    window = buffer.get_window(window_size=50)
    print(f"   Buffer: {buffer.samples_written} samples written")
    print(f"   Window shape: {window.shape if window is not None else None}")
    
    # Test 2: Online preprocessor
    print("\n2. Online Preprocessor")
    preprocessor = OnlinePreprocessor(
        sample_rate=500,
        bandpass=(0.5, 40.0),
        notch_freq=50.0
    )
    
    data = np.random.randn(128, 500)
    processed = preprocessor.process(data, stateful=True)
    
    print(f"   Input shape: {data.shape}")
    print(f"   Output shape: {processed.shape}")
    
    # Test 3: Streaming inference (dummy)
    print("\n3. Streaming Inference")
    
    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(128, 256, 3, padding=1)
            
        def encode(self, x):
            return self.conv(x)
            
    model = DummyModel()
    
    stream = StreamingInference(
        model,
        config,
        preprocessor=preprocessor,
        device='cpu'
    )
    
    # Simulate streaming
    for i in range(5):
        samples = np.random.randn(128, 100)
        stream.add_samples(samples)
        
        output, latency = stream.process_window(return_latency=True)
        if output is not None:
            print(f"   Window {i+1}: latency={latency:.2f}ms")
            
    metrics = stream.get_metrics()
    print(f"   Mean latency: {metrics.get('mean_latency_ms', 0):.2f}ms")
    
    # Test 4: Latency monitor
    print("\n4. Latency Monitor")
    monitor = LatencyMonitor()
    
    # Record latencies
    latencies = [50, 60, 55, 120, 58, 62, 54]
    for lat in latencies:
        monitor.record(lat)
        
    stats = monitor.get_statistics()
    print(f"   Mean: {stats['mean']:.2f}ms")
    print(f"   P95: {stats['p95']:.2f}ms")
    print(f"   P99: {stats['p99']:.2f}ms")
    
    meets_sla, violation_rate = monitor.check_sla(threshold_ms=100.0)
    print(f"   Meets 100ms SLA: {meets_sla} (violations: {violation_rate:.1%})")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
