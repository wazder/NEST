#!/usr/bin/env python3
"""
Real-Time Inference Demo

This script demonstrates real-time EEG-to-text decoding using streaming inference.
"""

import torch
import numpy as np
import argparse
import time
from pathlib import Path

from src.models import ModelFactory
from src.evaluation import (
    StreamingInference,
    StreamConfig,
    OnlinePreprocessor,
    LatencyMonitor
)


def simulate_eeg_stream(num_channels=128, sample_rate=500, duration=10.0):
    """Simulate streaming EEG data."""
    total_samples = int(duration * sample_rate)
    chunk_size = 100  # Samples per chunk
    
    for i in range(0, total_samples, chunk_size):
        # Simulate EEG data (random walk for realism)
        chunk = np.random.randn(num_channels, min(chunk_size, total_samples - i))
        yield chunk
        
        # Simulate real-time delay
        time.sleep(chunk_size / sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Real-time NEST inference demo')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Model config')
    parser.add_argument('--duration', type=float, default=10.0, help='Demo duration (sec)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    print("="*80)
    print("NEST Real-Time Inference Demo")
    print("="*80)
    
    # Load model
    print("\n[1/4] Loading model...")
    print("-"*80)
    
    checkpoint = torch.load(args.model, map_location=args.device)
    
    model = ModelFactory.from_config_file(
        args.config,
        model_key='nest_transformer',
        vocab_size=5000
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"✓ Model loaded")
    
    # Setup streaming
    print("\n[2/4] Setting up streaming pipeline...")
    print("-"*80)
    
    config = StreamConfig(
        sample_rate=500,
        window_size=500,
        hop_size=100,
        num_channels=128,
        max_latency_ms=100.0,
        buffer_size=5000
    )
    
    preprocessor = OnlinePreprocessor(
        sample_rate=500,
        bandpass=(0.5, 40.0),
        notch_freq=50.0
    )
    
    stream = StreamingInference(
        model=model,
        config=config,
        preprocessor=preprocessor,
        device=args.device
    )
    
    monitor = LatencyMonitor(max_samples=1000)
    
    print(f"✓ Streaming configured:")
    print(f"  - Sample rate: {config.sample_rate} Hz")
    print(f"  - Window size: {config.window_size} samples")
    print(f"  - Hop size: {config.hop_size} samples")
    print(f"  - Max latency: {config.max_latency_ms} ms")
    
    # Run demo
    print("\n[3/4] Running real-time inference...")
    print("-"*80)
    print("\nStreaming EEG data...")
    print("(Simulated data for demonstration)\n")
    
    window_count = 0
    
    for i, eeg_chunk in enumerate(simulate_eeg_stream(
        num_channels=128,
        sample_rate=500,
        duration=args.duration
    )):
        # Add to streaming buffer
        stream.add_samples(eeg_chunk)
        
        # Process window
        output, latency = stream.process_window(return_latency=True)
        
        if output is not None:
            window_count += 1
            
            if latency is not None:
                monitor.record(latency)
                
                # Print every 5 windows
                if window_count % 5 == 0:
                    print(f"Window {window_count:3d}: latency={latency:5.1f}ms", end='')
                    
                    if latency > config.max_latency_ms:
                        print(" ⚠️  EXCEEDS TARGET")
                    else:
                        print(" ✓")
    
    # Results
    print("\n[4/4] Results")
    print("-"*80)
    
    # Stream metrics
    metrics = stream.get_metrics()
    
    print("\nStream Performance:")
    print(f"  Windows processed: {metrics['num_processed']}")
    print(f"  Mean latency: {metrics['mean_latency_ms']:.2f} ms")
    print(f"  Std latency: {metrics['std_latency_ms']:.2f} ms")
    print(f"  P95 latency: {metrics['p95_latency_ms']:.2f} ms")
    print(f"  P99 latency: {metrics['p99_latency_ms']:.2f} ms")
    print(f"  Throughput: {metrics['throughput_hz']:.1f} Hz")
    
    # Latency analysis
    stats = monitor.get_statistics()
    
    print("\nLatency Statistics:")
    print(f"  Min: {stats['min']:.2f} ms")
    print(f"  Max: {stats['max']:.2f} ms")
    print(f"  Median: {stats['median']:.2f} ms")
    print(f"  P99.9: {stats['p999']:.2f} ms")
    
    # SLA check
    meets_sla, violation_rate = monitor.check_sla(config.max_latency_ms)
    
    print(f"\nSLA Check ({config.max_latency_ms}ms target):")
    print(f"  Meets SLA: {'✓ YES' if meets_sla else '✗ NO'}")
    print(f"  Violation rate: {violation_rate:.2%}")
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)
    
    if meets_sla:
        print("\n✓ System meets real-time requirements!")
        print("  Ready for online BCI applications.")
    else:
        print("\n⚠️  System does not meet real-time requirements.")
        print("  Consider:")
        print("  - Use faster device (GPU)")
        print("  - Apply model optimization")
        print("  - Reduce window size")
        print("  - Use greedy decoding")


if __name__ == '__main__':
    main()
