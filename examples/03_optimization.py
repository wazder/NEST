#!/usr/bin/env python3
"""
Optimization Example: Model Pruning and Quantization

This example demonstrates:
1. Model pruning (magnitude-based, structured, iterative)
2. Model quantization (PTQ, QAT)
3. Performance comparison
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time

from src.data.zuco_dataset import ZucoDataset
from src.models import ModelFactory
from src.utils.tokenizer import CharTokenizer
from src.evaluation.pruning import (
    MagnitudePruner,
    StructuredPruner,
    IterativePruner
)
from src.evaluation.quantization import (
    PostTrainingQuantizer,
    QuantizationAwareTrainer
)
from src.evaluation.profiling import ModelProfiler

def benchmark_model(model, test_loader, device, name="Model"):
    """Benchmark inference speed and accuracy"""
    model.eval()
    total_time = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch['eeg'].to(device)
            
            start = time.time()
            output = model(eeg)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed = time.time() - start
            
            total_time += elapsed
            num_batches += 1
            
            if num_batches >= 100:  # Benchmark on 100 batches
                break
    
    avg_time = (total_time / num_batches) * 1000  # Convert to ms
    print(f"{name}:")
    print(f"  Average inference time: {avg_time:.2f} ms/batch")
    print(f"  Throughput: {16 / (avg_time/1000):.1f} samples/sec")
    
    return avg_time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("Model Optimization: Pruning and Quantization")
    print("="*60)
    
    # Setup
    chars = list('abcdefghijklmnopqrstuvwxyz .,!?\'')
    tokenizer = CharTokenizer(vocab=chars)
    
    # Load test data
    test_dataset = ZucoDataset(
        data_dir='data/processed/zuco/test',
        tokenizer=tokenizer
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    # Load trained model
    print("\nLoading base model...")
    model = ModelFactory.from_config_file(
        'configs/model.yaml',
        model_key='nest_attention',
        vocab_size=len(tokenizer)
    )
    
    # Load checkpoint
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Profile base model
    profiler = ModelProfiler()
    base_flops = profiler.profile_flops(model, (1, 32, 1000))
    base_memory = profiler.profile_memory(model, torch.randn(1, 32, 1000).to(device))
    
    print(f"\nBase Model Statistics:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  FLOPs: {base_flops:,}")
    print(f"  Memory: {base_memory['model_size_mb']:.2f} MB")
    
    # Benchmark base model
    print("\nBenchmarking base model...")
    base_time = benchmark_model(model, test_loader, device, "Base Model")
    
    # ========== PRUNING ==========
    print("\n" + "="*60)
    print("PRUNING EXPERIMENTS")
    print("="*60)
    
    # 1. Magnitude Pruning
    print("\n1. Magnitude Pruning (50% sparsity)...")
    mag_pruner = MagnitudePruner(model, amount=0.5)
    pruned_model = mag_pruner.prune()
    pruned_model = pruned_model.to(device)
    
    mag_time = benchmark_model(pruned_model, test_loader, device, "Magnitude Pruned (50%)")
    speedup = base_time / mag_time
    print(f"  Speedup: {speedup:.2f}x")
    
    # 2. Structured Pruning
    print("\n2. Structured Pruning (30% filters)...")
    struct_pruner = StructuredPruner(model, amount=0.3, pruning_dim=0)
    struct_model = struct_pruner.prune()
    struct_model = struct_model.to(device)
    
    struct_time = benchmark_model(struct_model, test_loader, device, "Structured Pruned (30%)")
    speedup = base_time / struct_time
    print(f"  Speedup: {speedup:.2f}x")
    
    # ========== QUANTIZATION ==========
    print("\n" + "="*60)
    print("QUANTIZATION EXPERIMENTS")
    print("="*60)
    
    # 3. Dynamic Quantization
    print("\n3. Dynamic Quantization (INT8)...")
    dynamic_quantizer = PostTrainingQuantizer(
        model=model,
        quantization_type='dynamic'
    )
    dynamic_model = dynamic_quantizer.quantize()
    
    # Move to CPU for quantization benchmark
    dynamic_model = dynamic_model.cpu()
    dynamic_time = benchmark_model(
        dynamic_model,
        test_loader,
        torch.device('cpu'),
        "Dynamically Quantized (INT8)"
    )
    
    # Check model size reduction
    torch.save(dynamic_model.state_dict(), 'checkpoints/temp_dynamic.pt')
    dynamic_size = Path('checkpoints/temp_dynamic.pt').stat().st_size / 1024 / 1024
    base_size = base_memory['model_size_mb']
    print(f"  Size reduction: {base_size:.2f} MB -> {dynamic_size:.2f} MB "
          f"({dynamic_size/base_size*100:.1f}%)")
    
    # 4. Static Quantization
    print("\n4. Static Quantization (INT8)...")
    static_quantizer = PostTrainingQuantizer(
        model=model.cpu(),
        quantization_type='static'
    )
    
    # Need calibration data for static quantization
    calib_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    static_model = static_quantizer.quantize(calibration_loader=calib_loader)
    static_time = benchmark_model(
        static_model,
        test_loader,
        torch.device('cpu'),
        "Statically Quantized (INT8)"
    )
    
    torch.save(static_model.state_dict(), 'checkpoints/temp_static.pt')
    static_size = Path('checkpoints/temp_static.pt').stat().st_size / 1024 / 1024
    print(f"  Size reduction: {base_size:.2f} MB -> {static_size:.2f} MB "
          f"({static_size/base_size*100:.1f}%)")
    
    # ========== COMBINED OPTIMIZATION ==========
    print("\n" + "="*60)
    print("COMBINED OPTIMIZATION (Pruning + Quantization)")
    print("="*60)
    
    # Prune then quantize
    print("\n5. Magnitude Pruning (40%) + Dynamic Quantization...")
    combined_pruner = MagnitudePruner(model, amount=0.4)
    combined_model = combined_pruner.prune()
    
    combined_quantizer = PostTrainingQuantizer(
        model=combined_model.cpu(),
        quantization_type='dynamic'
    )
    combined_model = combined_quantizer.quantize()
    
    combined_time = benchmark_model(
        combined_model,
        test_loader,
        torch.device('cpu'),
        "Pruned (40%) + Quantized (INT8)"
    )
    
    torch.save(combined_model.state_dict(), 'checkpoints/temp_combined.pt')
    combined_size = Path('checkpoints/temp_combined.pt').stat().st_size / 1024 / 1024
    print(f"  Size reduction: {base_size:.2f} MB -> {combined_size:.2f} MB "
          f"({combined_size/base_size*100:.1f}%)")
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"\n{'Method':<30} {'Size (MB)':<15} {'Inference (ms)':<20} {'Speedup':<10}")
    print("-" * 75)
    print(f"{'Base Model':<30} {base_size:<15.2f} {base_time:<20.2f} {'1.00x':<10}")
    print(f"{'Magnitude Pruned (50%)':<30} {base_size:<15.2f} {mag_time:<20.2f} {base_time/mag_time:<10.2f}x")
    print(f"{'Structured Pruned (30%)':<30} {base_size:<15.2f} {struct_time:<20.2f} {base_time/struct_time:<10.2f}x")
    print(f"{'Dynamic Quantized':<30} {dynamic_size:<15.2f} {dynamic_time:<20.2f} {'-':<10}")
    print(f"{'Static Quantized':<30} {static_size:<15.2f} {static_time:<20.2f} {'-':<10}")
    print(f"{'Pruned + Quantized':<30} {combined_size:<15.2f} {combined_time:<20.2f} {'-':<10}")
    
    # Clean up temp files
    Path('checkpoints/temp_dynamic.pt').unlink(missing_ok=True)
    Path('checkpoints/temp_static.pt').unlink(missing_ok=True)
    Path('checkpoints/temp_combined.pt').unlink(missing_ok=True)
    
    print("\n" + "="*60)
    print("Optimization experiments complete!")
    print("="*60)

if __name__ == '__main__':
    main()
