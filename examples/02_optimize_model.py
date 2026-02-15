#!/usr/bin/env python3
"""
Model Optimization Example

This script demonstrates how to optimize a trained NEST model for deployment:
- Pruning
- Quantization
- Inference optimization
- ONNX export
"""

import torch
import argparse
from pathlib import Path
import json

from src.models import ModelFactory
from src.evaluation import (
    IterativePruner,
    QATTrainer,
    InferenceOptimizer,
    ModelProfiler,
    ComparisonBenchmark
)


def dummy_train(model):
    """Dummy training function for pruning."""
    return 0.5


def dummy_val(model):
    """Dummy validation function for pruning."""
    return 0.6


def main():
    parser = argparse.ArgumentParser(description='Optimize NEST model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Model config')
    parser.add_argument('--output_dir', type=str, default='optimized', help='Output directory')
    parser.add_argument('--prune_sparsity', type=float, default=0.4, help='Target sparsity')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    parser.add_argument('--fp16', action='store_true', help='Use FP16')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("NEST Model Optimization")
    print("="*80)
    
    # Load model
    print("\n[1/5] Loading model...")
    print("-"*80)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    model = ModelFactory.from_config_file(
        args.config,
        model_key='nest_transformer',
        vocab_size=5000  # Adjust as needed
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("✓ Model loaded")
    
    # Profile baseline
    print("\n[2/5] Profiling baseline...")
    print("-"*80)
    
    profiler = ModelProfiler(model)
    baseline_profile = profiler.profile_summary((1, 128, 500))
    
    print(f"Baseline model:")
    print(f"  Parameters: {baseline_profile['parameters']['total']:,}")
    print(f"  GFLOPs: {baseline_profile['flops']['gflops']:.2f}")
    print(f"  Memory: {baseline_profile['memory']['total_mb']:.2f} MB")
    
    models = {'Baseline': model}
    
    # Pruning
    if args.prune_sparsity > 0:
        print(f"\n[3/5] Pruning (target sparsity: {args.prune_sparsity:.0%})...")
        print("-"*80)
        
        pruner = IterativePruner(
            model,
            target_sparsity=args.prune_sparsity,
            num_iterations=3
        )
        
        # For real pruning, use actual train/val functions:
        # results = pruner.prune_and_finetune(train_fn, val_fn)
        
        # For demo, just apply magnitude pruning
        from src.evaluation import MagnitudePruner
        mag_pruner = MagnitudePruner(model, amount=args.prune_sparsity)
        mag_pruner.prune()
        
        sparsities = mag_pruner.compute_sparsity()
        print(f"✓ Overall sparsity: {sparsities['overall']:.2%}")
        
        mag_pruner.make_permanent()
        
        # Save pruned model
        pruned_path = output_dir / 'pruned_model.pt'
        torch.save(model.state_dict(), pruned_path)
        print(f"✓ Saved pruned model to {pruned_path}")
        
        models['Pruned'] = model
    
    # Quantization
    if args.quantize:
        print("\n[4/5] Quantizing model...")
        print("-"*80)
        
        from src.evaluation import PostTrainingQuantizer
        
        ptq = PostTrainingQuantizer(model, backend='fbgemm')
        ptq.prepare()
        
        # For real quantization, calibrate with data:
        # ptq.calibrate(calibration_loader, num_batches=100)
        
        quantized_model = ptq.quantize()
        
        # Measure size
        sizes = ptq.measure_size(model, quantized_model)
        print(f"✓ Size reduction: {sizes['reduction']:.1%}")
        print(f"✓ Compression ratio: {sizes['compression_ratio']:.1f}x")
        
        # Save quantized model
        quant_path = output_dir / 'quantized_model.pt'
        torch.save(quantized_model.state_dict(), quant_path)
        print(f"✓ Saved quantized model to {quant_path}")
        
        models['Quantized'] = quantized_model
        model = quantized_model
    
    # Inference optimization
    print("\n[5/5] Optimizing inference...")
    print("-"*80)
    
    optimizer = InferenceOptimizer(
        model,
        device='cpu',
        use_fp16=args.fp16,
        use_torchscript=True
    )
    
    # Benchmark
    metrics = optimizer.benchmark(
        batch_size=1,
        num_iterations=50,
        warmup_iterations=10
    )
    
    print(f"✓ Mean inference time: {metrics['mean_time']*1000:.2f} ms")
    print(f"✓ Throughput: {metrics['throughput']:.1f} samples/sec")
    
    # Export ONNX
    onnx_path = output_dir / 'model.onnx'
    optimizer.export_onnx(str(onnx_path))
    print(f"✓ Exported ONNX to {onnx_path}")
    
    # Export TorchScript
    ts_path = output_dir / 'model_torchscript.pt'
    optimizer.export_torchscript(str(ts_path), method='trace')
    print(f"✓ Exported TorchScript to {ts_path}")
    
    # Final comparison
    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)
    
    # Compare models
    if len(models) > 1:
        print("\nModel Comparison:")
        comparison = ComparisonBenchmark.compare_models(
            models,
            input_shape=(1, 128, 500),
            device='cpu'
        )
        ComparisonBenchmark.print_comparison(comparison)
    
    # Summary
    print("\nGenerated files:")
    if args.prune_sparsity > 0:
        print(f"  - pruned_model.pt (sparsity: {args.prune_sparsity:.0%})")
    if args.quantize:
        print(f"  - quantized_model.pt (INT8)")
    print(f"  - model.onnx (for deployment)")
    print(f"  - model_torchscript.pt (optimized)")
    
    print("\nNext steps:")
    print("  - Deploy with: examples/03_deploy_model.py")
    print("  - Test real-time: examples/04_realtime_demo.py")


if __name__ == '__main__':
    main()
