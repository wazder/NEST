"""
Profiling and Benchmarking Tools

Provides comprehensive profiling and analysis:
- Model profiling (FLOPs, params, memory)
- Layer-wise timing analysis
- Memory profiling
- Throughput benchmarking
- Visualization tools
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
from collections import defaultdict
import warnings


class ModelProfiler:
    """
    Profile model complexity and resource usage.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize model profiler.
        
        Args:
            model: Model to profile
        """
        self.model = model
        
    def count_parameters(
        self,
        trainable_only: bool = False
    ) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            trainable_only: Count only trainable parameters
            
        Returns:
            Parameter count dictionary
        """
        counts = {}
        total = 0
        trainable = 0
        
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            counts[name] = num_params
            total += num_params
            
            if param.requires_grad:
                trainable += num_params
                
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable,
            'by_layer': counts
        }
        
    def count_flops(
        self,
        input_shape: Tuple[int, ...],
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Count FLOPs (floating point operations).
        
        Args:
            input_shape: Input tensor shape
            detailed: Return layer-wise FLOPs
            
        Returns:
            FLOPs dictionary
        """
        # Use hooks to count operations
        flop_counts = {}
        
        def conv_hook(module, input, output):
            # Conv FLOPs = C_in * K_h * K_w * C_out * H_out * W_out
            batch_size, in_channels, *input_dims = input[0].shape
            out_channels, _, *kernel_size = module.weight.shape
            
            output_size = np.prod(output.shape[2:])
            kernel_size_prod = np.prod(kernel_size)
            
            flops = in_channels * kernel_size_prod * out_channels * output_size
            flop_counts[id(module)] = flops
            
        def linear_hook(module, input, output):
            # Linear FLOPs = in_features * out_features
            in_features = module.in_features
            out_features = module.out_features
            batch_size = input[0].shape[0]
            
            flops = in_features * out_features * batch_size
            flop_counts[id(module)] = flops
            
        # Register hooks
        hooks = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))
                
        # Forward pass
        dummy_input = torch.randn(*input_shape)
        with torch.no_grad():
            _ = self.model(dummy_input)
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        total_flops = sum(flop_counts.values())
        
        result = {
            'total_flops': total_flops,
            'gflops': total_flops / 1e9
        }
        
        if detailed:
            result['by_layer'] = flop_counts
            
        return result
        
    def estimate_memory(
        self,
        input_shape: Tuple[int, ...],
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Estimate memory usage.
        
        Args:
            input_shape: Input shape  
            batch_size: Batch size
            
        Returns:
            Memory estimates (MB)
        """
        # Parameter memory
        param_memory = sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        )
        
        # Activation memory (approximate)
        # Run forward pass and track activations
        activations = []
        
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append(output.numel() * output.element_size())
                
        hooks = []
        for module in self.model.modules():
            hooks.append(module.register_forward_hook(hook))
            
        dummy_input = torch.randn(batch_size, *input_shape[1:])
        with torch.no_grad():
            _ = self.model(dummy_input)
            
        for hook in hooks:
            hook.remove()
            
        activation_memory = sum(activations)
        
        return {
            'parameters_mb': param_memory / (1024**2),
            'activations_mb': activation_memory / (1024**2),
            'total_mb': (param_memory + activation_memory) / (1024**2)
        }
        
    def profile_summary(
        self,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """
        Get complete profile summary.
        
        Args:
            input_shape: Input shape
            
        Returns:
            Complete profile
        """
        params = self.count_parameters()
        flops = self.count_flops(input_shape)
        memory = self.estimate_memory(input_shape)
        
        return {
            'parameters': params,
            'flops': flops,
            'memory': memory
        }


class LayerTimingProfiler:
    """
    Profile layer-wise execution time.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize timing profiler.
        
        Args:
            model: Model to profile
            device: Device
        """
        self.model = model.to(device)
        self.device = device
        self.timings = defaultdict(list)
        
    def profile(
        self,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Profile layer execution times.
        
        Args:
            input_shape: Input shape
            num_iterations: Number of iterations
            warmup: Warmup iterations
            
        Returns:
            Layer timing statistics
        """
        # Hooks to measure time
        timing_data = defaultdict(list)
        
        def make_hook(name):
            def hook(module, input, output):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                    
                start = time.time()
                # Time is measured by the hook being called
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                
                # Record in global dict during forward
                pass
                
            return hook
            
        # Better approach: use torch.profiler
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA if self.device == 'cuda' else torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            dummy_input = torch.randn(*input_shape).to(self.device)
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = self.model(dummy_input)
                    
            # Profile
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = self.model(dummy_input)
                    
        # Parse results
        events = prof.key_averages()
        
        layer_stats = {}
        for evt in events:
            if evt.key.startswith('aten::'):
                continue  # Skip low-level ops
                
            layer_stats[evt.key] = {
                'cpu_time_ms': evt.cpu_time_total / 1000,
                'cuda_time_ms': evt.cuda_time_total / 1000 if self.device == 'cuda' else 0,
                'count': evt.count,
                'cpu_time_avg_ms': evt.cpu_time_total / evt.count / 1000
            }
            
        return layer_stats


class MemoryProfiler:
    """
    Profile memory usage during training/inference.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize memory profiler.
        
        Args:
            device: Device (cuda only)
        """
        if device != 'cuda':
            warnings.warn("Memory profiling only works on CUDA")
            
        self.device = device
        
    def profile_inference(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """
        Profile inference memory.
        
        Args:
            model: Model
            input_tensor: Input
            
        Returns:
            Memory statistics (MB)
        """
        if self.device != 'cuda':
            return {}
            
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Measure
        model = model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            _ = model(input_tensor)
            
        # Get stats
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_mb': peak
        }
        
    def profile_training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Profile training step memory.
        
        Args:
            model: Model
            optimizer: Optimizer
            input_tensor: Input
            target_tensor: Target
            criterion: Loss function
            
        Returns:
            Memory statistics
        """
        if self.device != 'cuda':
            return {}
            
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Setup
        model = model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        
        # Training step
        model.train()
        
        # Forward
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Get stats
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        allocated = torch.cuda.memory_allocated() / (1024**2)
        
        return {
            'peak_training_mb': peak,
            'post_training_mb': allocated
        }


class ThroughputBenchmark:
    """
    Benchmark model throughput.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ):
        """
        Initialize throughput benchmark.
        
        Args:
            model: Model
            device: Device
        """
        self.model = model.to(device).eval()
        self.device = device
        
    def measure_throughput(
        self,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """
        Measure throughput for different batch sizes.
        
        Args:
            input_shape: Input shape (without batch)
            batch_sizes: Batch sizes to test
            num_iterations: Iterations per batch size
            warmup: Warmup iterations
            
        Returns:
            Throughput results per batch size
        """
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")
            
            # Create input
            input_tensor = torch.randn(batch_size, *input_shape[1:]).to(self.device)
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = self.model(input_tensor)
                    
            if self.device == 'cuda':
                torch.cuda.synchronize()
                
            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.time()
                
                with torch.no_grad():
                    _ = self.model(input_tensor)
                    
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                    
                times.append(time.time() - start)
                
            # Compute metrics
            times = np.array(times)
            mean_time = np.mean(times)
            throughput = batch_size / mean_time
            
            results[batch_size] = {
                'mean_time_s': mean_time,
                'std_time_s': np.std(times),
                'throughput_samples_per_sec': throughput,
                'latency_ms': mean_time * 1000 / batch_size
            }
            
        return results


class ComparisonBenchmark:
    """
    Compare multiple models.
    """
    
    @staticmethod
    def compare_models(
        models: Dict[str, nn.Module],
        input_shape: Tuple[int, ...],
        device: str = 'cpu'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of models
            input_shape: Input shape
            device: Device
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, model in models.items():
            print(f"\nProfiling {name}...")
            
            # Profile
            profiler = ModelProfiler(model)
            summary = profiler.profile_summary(input_shape)
            
            # Benchmark
            throughput_bench = ThroughputBenchmark(model, device)
            throughput = throughput_bench.measure_throughput(
                input_shape,
                batch_sizes=[1, 8],
                num_iterations=50,
                warmup=5
            )
            
            results[name] = {
                'parameters': summary['parameters']['total'],
                'gflops': summary['flops']['gflops'],
                'memory_mb': summary['memory']['total_mb'],
                'throughput_bs1': throughput[1]['throughput_samples_per_sec'],
                'throughput_bs8': throughput[8]['throughput_samples_per_sec'],
                'latency_ms': throughput[1]['latency_ms']
            }
            
        return results
        
    @staticmethod
    def print_comparison(results: Dict[str, Dict[str, Any]]):
        """
        Print comparison table.
        
        Args:
            results: Comparison results
        """
        print("\n" + "="*80)
        print("Model Comparison")
        print("="*80)
        
        # Header
        print(f"{'Model':<20} {'Params':>12} {'GFLOPs':>10} {'Memory':>10} {'Latency':>10} {'Throughput':>12}")
        print("-"*80)
        
        # Rows
        for name, metrics in results.items():
            print(
                f"{name:<20} "
                f"{metrics['parameters']:>12,} "
                f"{metrics['gflops']:>10.2f} "
                f"{metrics['memory_mb']:>10.1f} "
                f"{metrics['latency_ms']:>10.2f} "
                f"{metrics['throughput_bs1']:>12.1f}"
            )
            
        print("="*80)


def main():
    """Example usage."""
    print("="*60)
    print("Profiling and Benchmarking")
    print("="*60)
    
    # Dummy models
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(128, 256, 3, padding=1)
            self.fc = nn.Linear(256, 1000)
            
        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=-1)
            x = self.fc(x)
            return x
            
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(128, 256, 3, padding=1)
            self.conv2 = nn.Conv1d(256, 512, 3, padding=1)
            self.fc = nn.Linear(512, 1000)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.mean(dim=-1)
            x = self.fc(x)
            return x
            
    # Test 1: Model profiler
    print("\n1. Model Profiler")
    model = SmallModel()
    profiler = ModelProfiler(model)
    
    params = profiler.count_parameters()
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")
    
    flops = profiler.count_flops((1, 128, 500))
    print(f"   GFLOPs: {flops['gflops']:.2f}")
    
    memory = profiler.estimate_memory((1, 128, 500))
    print(f"   Memory: {memory['total_mb']:.2f} MB")
    
    # Test 2: Throughput benchmark
    print("\n2. Throughput Benchmark")
    throughput_bench = ThroughputBenchmark(model, device='cpu')
    
    results = throughput_bench.measure_throughput(
        (1, 128, 500),
        batch_sizes=[1, 4],
        num_iterations=20,
        warmup=5
    )
    
    for bs, metrics in results.items():
        print(f"   Batch {bs}: {metrics['throughput_samples_per_sec']:.1f} samples/sec")
        
    # Test 3: Model comparison
    print("\n3. Model Comparison")
    models = {
        'Small': SmallModel(),
        'Large': LargeModel()
    }
    
    comparison = ComparisonBenchmark.compare_models(
        models,
        (1, 128, 500),
        device='cpu'
    )
    
    ComparisonBenchmark.print_comparison(comparison)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
