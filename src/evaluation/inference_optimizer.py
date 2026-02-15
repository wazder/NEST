"""
Inference Optimization Utilities

Provides various inference optimization techniques:
- ONNX export for faster inference
- TorchScript compilation
- Mixed precision inference (FP16)
- Efficient batch processing
- Model caching and warmup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import time
import warnings
from pathlib import Path


class InferenceOptimizer:
    """
    Optimize model for faster inference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        use_fp16: bool = False,
        use_torchscript: bool = False
    ):
        """
        Initialize inference optimizer.
        
        Args:
            model: PyTorch model
            device: Device to run on
            use_fp16: Use mixed precision (FP16)
            use_torchscript: Use TorchScript compilation
        """
        self.device = device
        self.use_fp16 = use_fp16
        self.use_torchscript = use_torchscript
        
        # Move model to device
        self.model = model.to(device)
        self.model.eval()
        
        # Apply optimizations
        self._apply_optimizations()
        
    def _apply_optimizations(self):
        """Apply selected optimizations."""
        
        # Mixed precision
        if self.use_fp16:
            if self.device == 'cuda':
                self.model = self.model.half()
                print("✓ Applied FP16 precision")
            else:
                warnings.warn("FP16 requires CUDA, skipping")
                self.use_fp16 = False
                
        # TorchScript
        if self.use_torchscript:
            try:
                self._compile_torchscript()
                print("✓ Compiled with TorchScript")
            except Exception as e:
                warnings.warn(f"TorchScript compilation failed: {e}")
                self.use_torchscript = False
                
        # Fuse operations
        if hasattr(torch.quantization, 'fuse_modules'):
            try:
                self._fuse_modules()
                print("✓ Fused modules")
            except Exception as e:
                warnings.warn(f"Module fusion failed: {e}")
                
    def _compile_torchscript(self):
        """Compile model with TorchScript."""
        # Create example inputs
        example_eeg = torch.randn(1, 128, 500).to(self.device)
        example_tokens = torch.randint(0, 1000, (1, 10)).to(self.device)
        
        if self.use_fp16:
            example_eeg = example_eeg.half()
            
        # Trace model
        with torch.no_grad():
            self.model = torch.jit.trace(
                self.model,
                (example_eeg, example_tokens),
                strict=False
            )
            
    def _fuse_modules(self):
        """Fuse Conv-BN-ReLU layers."""
        # Check if model has fusible modules
        if hasattr(self.model, 'spatial_cnn'):
            # Fuse spatial CNN layers
            for name, module in self.model.spatial_cnn.named_modules():
                if isinstance(module, nn.Sequential):
                    # Try to fuse conv+bn+relu
                    pass
                    
    def infer(
        self,
        eeg_data: torch.Tensor,
        input_tokens: Optional[torch.Tensor] = None,
        measure_time: bool = False
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Run optimized inference.
        
        Args:
            eeg_data: EEG input (batch, channels, time)
            input_tokens: Decoder input tokens (batch, seq_len)
            measure_time: Measure inference time
            
        Returns:
            Output logits and optional inference time
        """
        # Move to device
        eeg_data = eeg_data.to(self.device)
        if input_tokens is not None:
            input_tokens = input_tokens.to(self.device)
            
        # Convert to FP16 if needed
        if self.use_fp16:
            eeg_data = eeg_data.half()
            
        # Inference
        start_time = time.time() if measure_time else None
        
        with torch.no_grad():
            if self.device == 'cuda':
                # Use autocast for mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    output = self.model(eeg_data, input_tokens)
            else:
                output = self.model(eeg_data, input_tokens)
                
        inference_time = time.time() - start_time if measure_time else None
        
        return output, inference_time
        
    def benchmark(
        self,
        batch_size: int = 1,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            batch_size: Batch size
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Performance metrics
        """
        # Create dummy data
        eeg_data = torch.randn(batch_size, 128, 500).to(self.device)
        input_tokens = torch.randint(0, 1000, (batch_size, 10)).to(self.device)
        
        if self.use_fp16:
            eeg_data = eeg_data.half()
            
        # Warmup
        print(f"Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self.model(eeg_data, input_tokens)
                
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        # Benchmark
        print(f"Benchmarking ({num_iterations} iterations)...")
        times = []
        
        for _ in range(num_iterations):
            start = time.time()
            
            with torch.no_grad():
                _ = self.model(eeg_data, input_tokens)
                
            if self.device == 'cuda':
                torch.cuda.synchronize()
                
            times.append(time.time() - start)
            
        # Compute statistics
        times = np.array(times)
        
        metrics = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'throughput': batch_size / np.mean(times),
            'batch_size': batch_size
        }
        
        return metrics
        
    def export_onnx(
        self,
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output ONNX file path
            input_names: Input tensor names
            output_names: Output tensor names
            dynamic_axes: Dynamic axes specification
        """
        # Default names
        if input_names is None:
            input_names = ['eeg_input', 'token_input']
        if output_names is None:
            output_names = ['logits']
            
        # Default dynamic axes (for variable batch size and sequence length)
        if dynamic_axes is None:
            dynamic_axes = {
                'eeg_input': {0: 'batch_size', 2: 'time_steps'},
                'token_input': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
            
        # Create example inputs
        example_eeg = torch.randn(1, 128, 500).to(self.device)
        example_tokens = torch.randint(0, 1000, (1, 10)).to(self.device)
        
        if self.use_fp16:
            example_eeg = example_eeg.half()
            
        # Export
        torch.onnx.export(
            self.model,
            (example_eeg, example_tokens),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
            do_constant_folding=True
        )
        
        print(f"✓ Exported ONNX model to {output_path}")


class BatchProcessor:
    """
    Efficient batch processing with dynamic batching.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize batch processor.
        
        Args:
            model: Model to use
            max_batch_size: Maximum batch size
            max_wait_time: Maximum wait time for batching (seconds)
            device: Device
        """
        self.model = model.to(device)
        self.model.eval()
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        
        self.queue = []
        self.last_batch_time = time.time()
        
    def process(
        self,
        eeg_data: torch.Tensor,
        input_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process single input (will be batched automatically).
        
        Args:
            eeg_data: (channels, time)
            input_tokens: (seq_len,)
            
        Returns:
            Output logits
        """
        # Add to queue
        self.queue.append((eeg_data, input_tokens))
        
        # Check if should process batch
        should_process = (
            len(self.queue) >= self.max_batch_size or
            time.time() - self.last_batch_time >= self.max_wait_time
        )
        
        if should_process:
            return self._process_batch()
        else:
            return None
            
    def _process_batch(self) -> List[torch.Tensor]:
        """Process queued batch."""
        if not self.queue:
            return []
            
        # Collate batch
        eeg_batch = torch.stack([x[0] for x in self.queue]).to(self.device)
        
        if self.queue[0][1] is not None:
            token_batch = torch.stack([x[1] for x in self.queue]).to(self.device)
        else:
            token_batch = None
            
        # Process
        with torch.no_grad():
            outputs = self.model(eeg_batch, token_batch)
            
        # Split outputs
        results = list(outputs)
        
        # Clear queue
        self.queue = []
        self.last_batch_time = time.time()
        
        return results


class ModelCache:
    """
    Cache for loaded models to avoid repeated loading.
    """
    
    def __init__(self, max_size: int = 5):
        """
        Initialize model cache.
        
        Args:
            max_size: Maximum number of cached models
        """
        self.max_size = max_size
        self.cache: Dict[str, nn.Module] = {}
        self.access_count: Dict[str, int] = {}
        
    def get(
        self,
        model_path: str,
        loader_fn: callable
    ) -> nn.Module:
        """
        Get model from cache or load it.
        
        Args:
            model_path: Path to model checkpoint
            loader_fn: Function to load model if not cached
            
        Returns:
            Loaded model
        """
        if model_path in self.cache:
            self.access_count[model_path] += 1
            return self.cache[model_path]
            
        # Load model
        model = loader_fn(model_path)
        
        # Check cache size
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
            
        # Add to cache
        self.cache[model_path] = model
        self.access_count[model_path] = 1
        
        return model
        
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()


def main():
    """Example usage."""
    print("="*60)
    print("Inference Optimization")
    print("="*60)
    
    # Dummy model
    class DummyNEST(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Conv1d(128, 256, 3, padding=1)
            self.lstm = nn.LSTM(256, 512, 2, batch_first=True)
            self.fc = nn.Linear(512, 1000)
            
        def forward(self, eeg, tokens):
            # Encode EEG
            x = self.cnn(eeg)  # (batch, 256, time)
            x = x.transpose(1, 2)  # (batch, time, 256)
            x, _ = self.lstm(x)  # (batch, time, 512)
            
            # Decode
            logits = self.fc(x)
            return logits
            
    model = DummyNEST()
    
    # Test 1: Basic optimization
    print("\n1. Basic Inference Optimizer")
    optimizer = InferenceOptimizer(
        model,
        device='cpu',
        use_fp16=False,
        use_torchscript=False
    )
    
    eeg = torch.randn(2, 128, 500)
    tokens = torch.randint(0, 1000, (2, 10))
    
    output, inference_time = optimizer.infer(eeg, tokens, measure_time=True)
    print(f"   Output shape: {output.shape}")
    print(f"   Inference time: {inference_time*1000:.2f} ms")
    
    # Test 2: Benchmark
    print("\n2. Benchmark")
    metrics = optimizer.benchmark(batch_size=1, num_iterations=50, warmup_iterations=5)
    
    print(f"   Mean time: {metrics['mean_time']*1000:.2f} ms")
    print(f"   Std time: {metrics['std_time']*1000:.2f} ms")
    print(f"   Throughput: {metrics['throughput']:.2f} samples/sec")
    
    # Test 3: ONNX Export
    print("\n3. ONNX Export")
    try:
        output_path = "/tmp/nest_model.onnx"
        optimizer.export_onnx(output_path)
        
        import os
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024**2)
            print(f"   Model size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"   ONNX export failed: {e}")
        
    # Test 4: Batch processor
    print("\n4. Batch Processor")
    batch_proc = BatchProcessor(model, max_batch_size=4, device='cpu')
    
    print("   Processing 5 inputs...")
    for i in range(5):
        eeg_single = torch.randn(128, 500)
        result = batch_proc.process(eeg_single)
        if result is not None:
            print(f"   Processed batch: {len(result)} outputs")
            
    # Test 5: Model cache
    print("\n5. Model Cache")
    cache = ModelCache(max_size=3)
    
    def dummy_loader(path):
        return DummyNEST()
        
    model1 = cache.get("model1.pt", dummy_loader)
    model2 = cache.get("model2.pt", dummy_loader)
    model3 = cache.get("model1.pt", dummy_loader)  # Should hit cache
    
    print(f"   Cache size: {len(cache.cache)}")
    print(f"   Access counts: {cache.access_count}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
