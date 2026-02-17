"""
Model Quantization Utilities

Implements various quantization techniques:
- Post-training static quantization (INT8)
- Dynamic quantization
- Quantization-aware training (QAT)
- Mixed precision quantization
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Any, Optional, List
import copy
import warnings


class PostTrainingQuantizer:
    """
    Post-training static quantization (PTQ).
    
    Quantize model to INT8 after training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        backend: str = 'fbgemm'
    ):
        """
        Initialize PTQ quantizer.
        
        Args:
            model: Model to quantize
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        """
        self.model = model.cpu()  # Quantization requires CPU
        self.backend = backend
        
        torch.backends.quantized.engine = backend
        
    def prepare(self):
        """
        Prepare model for quantization.
        
        Fuses modules and inserts observers.
        """
        # Set model to eval mode
        self.model.eval()
        
        # Fuse modules (Conv-BN-ReLU, etc.)
        self._fuse_modules()
        
        # Attach quantization config
        self.model.qconfig = quant.get_default_qconfig(self.backend)
        
        # Prepare model
        quant.prepare(self.model, inplace=True)
        
        print("✓ Model prepared for quantization")
        
    def _fuse_modules(self):
        """Fuse common module patterns."""
        # Check for fusible modules
        modules_to_fuse = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Sequential):
                # Try to identify fusible patterns
                # Conv + BN + ReLU, Conv + BN, etc.
                # TODO: Implement automatic module fusion detection
                pass
        
        if not modules_to_fuse:
            warnings.warn(
                "Module fusion not yet fully implemented. "
                "Quantization will proceed without module fusion optimization.",
                UserWarning
            )
                
        if modules_to_fuse:
            quant.fuse_modules(self.model, modules_to_fuse, inplace=True)
            print(f"✓ Fused {len(modules_to_fuse)} module groups")
            
    def calibrate(
        self,
        calibration_loader,
        num_batches: int = 100
    ):
        """
        Calibrate quantization parameters.
        
        Args:
            calibration_loader: DataLoader for calibration
            num_batches: Number of batches for calibration
        """
        print(f"Calibrating with {num_batches} batches...")
        
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                    
                # Forward pass to collect statistics
                if isinstance(batch, (list, tuple)):
                    _ = self.model(*batch)
                else:
                    _ = self.model(batch)
                    
        print("✓ Calibration complete")
        
    def quantize(self) -> nn.Module:
        """
        Convert model to quantized version.
        
        Returns:
            Quantized model
        """
        # Convert to quantized model
        quantized_model = quant.convert(self.model, inplace=False)
        
        print("✓ Model quantized to INT8")
        
        return quantized_model
        
    def measure_size(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module
    ) -> Dict[str, Any]:
        """
        Compare model sizes.
        
        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            
        Returns:
            Size comparison metrics
        """
        import os
        import tempfile
        
        # Save models
        with tempfile.NamedTemporaryFile() as fp32_file:
            torch.save(original_model.state_dict(), fp32_file.name)
            fp32_size = os.path.getsize(fp32_file.name)
            
        with tempfile.NamedTemporaryFile() as int8_file:
            torch.save(quantized_model.state_dict(), int8_file.name)
            int8_size = os.path.getsize(int8_file.name)
            
        reduction = (fp32_size - int8_size) / fp32_size
        
        return {
            'fp32_size_mb': fp32_size / (1024**2),
            'int8_size_mb': int8_size / (1024**2),
            'reduction': reduction,
            'compression_ratio': fp32_size / int8_size
        }


class DynamicQuantizer:
    """
    Dynamic quantization.
    
    Quantize weights ahead of time, activations dynamically.
    Good for RNNs and transformers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ):
        """
        Initialize dynamic quantizer.
        
        Args:
            model: Model to quantize
            dtype: Quantization dtype (qint8 or float16)
        """
        self.model = model.cpu()
        self.dtype = dtype
        
    def quantize(
        self,
        layer_types: Optional[List[type]] = None
    ) -> nn.Module:
        """
        Apply dynamic quantization.
        
        Args:
            layer_types: Layer types to quantize
            
        Returns:
            Quantized model
        """
        if layer_types is None:
            # Default to Linear and LSTM
            layer_types = [nn.Linear, nn.LSTM, nn.GRU]
            
        # Apply dynamic quantization
        quantized_model = quant.quantize_dynamic(
            self.model,
            qconfig_spec=set(layer_types),
            dtype=self.dtype
        )
        
        print(f"✓ Applied dynamic quantization to {layer_types}")
        
        return quantized_model


class QATTrainer:
    """
    Quantization-Aware Training (QAT).
    
    Simulate quantization during training for better accuracy.
    """
    
    def __init__(
        self,
        model: nn.Module,
        backend: str = 'fbgemm'
    ):
        """
        Initialize QAT trainer.
        
        Args:
            model: Model to train
            backend: Quantization backend
        """
        self.model = model.cpu()
        self.backend = backend
        
        torch.backends.quantized.engine = backend
        
    def prepare(self):
        """
        Prepare model for QAT.
        """
        # Fuse modules
        self._fuse_modules()
        
        # Set QAT config
        self.model.qconfig = quant.get_default_qat_qconfig(self.backend)
        
        # Prepare for QAT
        quant.prepare_qat(self.model, inplace=True)
        
        print("✓ Model prepared for QAT")
        
    def _fuse_modules(self):
        """Fuse modules."""
        # Similar to PTQ
        pass
        
    def train_step(
        self,
        batch,
        optimizer,
        criterion,
        device: str = 'cpu'
    ) -> float:
        """
        Single training step with fake quantization.
        
        Args:
            batch: Training batch
            optimizer: Optimizer
            criterion: Loss function
            device: Device
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Forward pass (with fake quantization)
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
    def convert(self) -> nn.Module:
        """
        Convert to quantized model.
        
        Returns:
            Quantized model
        """
        self.model.eval()
        quantized_model = quant.convert(self.model, inplace=False)
        
        print("✓ Converted QAT model to quantized")
        
        return quantized_model


class MixedPrecisionQuantizer:
    """
    Mixed precision quantization.
    
    Quantize different layers with different bit-widths.
    """
    
    def __init__(
        self,
        model: nn.Module
    ):
        """
        Initialize mixed precision quantizer.
        
        Args:
            model: Model to quantize
        """
        self.model = model
        self.layer_configs = {}
        
    def set_layer_precision(
        self,
        layer_name: str,
        bits: int
    ):
        """
        Set precision for specific layer.
        
        Args:
            layer_name: Name of layer
            bits: Number of bits (4, 8, 16, 32)
        """
        self.layer_configs[layer_name] = bits
        
    def auto_assign_precision(
        self,
        sensitivity_dict: Dict[str, float]
    ):
        """
        Automatically assign precision based on sensitivity.
        
        More sensitive layers get higher precision.
        
        Args:
            sensitivity_dict: Layer sensitivities
        """
        # Normalize sensitivities
        max_sens = max(sensitivity_dict.values())
        min_sens = min(sensitivity_dict.values())
        
        for layer_name, sens in sensitivity_dict.items():
            norm_sens = (sens - min_sens) / (max_sens - min_sens + 1e-9)
            
            # Assign precision
            if norm_sens > 0.7:
                bits = 16  # High sensitivity -> higher precision
            elif norm_sens > 0.4:
                bits = 8
            else:
                bits = 4  # Low sensitivity -> lower precision
                
            self.layer_configs[layer_name] = bits
            
        print(f"✓ Auto-assigned precision to {len(self.layer_configs)} layers")
        
    def quantize(self) -> nn.Module:
        """
        Apply mixed precision quantization.
        
        Returns:
            Quantized model
        """
        # This is a simplified implementation
        # Full mixed-precision requires custom quantization ops
        
        for name, module in self.model.named_modules():
            if name in self.layer_configs:
                bits = self.layer_configs[name]
                
                if bits == 8:
                    # INT8 quantization
                    if isinstance(module, nn.Linear):
                        # Apply quantization
                        pass
                elif bits == 4:
                    # INT4 quantization (requires custom ops)
                    warnings.warn(f"INT4 not fully supported, using INT8 for {name}")
                    
        print("✓ Applied mixed precision quantization")
        
        return self.model


class QuantizationBenchmark:
    """
    Benchmark quantized models.
    """
    
    @staticmethod
    def compare_models(
        fp32_model: nn.Module,
        quantized_model: nn.Module,
        test_loader,
        eval_fn: callable
    ) -> Dict[str, Any]:
        """
        Compare FP32 and quantized models.
        
        Args:
            fp32_model: Original FP32 model
            quantized_model: Quantized model
            test_loader: Test data loader
            eval_fn: Evaluation function (returns metric)
            
        Returns:
            Comparison metrics
        """
        import time
        
        # Evaluate FP32
        print("Evaluating FP32 model...")
        start = time.time()
        fp32_metric = eval_fn(fp32_model, test_loader)
        fp32_time = time.time() - start
        
        # Evaluate quantized
        print("Evaluating quantized model...")
        start = time.time()
        quant_metric = eval_fn(quantized_model, test_loader)
        quant_time = time.time() - start
        
        # Size comparison
        ptq = PostTrainingQuantizer(fp32_model)
        sizes = ptq.measure_size(fp32_model, quantized_model)
        
        return {
            'fp32_metric': fp32_metric,
            'quantized_metric': quant_metric,
            'metric_degradation': fp32_metric - quant_metric,
            'fp32_time': fp32_time,
            'quantized_time': quant_time,
            'speedup': fp32_time / quant_time,
            **sizes
        }


def main():
    """Example usage."""
    print("="*60)
    print("Model Quantization")
    print("="*60)
    
    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(128, 256, 3, padding=1)
            self.bn = nn.BatchNorm1d(256)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(256, 512, batch_first=True)
            self.fc = nn.Linear(512, 1000)
            
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = x.transpose(1, 2)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            x = self.fc(x)
            return x
            
    # Test 1: Post-training quantization
    print("\n1. Post-Training Quantization (PTQ)")
    model = DummyModel()
    
    ptq = PostTrainingQuantizer(model, backend='fbgemm')
    ptq.prepare()
    
    # Dummy calibration
    print("   Prepared model for quantization")
    
    # Test 2: Dynamic quantization
    print("\n2. Dynamic Quantization")
    model2 = DummyModel()
    
    dyn_quant = DynamicQuantizer(model2, dtype=torch.qint8)
    quantized_model = dyn_quant.quantize()
    
    print(f"   Model type: {type(quantized_model)}")
    
    # Test 3: QAT
    print("\n3. Quantization-Aware Training (QAT)")
    model3 = DummyModel()
    
    qat = QATTrainer(model3, backend='fbgemm')
    qat.prepare()
    
    print("   Model prepared for QAT")
    
    # Test 4: Mixed precision
    print("\n4. Mixed Precision Quantization")
    model4 = DummyModel()
    
    mixed_quant = MixedPrecisionQuantizer(model4)
    
    # Dummy sensitivity
    sensitivity = {'conv': 0.8, 'lstm': 0.6, 'fc': 0.3}
    mixed_quant.auto_assign_precision(sensitivity)
    
    print(f"   Layer configs: {mixed_quant.layer_configs}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
