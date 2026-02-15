"""
Model Pruning Utilities

Implements various pruning techniques:
- Magnitude-based pruning
- Structured pruning (channels, filters)
- Iterative pruning with fine-tuning
- Sensitivity-based pruning
- Lottery ticket hypothesis
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from copy import deepcopy


class MagnitudePruner:
    """
    Prune weights by magnitude.
    """
    
    def __init__(
        self,
        model: nn.Module,
        amount: float = 0.3,
        structured: bool = False
    ):
        """
        Initialize magnitude pruner.
        
        Args:
            model: Model to prune
            amount: Fraction of weights to prune (0.0-1.0)
            structured: Use structured (channel) pruning
        """
        self.model = model
        self.amount = amount
        self.structured = structured
        
    def prune(
        self,
        layer_types: Optional[List[type]] = None
    ) -> nn.Module:
        """
        Apply magnitude-based pruning.
        
        Args:
            layer_types: Layer types to prune (default: Conv, Linear)
            
        Returns:
            Pruned model
        """
        if layer_types is None:
            layer_types = [nn.Conv1d, nn.Conv2d, nn.Linear]
            
        pruned_modules = []
        
        for name, module in self.model.named_modules():
            if any(isinstance(module, t) for t in layer_types):
                if self.structured:
                    # Structured pruning (entire channels/filters)
                    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                        prune.ln_structured(
                            module,
                            name='weight',
                            amount=self.amount,
                            n=2,
                            dim=0  # Prune output channels
                        )
                    else:
                        # For linear layers, use unstructured
                        prune.l1_unstructured(
                            module,
                            name='weight',
                            amount=self.amount
                        )
                else:
                    # Unstructured pruning
                    prune.l1_unstructured(
                        module,
                        name='weight',
                        amount=self.amount
                    )
                    
                pruned_modules.append((name, module))
                
        print(f"✓ Pruned {len(pruned_modules)} modules")
        return self.model
        
    def make_permanent(self):
        """Make pruning permanent (remove pruning reparameterization)."""
        for module in self.model.modules():
            if prune.is_pruned(module):
                prune.remove(module, 'weight')
                
        print("✓ Made pruning permanent")
        
    def compute_sparsity(self) -> Dict[str, float]:
        """
        Compute sparsity of pruned model.
        
        Returns:
            Dictionary of layer sparsities
        """
        sparsities = {}
        total_params = 0
        total_zero = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                zero_params = (weight == 0).sum().item()
                total_params_layer = weight.numel()
                
                sparsity = zero_params / total_params_layer if total_params_layer > 0 else 0
                sparsities[name] = sparsity
                
                total_params += total_params_layer
                total_zero += zero_params
                
        sparsities['overall'] = total_zero / total_params if total_params > 0 else 0
        
        return sparsities


class IterativePruner:
    """
    Iterative pruning with fine-tuning.
    
    Gradually prune the model over multiple iterations,
    fine-tuning between each pruning step.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        num_iterations: int = 5
    ):
        """
        Initialize iterative pruner.
        
        Args:
            model: Model to prune
            target_sparsity: Target overall sparsity
            num_iterations: Number of pruning iterations
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.num_iterations = num_iterations
        
        # Compute per-iteration pruning amount
        self.per_iter_amount = 1 - (1 - target_sparsity) ** (1 / num_iterations)
        
    def prune_and_finetune(
        self,
        train_fn: callable,
        val_fn: callable,
        layer_types: Optional[List[type]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prune model iteratively with fine-tuning.
        
        Args:
            train_fn: Function to train/finetune model
            val_fn: Function to validate model
            layer_types: Layer types to prune
            
        Returns:
            List of iteration results
        """
        results = []
        
        for iteration in range(self.num_iterations):
            print(f"\nIteration {iteration + 1}/{self.num_iterations}")
            print("-" * 40)
            
            # Prune
            pruner = MagnitudePruner(self.model, amount=self.per_iter_amount)
            pruner.prune(layer_types=layer_types)
            
            # Compute sparsity
            sparsities = pruner.compute_sparsity()
            print(f"Overall sparsity: {sparsities['overall']:.2%}")
            
            # Validate before fine-tuning
            val_loss_before = val_fn(self.model)
            print(f"Val loss before fine-tuning: {val_loss_before:.4f}")
            
            # Fine-tune
            train_loss = train_fn(self.model)
            print(f"Training loss: {train_loss:.4f}")
            
            # Validate after fine-tuning
            val_loss_after = val_fn(self.model)
            print(f"Val loss after fine-tuning: {val_loss_after:.4f}")
            
            results.append({
                'iteration': iteration + 1,
                'sparsity': sparsities['overall'],
                'val_loss_before': val_loss_before,
                'train_loss': train_loss,
                'val_loss_after': val_loss_after
            })
            
        # Make pruning permanent
        pruner = MagnitudePruner(self.model, amount=0)
        pruner.make_permanent()
        
        return results


class SensitivityPruner:
    """
    Prune based on layer sensitivity.
    
    Different layers have different sensitivities to pruning.
    This pruner assigns different pruning amounts to each layer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5
    ):
        """
        Initialize sensitivity-based pruner.
        
        Args:
            model: Model to prune
            target_sparsity: Overall target sparsity
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.sensitivities = {}
        
    def compute_sensitivity(
        self,
        val_fn: callable,
        probe_amount: float = 0.1
    ) -> Dict[str, float]:
        """
        Compute sensitivity of each layer.
        
        Args:
            val_fn: Validation function (returns loss)
            probe_amount: Amount to prune for sensitivity test
            
        Returns:
            Layer sensitivities (higher = more sensitive)
        """
        print("Computing layer sensitivities...")
        
        # Baseline performance
        baseline_loss = val_fn(self.model)
        
        # Test each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                # Save original weights
                orig_weight = module.weight.data.clone()
                
                # Prune layer
                prune.l1_unstructured(module, name='weight', amount=probe_amount)
                
                # Measure performance drop
                loss = val_fn(self.model)
                sensitivity = loss - baseline_loss
                
                self.sensitivities[name] = sensitivity
                
                # Restore weights
                prune.remove(module, 'weight')
                module.weight.data = orig_weight
                
        print(f"✓ Computed sensitivities for {len(self.sensitivities)} layers")
        
        return self.sensitivities
        
    def prune(
        self,
        sensitivities: Optional[Dict[str, float]] = None
    ) -> nn.Module:
        """
        Prune model based on sensitivity.
        
        Less sensitive layers are pruned more aggressively.
        
        Args:
            sensitivities: Pre-computed sensitivities (optional)
            
        Returns:
            Pruned model
        """
        if sensitivities is None:
            sensitivities = self.sensitivities
            
        if not sensitivities:
            raise ValueError("No sensitivities available. Run compute_sensitivity first.")
            
        # Normalize sensitivities
        max_sens = max(sensitivities.values())
        min_sens = min(sensitivities.values())
        
        for name, module in self.model.named_modules():
            if name in sensitivities:
                sens = sensitivities[name]
                
                # Normalized sensitivity (0-1, lower = less sensitive)
                norm_sens = (sens - min_sens) / (max_sens - min_sens + 1e-9)
                
                # Prune amount inversely proportional to sensitivity
                # Less sensitive layers get pruned more
                amount = self.target_sparsity * (1 - norm_sens)
                amount = max(0.0, min(0.9, amount))  # Clip to [0, 0.9]
                
                if amount > 0:
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    
        print("✓ Applied sensitivity-based pruning")
        
        return self.model


class StructuredPruner:
    """
    Structured pruning - remove entire channels/filters.
    
    Provides actual speedup (compared to unstructured pruning).
    """
    
    def __init__(
        self,
        model: nn.Module,
        amount: float = 0.3
    ):
        """
        Initialize structured pruner.
        
        Args:
            model: Model to prune
            amount: Fraction of channels to prune
        """
        self.model = model
        self.amount = amount
        
    def prune_conv_channels(
        self,
        conv_module: nn.Module,
        amount: float
    ) -> nn.Module:
        """
        Prune entire channels from conv layer.
        
        Args:
            conv_module: Conv layer
            amount: Fraction to prune
            
        Returns:
            Pruned module
        """
        # Get channel importances (L1 norm)
        weight = conv_module.weight.data
        num_channels = weight.size(0)
        
        channel_norms = weight.view(num_channels, -1).norm(p=1, dim=1)
        
        # Number of channels to keep
        num_keep = int(num_channels * (1 - amount))
        
        # Get top channels
        _, keep_indices = channel_norms.topk(num_keep)
        keep_indices = sorted(keep_indices.tolist())
        
        # Create new conv layer
        new_conv = nn.Conv1d(
            conv_module.in_channels,
            num_keep,
            conv_module.kernel_size,
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            groups=conv_module.groups,
            bias=conv_module.bias is not None
        )
        
        # Copy weights
        new_conv.weight.data = weight[keep_indices]
        if conv_module.bias is not None:
            new_conv.bias.data = conv_module.bias.data[keep_indices]
            
        return new_conv, keep_indices
        
    def prune(self) -> nn.Module:
        """
        Apply structured pruning to all conv layers.
        
        Returns:
            Pruned model
        """
        pruned_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # Prune channels
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=self.amount,
                    n=2,
                    dim=0
                )
                pruned_count += 1
                
        print(f"✓ Applied structured pruning to {pruned_count} layers")
        
        return self.model


class LotteryTicketPruner:
    """
    Lottery Ticket Hypothesis pruner.
    
    Find winning lottery tickets by:
    1. Train model
    2. Prune small weights
    3. Reset remaining weights to initialization
    4. Retrain
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5
    ):
        """
        Initialize lottery ticket pruner.
        
        Args:
            model: Model to prune
            target_sparsity: Target sparsity
        """
        self.model = model
        self.target_sparsity = target_sparsity
        
        # Save initial weights
        self.initial_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        
    def find_ticket(
        self,
        train_fn: callable,
        prune_fn: Optional[callable] = None
    ) -> nn.Module:
        """
        Find winning lottery ticket.
        
        Args:
            train_fn: Training function
            prune_fn: Custom pruning function (default: magnitude)
            
        Returns:
            Pruned model with reset weights
        """
        # Train model
        print("Training full model...")
        train_fn(self.model)
        
        # Prune
        if prune_fn is None:
            print(f"Pruning to {self.target_sparsity:.0%} sparsity...")
            pruner = MagnitudePruner(self.model, amount=self.target_sparsity)
            pruner.prune()
        else:
            prune_fn(self.model)
            
        # Create mask
        masks = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_mask'):
                masks[name] = module.weight_mask.clone()
                
        # Reset weights to initialization
        print("Resetting weights to initialization...")
        for name, param in self.model.named_parameters():
            if name in self.initial_weights:
                param.data = self.initial_weights[name].clone()
                
        # Reapply masks
        for name, module in self.model.named_modules():
            if name in masks:
                module.weight_mask = masks[name]
                
        print("✓ Found lottery ticket")
        
        return self.model


def main():
    """Example usage."""
    print("="*60)
    print("Model Pruning")
    print("="*60)
    
    # Dummy model
    class DummyModel(nn.Module):
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
            
    # Test 1: Magnitude pruning
    print("\n1. Magnitude Pruning")
    model = DummyModel()
    
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"   Original parameters: {orig_params:,}")
    
    pruner = MagnitudePruner(model, amount=0.3, structured=False)
    pruner.prune()
    
    sparsities = pruner.compute_sparsity()
    print(f"   Overall sparsity: {sparsities['overall']:.2%}")
    
    # Test 2: Structured pruning
    print("\n2. Structured Pruning")
    model2 = DummyModel()
    
    struct_pruner = StructuredPruner(model2, amount=0.3)
    struct_pruner.prune()
    
    print(f"   Applied structured pruning")
    
    # Test 3: Iterative pruning (dummy)
    print("\n3. Iterative Pruning")
    model3 = DummyModel()
    
    def dummy_train(model):
        return 0.5  # Dummy loss
        
    def dummy_val(model):
        return 0.6  # Dummy loss
        
    iter_pruner = IterativePruner(model3, target_sparsity=0.5, num_iterations=3)
    
    print("   Would prune iteratively with fine-tuning")
    print(f"   Per-iteration amount: {iter_pruner.per_iter_amount:.2%}")
    
    # Test 4: Sensitivity pruning
    print("\n4. Sensitivity Pruning")
    model4 = DummyModel()
    
    sens_pruner = SensitivityPruner(model4, target_sparsity=0.5)
    sensitivities = sens_pruner.compute_sensitivity(dummy_val, probe_amount=0.1)
    
    print(f"   Computed sensitivities for {len(sensitivities)} layers")
    for name, sens in list(sensitivities.items())[:3]:
        print(f"   {name}: {sens:.4f}")
        
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
