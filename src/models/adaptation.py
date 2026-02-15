"""
Subject Adaptation and Domain Adaptation for NEST

This module implements techniques for cross-subject generalization:
- Subject embedding layer
- Domain adaptation (DANN, CORAL)
- Fine-tuning strategies
- Subject-invariant feature learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class SubjectEmbedding(nn.Module):
    """
    Subject-specific embedding layer.
    
    Adds subject-specific parameters to help model adapt to individual differences.
    """
    
    def __init__(
        self,
        num_subjects: int,
        embedding_dim: int,
        feature_dim: int
    ):
        """
        Initialize subject embedding.
        
        Args:
            num_subjects: Number of subjects
            embedding_dim: Dimension of subject embedding
            feature_dim: Dimension of EEG features
        """
        super().__init__()
        
        self.num_subjects = num_subjects
        self.embedding_dim = embedding_dim
        
        # Subject embedding lookup
        self.subject_embeddings = nn.Embedding(num_subjects, embedding_dim)
        
        # Projection to feature space
        self.projection = nn.Linear(embedding_dim, feature_dim)
        
    def forward(
        self,
        features: torch.Tensor,
        subject_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Add subject-specific information to features.
        
        Args:
            features: (batch, seq_len, feature_dim)
            subject_ids: (batch,) - subject IDs
            
        Returns:
            Modified features with subject information
        """
        # Get subject embeddings
        subj_emb = self.subject_embeddings(subject_ids)  # (batch, embedding_dim)
        
        # Project to feature space
        subj_features = self.projection(subj_emb)  # (batch, feature_dim)
        
        # Add to features (broadcast over sequence)
        subj_features = subj_features.unsqueeze(1)  # (batch, 1, feature_dim)
        
        return features + subj_features


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for Domain-Adversarial Neural Networks (DANN).
    
    Ganin et al. (2016): "Domain-Adversarial Training of Neural Networks"
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DomainAdversarialNetwork(nn.Module):
    """
    Domain-Adversarial Neural Network for subject adaptation.
    
    Encourages learning of subject-invariant features.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_subjects: int,
        hidden_dim: int = 256,
        lambda_: float = 1.0
    ):
        """
        Initialize DANN.
        
        Args:
            feature_dim: Dimension of features
            num_subjects: Number of subjects (domains)
            hidden_dim: Hidden layer dimension
            lambda_: Gradient reversal weight
        """
        super().__init__()
        
        self.lambda_ = lambda_
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_subjects)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict domain (subject) from features.
        
        Args:
            features: (batch, feature_dim) or (batch, seq_len, feature_dim)
            
        Returns:
            Domain predictions: (batch, num_subjects)
        """
        # If 3D, pool over sequence
        if features.dim() == 3:
            features = features.mean(dim=1)  # (batch, feature_dim)
            
        # Apply gradient reversal
        reversed_features = GradientReversalLayer.apply(features, self.lambda_)
        
        # Classify domain
        domain_logits = self.domain_classifier(reversed_features)
        
        return domain_logits
        
    def set_lambda(self, lambda_: float):
        """Update gradient reversal weight."""
        self.lambda_ = lambda_


class CORAL(nn.Module):
    """
    CORAL (CORrelation ALignment) for domain adaptation.
    
    Sun & Saenko (2016): "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
    """
    
    def __init__(self):
        """Initialize CORAL."""
        super().__init__()
        
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CORAL loss.
        
        Args:
            source_features: (batch, feature_dim)
            target_features: (batch, feature_dim)
            
        Returns:
            CORAL loss (scalar)
        """
        # Compute covariance matrices
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)
        
        # Frobenius norm of difference
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        loss = loss / (4 * source_features.size(1) ** 2)
        
        return loss
        
    def _compute_covariance(self, features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix."""
        n = features.size(0)
        
        # Center features
        features = features - features.mean(dim=0, keepdim=True)
        
        # Compute covariance
        cov = (features.t() @ features) / (n - 1)
        
        return cov


class SubjectAdaptiveBatchNorm(nn.Module):
    """
    Subject-Adaptive Batch Normalization.
    
    Each subject has its own batch norm parameters (shift/scale).
    """
    
    def __init__(
        self,
        num_subjects: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        """
        Initialize subject-adaptive batch norm.
        
        Args:
            num_subjects: Number of subjects
            num_features: Number of features
            eps: Epsilon for numerical stability
            momentum: Momentum for running statistics
        """
        super().__init__()
        
        self.num_subjects = num_subjects
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Shared running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # Subject-specific affine parameters
        self.weight = nn.Parameter(torch.ones(num_subjects, num_features))
        self.bias = nn.Parameter(torch.zeros(num_subjects, num_features))
        
    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, num_features) or (batch, num_features, ...)
            subject_ids: (batch,) - subject IDs
            
        Returns:
            Normalized features
        """
        if self.training:
            # Compute batch statistics
            if x.dim() > 2:
                # Flatten spatial dimensions
                batch_size = x.size(0)
                x_flat = x.view(batch_size, self.num_features, -1)
                mean = x_flat.mean(dim=[0, 2])
                var = x_flat.var(dim=[0, 2], unbiased=False)
            else:
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
                
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        # Normalize
        x_norm = (x - mean.view(1, -1, *([1] * (x.dim() - 2)))) / torch.sqrt(var.view(1, -1, *([1] * (x.dim() - 2))) + self.eps)
        
        # Apply subject-specific affine transform
        weight = self.weight[subject_ids]  # (batch, num_features)
        bias = self.bias[subject_ids]  # (batch, num_features)
        
        # Reshape for broadcasting
        weight = weight.view(x.size(0), -1, *([1] * (x.dim() - 2)))
        bias = bias.view(x.size(0), -1, *([1] * (x.dim() - 2)))
        
        return x_norm * weight + bias


class FineTuningStrategy:
    """
    Strategies for fine-tuning on new subjects.
    """
    
    @staticmethod
    def freeze_backbone(model: nn.Module, unfreeze_last_n: int = 0):
        """
        Freeze backbone and optionally unfreeze last N layers.
        
        Args:
            model: PyTorch model
            unfreeze_last_n: Number of last layers to unfreeze
        """
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze last N layers
        if unfreeze_last_n > 0:
            layers = list(model.modules())[-unfreeze_last_n:]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = True
                    
    @staticmethod
    def progressive_unfreezing(
        model: nn.Module,
        epoch: int,
        total_epochs: int,
        num_layers: int
    ):
        """
        Progressively unfreeze layers during training.
        
        Args:
            model: PyTorch model
            epoch: Current epoch
            total_epochs: Total epochs
            num_layers: Total number of layers
        """
        # Calculate how many layers to unfreeze
        progress = epoch / total_epochs
        layers_to_unfreeze = int(progress * num_layers)
        
        # Get all layers
        all_layers = list(model.modules())
        
        # Unfreeze from the end
        for i, layer in enumerate(reversed(all_layers)):
            if i < layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
                    
    @staticmethod
    def discriminative_learning_rates(
        model: nn.Module,
        base_lr: float = 1e-4,
        decay_factor: float = 0.5
    ) -> list:
        """
        Use different learning rates for different layers.
        
        Args:
            model: PyTorch model
            base_lr: Base learning rate (for last layer)
            decay_factor: Decay factor for earlier layers
            
        Returns:
            Parameter groups for optimizer
        """
        layers = list(model.children())
        param_groups = []
        
        for i, layer in enumerate(reversed(layers)):
            lr = base_lr * (decay_factor ** i)
            param_groups.append({
                'params': layer.parameters(),
                'lr': lr
            })
            
        return param_groups


def main():
    """Example usage."""
    print("="*60)
    print("Subject Adaptation")
    print("="*60)
    
    batch_size = 4
    seq_len = 100
    feature_dim = 512
    num_subjects = 10
    
    # Test Subject Embedding
    print("\n1. Subject Embedding")
    subj_emb = SubjectEmbedding(num_subjects, embedding_dim=64, feature_dim=feature_dim)
    
    features = torch.randn(batch_size, seq_len, feature_dim)
    subject_ids = torch.randint(0, num_subjects, (batch_size,))
    
    adapted_features = subj_emb(features, subject_ids)
    print(f"   Input: {features.shape}")
    print(f"   Subject IDs: {subject_ids}")
    print(f"   Output: {adapted_features.shape}")
    
    # Test Domain-Adversarial Network
    print("\n2. Domain-Adversarial Network")
    dann = DomainAdversarialNetwork(feature_dim, num_subjects)
    
    domain_pred = dann(features)
    print(f"   Features: {features.shape}")
    print(f"   Domain predictions: {domain_pred.shape}")
    
    # Test CORAL
    print("\n3. CORAL Domain Adaptation")
    coral = CORAL()
    
    source_features = torch.randn(batch_size, feature_dim)
    target_features = torch.randn(batch_size, feature_dim)
    
    coral_loss = coral(source_features, target_features)
    print(f"   Source features: {source_features.shape}")
    print(f"   Target features: {target_features.shape}")
    print(f"   CORAL loss: {coral_loss.item():.4f}")
    
    # Test Subject-Adaptive BatchNorm
    print("\n4. Subject-Adaptive Batch Normalization")
    sabn = SubjectAdaptiveBatchNorm(num_subjects, feature_dim)
    
    normalized = sabn(features, subject_ids)
    print(f"   Input: {features.shape}")
    print(f"   Output: {normalized.shape}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
