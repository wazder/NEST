"""
Noise Robustness and Adversarial Training for NEST

This module implements techniques for handling noisy EEG signals:
- Adversarial training (FGSM, PGD)
- Noise injection during training
- Denoising autoencoder
- Robust loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import numpy as np


class AdversarialTrainer:
    """
    Adversarial training for robust models.
    
    Goodfellow et al. (2015): "Explaining and Harnessing Adversarial Examples"
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        attack_method: str = 'fgsm',
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 10
    ):
        """
        Initialize adversarial trainer.
        
        Args:
            model: Model to train
            criterion: Loss function
            attack_method: 'fgsm' or 'pgd'
            epsilon: Maximum perturbation
            alpha: Step size for iterative attacks
            num_steps: Number of steps for PGD
        """
        self.model = model
        self.criterion = criterion
        self.attack_method = attack_method
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        
    def fgsm_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            x: Input data
            y: True labels
            
        Returns:
            Adversarial examples
        """
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(x_adv)
        loss = self.criterion(output, y)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + self.epsilon * grad_sign
        
        return x_adv.detach()
        
    def pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack.
        
        Args:
            x: Input data
            y: True labels
            
        Returns:
            Adversarial examples
        """
        x_adv = x.clone().detach()
        
        for _ in range(self.num_steps):
            x_adv.requires_grad = True
            
            # Forward pass
            output = self.model(x_adv)
            loss = self.criterion(output, y)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update perturbation
            grad = x_adv.grad
            x_adv = x_adv + self.alpha * grad.sign()
            
            # Project to epsilon ball
            perturbation = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + perturbation
            x_adv = x_adv.detach()
            
        return x_adv
        
    def generate_adversarial_examples(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial examples using specified method.
        
        Args:
            x: Input data
            y: True labels
            
        Returns:
            Adversarial examples
        """
        if self.attack_method == 'fgsm':
            return self.fgsm_attack(x, y)
        elif self.attack_method == 'pgd':
            return self.pgd_attack(x, y)
        else:
            raise ValueError(f"Unknown attack method: {self.attack_method}")


class NoiseInjection(nn.Module):
    """
    Noise injection for training robustness.
    """
    
    def __init__(
        self,
        noise_type: str = 'gaussian',
        noise_level: float = 0.1,
        apply_prob: float = 0.5
    ):
        """
        Initialize noise injection.
        
        Args:
            noise_type: 'gaussian', 'uniform', or 'salt_pepper'
            noise_level: Noise intensity
            apply_prob: Probability of applying noise
        """
        super().__init__()
        
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.apply_prob = apply_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply noise to input.
        
        Args:
            x: Input tensor
            
        Returns:
            Noisy tensor
        """
        if not self.training or torch.rand(1).item() > self.apply_prob:
            return x
            
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
            
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(x) - 0.5) * 2 * self.noise_level
            return x + noise
            
        elif self.noise_type == 'salt_pepper':
            mask = torch.rand_like(x)
            x_noisy = x.clone()
            x_noisy[mask < self.noise_level / 2] = x.min()
            x_noisy[mask > 1 - self.noise_level / 2] = x.max()
            return x_noisy
            
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")


class DenoisingAutoencoder(nn.Module):
    """
    Denoising autoencoder for EEG signal cleaning.
    
    Can be used for pre-training or as a preprocessing step.
    """
    
    def __init__(
        self,
        input_channels: int,
        latent_dim: int = 64,
        noise_level: float = 0.1
    ):
        """
        Initialize denoising autoencoder.
        
        Args:
            input_channels: Number of EEG channels
            latent_dim: Latent dimension
            noise_level: Training noise level
        """
        super().__init__()
        
        self.noise_level = noise_level
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, input_channels, kernel_size=3, padding=1)
        )
        
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input."""
        if self.training:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        return x
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, channels, time)
            
        Returns:
            reconstructed: Reconstructed signal
            latent: Latent representation
        """
        # Add noise during training
        x_noisy = self.add_noise(x)
        
        # Encode
        latent = self.encoder(x_noisy)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
        
    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise signal (evaluation mode).
        
        Args:
            x: Noisy signal
            
        Returns:
            Denoised signal
        """
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
        return reconstructed


class RobustLoss(nn.Module):
    """
    Robust loss functions less sensitive to outliers.
    """
    
    def __init__(self, loss_type: str = 'huber', delta: float = 1.0):
        """
        Initialize robust loss.
        
        Args:
            loss_type: 'huber', 'smooth_l1', or 'tukey'
            delta: Threshold parameter
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.delta = delta
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute robust loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            
        Returns:
            Loss value
        """
        if self.loss_type == 'huber':
            return F.smooth_l1_loss(predictions, targets, beta=self.delta)
            
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(predictions, targets)
            
        elif self.loss_type == 'tukey':
            # Tukey's biweight loss
            diff = predictions - targets
            abs_diff = torch.abs(diff)
            
            # Quadratic for small errors, constant for large
            mask = abs_diff <= self.delta
            loss = torch.where(
                mask,
                (diff ** 2) / 2,
                (self.delta ** 2) / 2
            )
            
            return loss.mean()
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class GradientNoise(nn.Module):
    """
    Add noise to gradients during training (Neelakantan et al., 2015).
    
    "Adding Gradient Noise Improves Learning for Very Deep Networks"
    """
    
    def __init__(
        self,
        eta: float = 0.3,
        gamma: float = 0.55
    ):
        """
        Initialize gradient noise.
        
        Args:
            eta: Noise scale parameter
            gamma: Annealing rate
        """
        super().__init__()
        
        self.eta = eta
        self.gamma = gamma
        self.t = 0  # Training step counter
        
    def add_noise_to_gradients(self, model: nn.Module):
        """
        Add noise to model gradients.
        
        Args:
            model: PyTorch model
        """
        self.t += 1
        
        # Compute noise scale (annealing over time)
        noise_scale = self.eta / ((1 + self.t) ** self.gamma)
        
        # Add noise to gradients
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad = param.grad + noise


def main():
    """Example usage."""
    print("="*60)
    print("Noise Robustness and Adversarial Training")
    print("="*60)
    
    batch_size = 4
    channels = 104
    seq_len = 500
    
    # Test Noise Injection
    print("\n1. Noise Injection")
    noise_injector = NoiseInjection(noise_type='gaussian', noise_level=0.1)
    
    x = torch.randn(batch_size, channels, seq_len)
    x_noisy = noise_injector(x)
    
    noise_added = (x_noisy - x).abs().mean()
    print(f"   Input shape: {x.shape}")
    print(f"   Average noise added: {noise_added.item():.4f}")
    
    # Test Denoising Autoencoder
    print("\n2. Denoising Autoencoder")
    dae = DenoisingAutoencoder(input_channels=channels, latent_dim=64)
    
    reconstructed, latent = dae(x)
    reconstruction_error = F.mse_loss(reconstructed, x)
    
    print(f"   Input: {x.shape}")
    print(f"   Latent: {latent.shape}")
    print(f"   Reconstructed: {reconstructed.shape}")
    print(f"   Reconstruction error: {reconstruction_error.item():.4f}")
    
    # Test Robust Loss
    print("\n3. Robust Loss Functions")
    
    predictions = torch.randn(batch_size, 10)
    targets = torch.randn(batch_size, 10)
    
    # Add outlier
    predictions[0, 0] = 100.0
    
    mse_loss = F.mse_loss(predictions, targets)
    huber_loss = RobustLoss(loss_type='huber', delta=1.0)(predictions, targets)
    
    print(f"   MSE Loss: {mse_loss.item():.4f} (sensitive to outliers)")
    print(f"   Huber Loss: {huber_loss.item():.4f} (robust to outliers)")
    
    # Test Adversarial Training
    print("\n4. Adversarial Training")
    
    dummy_model = nn.Linear(channels * seq_len, 10)
    criterion = nn.CrossEntropyLoss()
    
    adv_trainer = AdversarialTrainer(
        dummy_model,
        criterion,
        attack_method='fgsm',
        epsilon=0.1
    )
    
    x_flat = x.view(batch_size, -1)
    y = torch.randint(0, 10, (batch_size,))
    
    x_adv = adv_trainer.generate_adversarial_examples(x_flat, y)
    perturbation = (x_adv - x_flat).abs().mean()
    
    print(f"   Original: {x_flat.shape}")
    print(f"   Adversarial: {x_adv.shape}")
    print(f"   Average perturbation: {perturbation.item():.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
