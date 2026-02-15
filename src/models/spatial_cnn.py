"""
CNN-based Spatial Feature Extractor for EEG Signals

This module implements convolutional neural networks to extract spatial features
from multi-channel EEG data. The CNN processes the electrode dimension to learn
spatial patterns across brain regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialCNN(nn.Module):
    """
    CNN-based spatial feature extractor for EEG signals.
    
    Processes multi-channel EEG data to extract spatial features across electrodes.
    Inspired by EEGNet and DeepConvNet architectures.
    """
    
    def __init__(
        self,
        n_channels: int,
        n_temporal_filters: int = 8,
        temporal_kernel_size: int = 64,
        n_spatial_filters: int = 16,
        dropout: float = 0.5,
        pool_size: int = 4
    ):
        """
        Initialize spatial CNN.
        
        Args:
            n_channels: Number of EEG channels
            n_temporal_filters: Number of temporal convolution filters
            temporal_kernel_size: Kernel size for temporal convolution
            n_spatial_filters: Number of spatial convolution filters
            dropout: Dropout rate
            pool_size: Pooling size
        """
        super(SpatialCNN, self).__init__()
        
        self.n_channels = n_channels
        self.n_temporal_filters = n_temporal_filters
        self.n_spatial_filters = n_spatial_filters
        
        # Temporal convolution (across time)
        # Input: (batch, 1, channels, time)
        # Output: (batch, n_temporal_filters, channels, time)
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_temporal_filters,
            kernel_size=(1, temporal_kernel_size),
            padding=(0, temporal_kernel_size // 2),
            bias=False
        )
        
        self.temporal_bn = nn.BatchNorm2d(n_temporal_filters)
        
        # Spatial (depthwise) convolution (across channels)
        # Input: (batch, n_temporal_filters, channels, time)
        # Output: (batch, n_temporal_filters, 1, time)
        self.spatial_conv = nn.Conv2d(
            in_channels=n_temporal_filters,
            out_channels=n_temporal_filters,
            kernel_size=(n_channels, 1),
            groups=n_temporal_filters,  # Depthwise convolution
            bias=False
        )
        
        self.spatial_bn = nn.BatchNorm2d(n_temporal_filters)
        
        # Separable convolution
        # Pointwise
        self.pointwise_conv = nn.Conv2d(
            in_channels=n_temporal_filters,
            out_channels=n_spatial_filters,
            kernel_size=(1, 1),
            bias=False
        )
        
        # Depthwise
        self.depthwise_conv = nn.Conv2d(
            in_channels=n_spatial_filters,
            out_channels=n_spatial_filters,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=n_spatial_filters,
            bias=False
        )
        
        self.separable_bn = nn.BatchNorm2d(n_spatial_filters)
        
        # Pooling and dropout
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        
        logger.info(
            f"Initialized SpatialCNN: {n_channels} channels, "
            f"{n_temporal_filters} temporal filters, {n_spatial_filters} spatial filters"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, time) or (batch, 1, channels, time)
            
        Returns:
            Spatial features (batch, n_spatial_filters, time_reduced)
        """
        # Ensure 4D input: (batch, 1, channels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
            
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Separable convolution
        x = self.pointwise_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_bn(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Remove height dimension (should be 1 after spatial conv)
        x = x.squeeze(2)  # (batch, n_spatial_filters, time)
        
        return x


class EEGNet(nn.Module):
    """
    EEGNet architecture for EEG classification/feature extraction.
    
    Reference: Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural 
    Network for EEG-based Brain-Computer Interfaces"
    """
    
    def __init__(
        self,
        n_channels: int,
        sampling_rate: int = 500,
        n_temporal_filters: int = 8,
        depth_multiplier: int = 2,
        n_pointwise_filters: int = 16,
        dropout: float = 0.5
    ):
        """
        Initialize EEGNet.
        
        Args:
            n_channels: Number of EEG channels
            sampling_rate: Sampling rate in Hz
            n_temporal_filters: Number of temporal filters (F1)
            depth_multiplier: Depth multiplier for spatial filters (D)
            n_pointwise_filters: Number of pointwise filters (F2)
            dropout: Dropout rate
        """
        super(EEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        
        # Calculate kernel sizes based on sampling rate
        temporal_kernel = sampling_rate // 2  # 250ms at 500Hz
        
        # Block 1: Temporal convolution
        self.block1_conv = nn.Conv2d(
            1, n_temporal_filters,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False
        )
        self.block1_bn = nn.BatchNorm2d(n_temporal_filters)
        
        # Block 1: Depthwise convolution (spatial)
        n_spatial_filters = n_temporal_filters * depth_multiplier
        self.block1_depthwise = nn.Conv2d(
            n_temporal_filters, n_spatial_filters,
            kernel_size=(n_channels, 1),
            groups=n_temporal_filters,
            bias=False
        )
        self.block1_depthwise_bn = nn.BatchNorm2d(n_spatial_filters)
        self.block1_pool = nn.AvgPool2d((1, 4))
        self.block1_dropout = nn.Dropout(dropout)
        
        # Block 2: Separable convolution
        self.block2_separable1 = nn.Conv2d(
            n_spatial_filters, n_spatial_filters,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=n_spatial_filters,
            bias=False
        )
        self.block2_separable2 = nn.Conv2d(
            n_spatial_filters, n_pointwise_filters,
            kernel_size=(1, 1),
            bias=False
        )
        self.block2_bn = nn.BatchNorm2d(n_pointwise_filters)
        self.block2_pool = nn.AvgPool2d((1, 8))
        self.block2_dropout = nn.Dropout(dropout)
        
        logger.info(
            f"Initialized EEGNet: {n_channels} channels, "
            f"F1={n_temporal_filters}, D={depth_multiplier}, F2={n_pointwise_filters}"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, time) or (batch, 1, channels, time)
            
        Returns:
            Features (batch, n_pointwise_filters, time_reduced)
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Block 1
        x = self.block1_conv(x)
        x = self.block1_bn(x)
        
        x = self.block1_depthwise(x)
        x = self.block1_depthwise_bn(x)
        x = F.elu(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)
        
        # Block 2
        x = self.block2_separable1(x)
        x = self.block2_separable2(x)
        x = self.block2_bn(x)
        x = F.elu(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)
        
        # Remove height dimension
        x = x.squeeze(2)
        
        return x


class DeepConvNet(nn.Module):
    """
    Deep Convolutional Network for EEG.
    
    Reference: Schirrmeister et al. (2017) "Deep learning with convolutional 
    neural networks for EEG decoding and visualization"
    """
    
    def __init__(
        self,
        n_channels: int,
        n_filters: int = 25,
        dropout: float = 0.5
    ):
        """
        Initialize DeepConvNet.
        
        Args:
            n_channels: Number of EEG channels
            n_filters: Number of filters in first layer
            dropout: Dropout rate
        """
        super(DeepConvNet, self).__init__()
        
        # Spatial convolution
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=(n_channels, 1))
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=(1, 10))
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.pool1 = nn.MaxPool2d((1, 3))
        self.dropout1 = nn.Dropout(dropout)
        
        # Conv block 2
        self.conv3 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=(1, 10))
        self.bn2 = nn.BatchNorm2d(n_filters * 2)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.dropout2 = nn.Dropout(dropout)
        
        # Conv block 3
        self.conv4 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=(1, 10))
        self.bn3 = nn.BatchNorm2d(n_filters * 4)
        self.pool3 = nn.MaxPool2d((1, 3))
        self.dropout3 = nn.Dropout(dropout)
        
        logger.info(f"Initialized DeepConvNet: {n_channels} channels, {n_filters} filters")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = x.squeeze(2)
        
        return x


def main():
    """Test spatial CNN models."""
    # Create sample EEG data
    batch_size = 8
    n_channels = 64
    n_timepoints = 500
    
    x = torch.randn(batch_size, n_channels, n_timepoints)
    
    print("="*60)
    print("Testing Spatial CNN Models")
    print("="*60)
    
    # Test SpatialCNN
    print("\n1. Testing SpatialCNN...")
    model = SpatialCNN(n_channels=n_channels)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test EEGNet
    print("\n2. Testing EEGNet...")
    model = EEGNet(n_channels=n_channels)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test DeepConvNet
    print("\n3. Testing DeepConvNet...")
    model = DeepConvNet(n_channels=n_channels)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n" + "="*60)
    print("All spatial CNN models tested successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
