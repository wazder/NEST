"""
Model Factory for NEST Architectures

This module provides utilities to build models from configuration files.
"""

import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union

from .spatial_cnn import SpatialCNN, EEGNet, DeepConvNet
from .temporal_encoder import LSTMEncoder, GRUEncoder, TransformerEncoder, ConformerEncoder
from .attention import CrossAttention, AdditiveAttention, LocationAwareAttention
from .decoder import CTCDecoder, AttentionDecoder, TransducerDecoder, JointNetwork
from .nest import NEST_RNN_T, NEST_Transformer_T, NEST_Attention, NEST_CTC


class ModelFactory:
    """Factory for creating NEST models from configuration."""
    
    # Model registry
    MODELS = {
        'NEST_RNN_T': NEST_RNN_T,
        'NEST_Transformer_T': NEST_Transformer_T,
        'NEST_Attention': NEST_Attention,
        'NEST_CTC': NEST_CTC,
    }
    
    # Spatial CNN registry
    SPATIAL_CNN = {
        'SpatialCNN': SpatialCNN,
        'EEGNet': EEGNet,
        'DeepConvNet': DeepConvNet,
    }
    
    # Temporal encoder registry
    TEMPORAL_ENCODER = {
        'LSTM': LSTMEncoder,
        'GRU': GRUEncoder,
        'Transformer': TransformerEncoder,
        'Conformer': ConformerEncoder,
    }
    
    # Attention registry
    ATTENTION = {
        'CrossAttention': CrossAttention,
        'AdditiveAttention': AdditiveAttention,
        'LocationAwareAttention': LocationAwareAttention,
    }
    
    @classmethod
    def create_spatial_cnn(cls, config: Dict) -> nn.Module:
        """
        Create spatial CNN from configuration.
        
        Args:
            config: Spatial CNN configuration
            
        Returns:
            Spatial CNN module
        """
        cnn_type = config.pop('type')
        
        if cnn_type not in cls.SPATIAL_CNN:
            raise ValueError(f"Unknown spatial CNN type: {cnn_type}")
            
        return cls.SPATIAL_CNN[cnn_type](**config)
        
    @classmethod
    def create_temporal_encoder(cls, config: Dict) -> nn.Module:
        """
        Create temporal encoder from configuration.
        
        Args:
            config: Temporal encoder configuration
            
        Returns:
            Temporal encoder module
        """
        encoder_type = config.pop('type')
        
        if encoder_type not in cls.TEMPORAL_ENCODER:
            raise ValueError(f"Unknown temporal encoder type: {encoder_type}")
            
        return cls.TEMPORAL_ENCODER[encoder_type](**config)
        
    @classmethod
    def create_attention(cls, config: Dict) -> nn.Module:
        """
        Create attention mechanism from configuration.
        
        Args:
            config: Attention configuration
            
        Returns:
            Attention module
        """
        attention_type = config.pop('type')
        
        if attention_type not in cls.ATTENTION:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
        return cls.ATTENTION[attention_type](**config)
        
    @classmethod
    def create_model(cls, config: Dict) -> nn.Module:
        """
        Create complete model from configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            NEST model
        """
        model_name = config['model_name']
        
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Extract component configs
        spatial_config = config['spatial_cnn'].copy()
        temporal_config = config['temporal_encoder'].copy()
        
        # Create components
        spatial_cnn = cls.create_spatial_cnn(spatial_config)
        temporal_encoder = cls.create_temporal_encoder(temporal_config)
        
        # Create model based on type
        if model_name == 'NEST_RNN_T':
            decoder_config = config['decoder']
            joint_config = config['joint']
            
            decoder = TransducerDecoder(**decoder_config)
            joint = JointNetwork(**joint_config)
            
            model = NEST_RNN_T(
                spatial_cnn=spatial_cnn,
                temporal_encoder=temporal_encoder,
                decoder=decoder,
                joint=joint
            )
            
        elif model_name == 'NEST_Transformer_T':
            decoder_config = config['decoder']
            joint_config = config['joint']
            
            decoder = TransducerDecoder(**decoder_config)
            joint = JointNetwork(**joint_config)
            
            model = NEST_Transformer_T(
                spatial_cnn=spatial_cnn,
                temporal_encoder=temporal_encoder,
                decoder=decoder,
                joint=joint
            )
            
        elif model_name == 'NEST_Attention':
            attention_config = config['attention'].copy()
            decoder_config = config['decoder']
            
            attention = cls.create_attention(attention_config)
            decoder = AttentionDecoder(**decoder_config)
            
            model = NEST_Attention(
                spatial_cnn=spatial_cnn,
                temporal_encoder=temporal_encoder,
                attention=attention,
                decoder=decoder
            )
            
        elif model_name == 'NEST_CTC':
            decoder_config = config['decoder']
            decoder = CTCDecoder(**decoder_config)
            
            model = NEST_CTC(
                spatial_cnn=spatial_cnn,
                temporal_encoder=temporal_encoder,
                decoder=decoder
            )
            
        else:
            raise ValueError(f"Model {model_name} not implemented in factory")
            
        return model
        
    @classmethod
    def from_config_file(
        cls,
        config_path: str,
        model_key: str,
        vocab_size: Optional[int] = None
    ) -> nn.Module:
        """
        Create model from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            model_key: Key of model configuration (e.g., 'nest_rnn_t')
            vocab_size: Vocabulary size (overrides config if provided)
            
        Returns:
            NEST model
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load configuration
        with open(config_path, 'r') as f:
            all_configs = yaml.safe_load(f)
            
        if model_key not in all_configs:
            raise ValueError(f"Model key '{model_key}' not found in config")
            
        config = all_configs[model_key]
        
        # Override vocab size if provided
        if vocab_size is not None:
            if 'decoder' in config:
                config['decoder']['vocab_size'] = vocab_size
            if 'joint' in config:
                config['joint']['vocab_size'] = vocab_size
                
        # Create model
        model = cls.create_model(config)
        
        return model


def load_pretrained(
    checkpoint_path: str,
    model_config: Optional[Dict] = None,
    map_location: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Load pretrained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model_config: Model configuration (optional, will use checkpoint config)
        map_location: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Get model config from checkpoint or use provided
    if model_config is None:
        if 'model_config' not in checkpoint:
            raise ValueError("No model config found in checkpoint")
        model_config = checkpoint['model_config']
        
    # Create model
    model = ModelFactory.create_model(model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in model (total and trainable).
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def main():
    """Example usage."""
    print("="*60)
    print("Model Factory")
    print("="*60)
    
    # Load model from config file
    config_path = 'configs/model.yaml'
    
    if Path(config_path).exists():
        print(f"\nLoading model from {config_path}")
        
        # Create NEST_RNN_T
        model = ModelFactory.from_config_file(
            config_path,
            model_key='nest_rnn_t',
            vocab_size=5000
        )
        
        print(f"Created model: {model.__class__.__name__}")
        
        # Count parameters
        params = count_parameters(model)
        print(f"\nParameters:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Frozen: {params['frozen']:,}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 100
        n_channels = 104
        
        x = torch.randn(batch_size, n_channels, seq_len)
        targets = torch.randint(0, 5000, (batch_size, 20))
        
        output = model(x, targets)
        print(f"\nForward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
    else:
        print(f"\nConfig file not found: {config_path}")
        print("Please create config file first")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
