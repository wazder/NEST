"""
NEST: Neural EEG Sequence Transducer

Complete model architectures combining spatial CNN, temporal encoders,
and sequence transduction mechanisms for EEG-to-text decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging

from .spatial_cnn import SpatialCNN, EEGNet
from .temporal_encoder import LSTMEncoder, GRUEncoder, TransformerEncoder
from .attention import CrossAttention
from .decoder import CTCDecoder, AttentionDecoder, TransducerDecoder, JointNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NEST_RNN_T(nn.Module):
    """
    NEST with RNN-Transducer architecture.
    """
    
    def __init__(
        self,
        n_channels: int,
        vocab_size: int,
        spatial_cnn_type: str = 'EEGNet',
        temporal_encoder_type: str = 'LSTM',
        encoder_hidden_dim: int = 512,
        decoder_hidden_dim: int = 512,
        joint_dim: int = 512,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dropout: float = 0.3,
        blank_id: int = 0,
        spatial_cnn: Optional[nn.Module] = None,
        temporal_encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        joint: Optional[nn.Module] = None
    ):
        super(NEST_RNN_T, self).__init__()
        
        self.n_channels = n_channels
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        
        # Spatial CNN
        if spatial_cnn is not None:
             self.spatial_cnn = spatial_cnn
             # Infer output dim if possible, or assume standard from factory
             cnn_output_dim = 64 # Default for our factory setup
             if hasattr(spatial_cnn, 'output_dim'):
                 cnn_output_dim = spatial_cnn.output_dim
        elif spatial_cnn_type == 'EEGNet':
            self.spatial_cnn = EEGNet(
                n_channels=n_channels,
                n_pointwise_filters=64,
                dropout=dropout
            )
            cnn_output_dim = 64
        else:
            self.spatial_cnn = SpatialCNN(
                n_channels=n_channels,
                n_spatial_filters=64,
                dropout=dropout
            )
            cnn_output_dim = 64
            
        # Temporal Encoder
        if temporal_encoder is not None:
            self.temporal_encoder = temporal_encoder
            encoder_output_dim = encoder_hidden_dim * 2 # Assumption/Default
            if hasattr(temporal_encoder, 'output_dim'):
                encoder_output_dim = temporal_encoder.output_dim
        elif temporal_encoder_type == 'LSTM':
            self.temporal_encoder = LSTMEncoder(
                input_dim=cnn_output_dim,
                hidden_dim=encoder_hidden_dim,
                num_layers=num_encoder_layers,
                dropout=dropout,
                bidirectional=True
            )
            encoder_output_dim = encoder_hidden_dim * 2
        elif temporal_encoder_type == 'GRU':
            self.temporal_encoder = GRUEncoder(
                input_dim=cnn_output_dim,
                hidden_dim=encoder_hidden_dim,
                num_layers=num_encoder_layers,
                dropout=dropout,
                bidirectional=True
            )
            encoder_output_dim = encoder_hidden_dim * 2
        else:
            raise ValueError(f"Unknown encoder type: {temporal_encoder_type}")
            
        # Prediction Network (Decoder)
        if decoder is not None:
            self.prediction_network = decoder
        else:
            self.prediction_network = TransducerDecoder(
                vocab_size=vocab_size,
                embedding_dim=256,
                hidden_dim=decoder_hidden_dim,
                num_layers=num_decoder_layers,
                dropout=dropout
            )
        
        # Joint Network
        if joint is not None:
            self.joint_network = joint
        else:
            self.joint_network = JointNetwork(
                encoder_dim=encoder_output_dim,
                decoder_dim=decoder_hidden_dim,
                joint_dim=joint_dim,
                vocab_size=vocab_size
            )
        
        logger.info(f"Initialized NEST_RNN_T")
        
    def forward(
        self,
        eeg_data: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Spatial feature extraction
        spatial_features = self.spatial_cnn(eeg_data)  # (batch, features, time)
        
        # Transpose for temporal encoder: (batch, time, features)
        spatial_features = spatial_features.transpose(1, 2)
        
        # Temporal encoding
        if isinstance(self.temporal_encoder, TransformerEncoder):
             encoder_output = self.temporal_encoder(spatial_features)
        else:
             encoder_output, _ = self.temporal_encoder(spatial_features)
        
        # Prediction network
        if target_labels is None:
            # Inference: start with blank token
            batch_size = eeg_data.size(0)
            target_labels = torch.full(
                (batch_size, 1),
                self.blank_id,
                dtype=torch.long,
                device=eeg_data.device
            )
            
        decoder_output, _ = self.prediction_network(target_labels)
        
        # Joint network
        joint_output = self.joint_network(encoder_output, decoder_output)
        
        return joint_output


class NEST_Transformer_T(nn.Module):
    """
    NEST with Transformer-Transducer architecture.
    """
    
    def __init__(
        self,
        n_channels: int,
        vocab_size: int,
        spatial_cnn_type: str = 'EEGNet',
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        blank_id: int = 0,
        spatial_cnn: Optional[nn.Module] = None,
        temporal_encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        joint: Optional[nn.Module] = None
    ):
        super(NEST_Transformer_T, self).__init__()
        
        self.n_channels = n_channels
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        
        # Spatial CNN
        if spatial_cnn is not None:
            self.spatial_cnn = spatial_cnn
            cnn_output_dim = 64
            if hasattr(spatial_cnn, 'output_dim'):
                 cnn_output_dim = spatial_cnn.output_dim
        elif spatial_cnn_type == 'EEGNet':
            self.spatial_cnn = EEGNet(
                n_channels=n_channels,
                n_pointwise_filters=64,
                dropout=dropout
            )
            cnn_output_dim = 64
        else:
            self.spatial_cnn = SpatialCNN(
                n_channels=n_channels,
                n_spatial_filters=64,
                dropout=dropout
            )
            cnn_output_dim = 64
            
        # Transformer Encoder
        if temporal_encoder is not None:
            self.temporal_encoder = temporal_encoder
        else:
            self.temporal_encoder = TransformerEncoder(
                input_dim=cnn_output_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        
        # Prediction Network
        if decoder is not None:
             self.prediction_network = decoder
        else:
            self.prediction_network = TransducerDecoder(
                vocab_size=vocab_size,
                embedding_dim=256,
                hidden_dim=d_model,
                num_layers=num_decoder_layers,
                dropout=dropout
            )
        
        # Joint Network
        if joint is not None:
             self.joint_network = joint
        else:
            self.joint_network = JointNetwork(
                encoder_dim=d_model,
                decoder_dim=d_model,
                joint_dim=d_model,
                vocab_size=vocab_size
            )
        
        logger.info(f"Initialized NEST_Transformer_T")
        
    def forward(
        self,
        eeg_data: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        # Spatial feature extraction
        spatial_features = self.spatial_cnn(eeg_data)
        spatial_features = spatial_features.transpose(1, 2)
        
        # Transformer encoding
        encoder_output = self.temporal_encoder(
            spatial_features,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Prediction network
        if target_labels is None:
            batch_size = eeg_data.size(0)
            target_labels = torch.full(
                (batch_size, 1),
                self.blank_id,
                dtype=torch.long,
                device=eeg_data.device
            )
            
        decoder_output, _ = self.prediction_network(target_labels)
        
        # Joint network
        joint_output = self.joint_network(encoder_output, decoder_output)
        
        return joint_output


class NEST_Attention(nn.Module):
    """
    NEST with attention-based encoder-decoder architecture.
    """
    
    def __init__(
        self,
        n_channels: int,
        vocab_size: int,
        spatial_cnn_type: str = 'EEGNet',
        encoder_type: str = 'LSTM',
        encoder_hidden_dim: int = 512,
        decoder_hidden_dim: int = 512,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dropout: float = 0.3,
        spatial_cnn: Optional[nn.Module] = None,
        temporal_encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None
    ):
        super(NEST_Attention, self).__init__()
        
        # Spatial CNN
        if spatial_cnn is not None:
            self.spatial_cnn = spatial_cnn
            cnn_output_dim = 16
            if hasattr(spatial_cnn, 'output_dim'):
                 cnn_output_dim = spatial_cnn.output_dim
        elif spatial_cnn_type == 'EEGNet':
            self.spatial_cnn = EEGNet(n_channels=n_channels, dropout=dropout)
            cnn_output_dim = 16  # EEGNet default
        else:
            self.spatial_cnn = SpatialCNN(n_channels=n_channels, dropout=dropout)
            cnn_output_dim = 16
            
        # Temporal Encoder
        if temporal_encoder is not None:
             self.temporal_encoder = temporal_encoder
             encoder_output_dim = encoder_hidden_dim * 2
             if hasattr(temporal_encoder, 'output_dim'):
                 encoder_output_dim = temporal_encoder.output_dim
        elif encoder_type == 'LSTM':
            self.temporal_encoder = LSTMEncoder(
                input_dim=cnn_output_dim,
                hidden_dim=encoder_hidden_dim,
                num_layers=num_encoder_layers,
                dropout=dropout,
                bidirectional=True
            )
            encoder_output_dim = encoder_hidden_dim * 2
        elif encoder_type == 'Transformer':
            self.temporal_encoder = TransformerEncoder(
                input_dim=cnn_output_dim,
                d_model=encoder_hidden_dim,
                num_layers=num_encoder_layers,
                dropout=dropout
            )
            encoder_output_dim = encoder_hidden_dim
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        # Attention Decoder
        if decoder is not None:
             self.decoder = decoder
        else:
            self.decoder = AttentionDecoder(
                vocab_size=vocab_size,
                embedding_dim=256,
                encoder_dim=encoder_output_dim,
                hidden_dim=decoder_hidden_dim,
                num_layers=num_decoder_layers,
                dropout=dropout
            )
        
        logger.info(f"Initialized NEST_Attention")
        
    def forward(
        self,
        eeg_data: torch.Tensor,
        target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        # Spatial features
        spatial_features = self.spatial_cnn(eeg_data)
        spatial_features = spatial_features.transpose(1, 2)
        
        # Temporal encoding
        if isinstance(self.temporal_encoder, TransformerEncoder):
            encoder_output = self.temporal_encoder(spatial_features)
        else:
            encoder_output, _ = self.temporal_encoder(spatial_features)
            
        # Decode with attention
        output, _ = self.decoder(encoder_output, target_tokens)
        
        return output


class NEST_CTC(nn.Module):
    """
    NEST with CTC (Connectionist Temporal Classification).
    """
    
    def __init__(
        self,
        n_channels: int = 105, # Default to 105 for ZuCo
        vocab_size: int = 30,
        spatial_cnn_type: str = 'EEGNet',
        encoder_type: str = 'LSTM',
        encoder_hidden_dim: int = 512,
        num_encoder_layers: int = 3,
        dropout: float = 0.3,
        blank_id: int = 0,
        spatial_cnn: Optional[nn.Module] = None,
        temporal_encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None
    ):
        super(NEST_CTC, self).__init__()
        
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        
        # Spatial CNN
        if spatial_cnn is not None:
            self.spatial_cnn = spatial_cnn
            cnn_output_dim = 16
            if hasattr(spatial_cnn, 'output_dim'):
                 cnn_output_dim = spatial_cnn.output_dim
        elif spatial_cnn_type == 'EEGNet':
            self.spatial_cnn = EEGNet(n_channels=n_channels, dropout=dropout)
            cnn_output_dim = 16
        else:
            self.spatial_cnn = SpatialCNN(n_channels=n_channels, dropout=dropout)
            cnn_output_dim = 16
            
        # Temporal Encoder
        if temporal_encoder is not None:
             self.temporal_encoder = temporal_encoder
             encoder_output_dim = encoder_hidden_dim * 2
             if hasattr(temporal_encoder, 'output_dim'):
                 encoder_output_dim = temporal_encoder.output_dim
        elif encoder_type == 'LSTM':
            self.temporal_encoder = LSTMEncoder(
                input_dim=cnn_output_dim,
                hidden_dim=encoder_hidden_dim,
                num_layers=num_encoder_layers,
                dropout=dropout,
                bidirectional=True
            )
            encoder_output_dim = encoder_hidden_dim * 2
        else:
            self.temporal_encoder = TransformerEncoder(
                input_dim=cnn_output_dim,
                d_model=encoder_hidden_dim,
                num_layers=num_encoder_layers,
                dropout=dropout
            )
            encoder_output_dim = encoder_hidden_dim
            
        # CTC Decoder
        # Note: factory passes 'decoder' as CTCDecoder instance
        if decoder is not None:
             self.ctc_decoder = decoder
        else:
            self.ctc_decoder = CTCDecoder(
                input_dim=encoder_output_dim,
                vocab_size=vocab_size,
                blank_id=blank_id
            )
        
        logger.info(f"Initialized NEST_CTC")
        
    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Spatial features
        spatial_features = self.spatial_cnn(eeg_data)
        spatial_features = spatial_features.transpose(1, 2)
        
        # Temporal encoding
        if isinstance(self.temporal_encoder, TransformerEncoder):
            encoder_output = self.temporal_encoder(spatial_features)
        else:
            encoder_output, _ = self.temporal_encoder(spatial_features)
            
        # CTC decoding
        log_probs = self.ctc_decoder(encoder_output)
        
        return log_probs


def main():
    """Test NEST models."""
    batch_size = 2
    n_channels = 64
    n_timepoints = 500
    vocab_size = 50
    label_len = 20
    
    eeg_data = torch.randn(batch_size, n_channels, n_timepoints)
    target_labels = torch.randint(0, vocab_size, (batch_size, label_len))
    
    print("="*60)
    print("Testing NEST Model Architectures")
    print("="*60)
    
    # Test NEST_RNN_T
    print("\n1. Testing NEST_RNN_T...")
    model = NEST_RNN_T(n_channels=n_channels, vocab_size=vocab_size)
    output = model(eeg_data, target_labels)
    print(f"   Input shape: {eeg_data.shape}")
    print(f"   Target shape: {target_labels.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test NEST_Transformer_T
    print("\n2. Testing NEST_Transformer_T...")
    model = NEST_Transformer_T(n_channels=n_channels, vocab_size=vocab_size)
    output = model(eeg_data, target_labels)
    print(f"   Output shape: {output.shape}")
    
    # Test NEST_Attention
    print("\n3. Testing NEST_Attention...")
    model = NEST_Attention(n_channels=n_channels, vocab_size=vocab_size)
    output = model(eeg_data, target_labels)
    print(f"   Output shape: {output.shape}")
    
    # Test NEST_CTC
    print("\n4. Testing NEST_CTC...")
    model = NEST_CTC(n_channels=n_channels, vocab_size=vocab_size)
    output = model(eeg_data)
    print(f"   Output shape: {output.shape}")
    
    print("\n" + "="*60)
    print("All NEST models tested successfully!")
    print("="*60)
    print("\nAvailable architectures:")
    print("  - NEST_RNN_T: RNN-Transducer")
    print("  - NEST_Transformer_T: Transformer-Transducer")
    print("  - NEST_Attention: Seq2seq with attention")
    print("  - NEST_CTC: CTC-based model")


if __name__ == "__main__":
    main()
