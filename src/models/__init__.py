"""Model architectures for NEST."""

from .spatial_cnn import SpatialCNN, EEGNet, DeepConvNet
from .temporal_encoder import LSTMEncoder, GRUEncoder, TransformerEncoder, ConformerEncoder
from .attention import CrossAttention, AdditiveAttention, LocationAwareAttention, MultiHeadAttention
from .advanced_attention import RelativePositionAttention, LocalAttention, LinearAttention
from .decoder import CTCDecoder, AttentionDecoder, TransducerDecoder, JointNetwork
from .nest import NEST_RNN_T, NEST_Transformer_T, NEST_Attention, NEST_CTC
from .factory import ModelFactory, load_pretrained, count_parameters
from .adaptation import SubjectEmbedding, DomainAdversarialNetwork, CORAL, SubjectAdaptiveBatchNorm, FineTuningStrategy
from .language_model import ShallowFusion, DeepFusion, LanguageModelRescorer, SimpleLSTMLM

__all__ = [
    # Spatial CNNs
    'SpatialCNN',
    'EEGNet',
    'DeepConvNet',
    # Temporal Encoders
    'LSTMEncoder',
    'GRUEncoder',
    'TransformerEncoder',
    'ConformerEncoder',
    # Attention Mechanisms
    'CrossAttention',
    'AdditiveAttention',
    'LocationAwareAttention',
    'MultiHeadAttention',
    # Advanced Attention
    'RelativePositionAttention',
    'LocalAttention',
    'LinearAttention',
    # Decoders
    'CTCDecoder',
    'AttentionDecoder',
    'TransducerDecoder',
    'JointNetwork',
    # Complete NEST Models
    'NEST_RNN_T',
    'NEST_Transformer_T',
    'NEST_Attention',
    'NEST_CTC',
    # Factory
    'ModelFactory',
    'load_pretrained',
    'count_parameters',
    # Subject Adaptation
    'SubjectEmbedding',
    'DomainAdversarialNetwork',
    'CORAL',
    'SubjectAdaptiveBatchNorm',
    'FineTuningStrategy',
    # Language Model Integration
    'ShallowFusion',
    'DeepFusion',
    'LanguageModelRescorer',
    'SimpleLSTMLM',
]
