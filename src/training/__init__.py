"""
Training utilities for NEST models.
"""

from .trainer import (
    Trainer,
    CTCTrainer,
    EarlyStopping,
    get_optimizer,
    get_scheduler,
    count_parameters
)

from .metrics import (
    word_error_rate,
    character_error_rate,
    compute_bleu,
    perplexity,
    accuracy,
    MetricsTracker
)

from .checkpoint import (
    CheckpointManager,
    save_config,
    load_config
)

from .robustness import (
    AdversarialTrainer,
    NoiseInjection,
    DenoisingAutoencoder,
    RobustLoss,
    GradientNoise
)

__all__ = [
    # Trainers
    'Trainer',
    'CTCTrainer',
    'EarlyStopping',
    'get_optimizer',
    'get_scheduler',
    'count_parameters',
    # Metrics
    'word_error_rate',
    'character_error_rate',
    'compute_bleu',
    'perplexity',
    'accuracy',
    'MetricsTracker',
    # Checkpoint
    'CheckpointManager',
    'save_config',
    'load_config',
    # Robustness
    'AdversarialTrainer',
    'NoiseInjection',
    'DenoisingAutoencoder',
    'RobustLoss',
    'GradientNoise'
]
