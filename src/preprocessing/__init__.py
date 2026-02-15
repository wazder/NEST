"""Preprocessing modules for EEG signal processing."""

from .filtering import EEGFilter, AdaptiveFilter
from .artifact_removal import ICArtifactRemoval, AutomaticArtifactRejection
from .electrode_selection import ElectrodeSelector, ChannelOptimizer
from .augmentation import EEGAugmentation, AdvancedAugmentation
from .data_split import EEGDataSplitter, DatasetOrganizer
from .pipeline import PreprocessingPipeline

__all__ = [
    'EEGFilter',
    'AdaptiveFilter',
    'ICArtifactRemoval',
    'AutomaticArtifactRejection',
    'ElectrodeSelector',
    'ChannelOptimizer',
    'EEGAugmentation',
    'AdvancedAugmentation',
    'EEGDataSplitter',
    'DatasetOrganizer',
    'PreprocessingPipeline',
]
