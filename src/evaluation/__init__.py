"""
Evaluation and Optimization Module

This module provides comprehensive evaluation, optimization, and deployment tools:
- Benchmarking and metrics
- Beam search decoding
- Inference optimization (ONNX, TorchScript, FP16)
- Model pruning (magnitude, structured, iterative, sensitivity)
- Model quantization (PTQ, QAT, dynamic, mixed-precision)
- Real-time inference pipeline
- Profiling and benchmarking
- Deployment utilities
"""

from .benchmark import ModelEvaluator, benchmark_suite, load_model_and_tokenizer
from .beam_search import BeamSearchDecoder, GreedyDecoder, BatchBeamSearch, Hypothesis
from .inference_optimizer import InferenceOptimizer, BatchProcessor, ModelCache
from .pruning import (
    MagnitudePruner,
    IterativePruner,
    SensitivityPruner,
    StructuredPruner,
    LotteryTicketPruner
)
from .quantization import (
    PostTrainingQuantizer,
    DynamicQuantizer,
    QATTrainer,
    MixedPrecisionQuantizer,
    QuantizationBenchmark
)
from .realtime_inference import (
    StreamConfig,
    CircularBuffer,
    OnlinePreprocessor,
    StreamingInference,
    SlidingWindowProcessor,
    LatencyMonitor
)
from .profiling import (
    ModelProfiler,
    LayerTimingProfiler,
    MemoryProfiler,
    ThroughputBenchmark,
    ComparisonBenchmark
)
from .deployment import (
    ModelExporter,
    ModelPackager,
    DeploymentConfig
)


__all__ = [
    # Benchmarking
    'ModelEvaluator',
    'benchmark_suite',
    'load_model_and_tokenizer',
    
    # Beam search
    'BeamSearchDecoder',
    'GreedyDecoder',
    'BatchBeamSearch',
    'Hypothesis',
    
    # Inference optimization
    'InferenceOptimizer',
    'BatchProcessor',
    'ModelCache',
    
    # Pruning
    'MagnitudePruner',
    'IterativePruner',
    'SensitivityPruner',
    'StructuredPruner',
    'LotteryTicketPruner',
    
    # Quantization
    'PostTrainingQuantizer',
    'DynamicQuantizer',
    'QATTrainer',
    'MixedPrecisionQuantizer',
    'QuantizationBenchmark',
    
    # Real-time inference
    'StreamConfig',
    'CircularBuffer',
    'OnlinePreprocessor',
    'StreamingInference',
    'SlidingWindowProcessor',
    'LatencyMonitor',
    
    # Profiling
    'ModelProfiler',
    'LayerTimingProfiler',
    'MemoryProfiler',
    'ThroughputBenchmark',
    'ComparisonBenchmark',
    
    # Deployment
    'ModelExporter',
    'ModelPackager',
    'DeploymentConfig',
]
