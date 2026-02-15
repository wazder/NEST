# NEST Usage Guide

Complete guide for using the NEST framework for EEG-to-text decoding.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Evaluation](#evaluation)
5. [Optimization](#optimization)
6. [Deployment](#deployment)
7. [Advanced Usage](#advanced-usage)

## Quick Start

### 1. Basic Pipeline

The fastest way to get started with NEST:

```python
import torch
from src.preprocessing import PreprocessingPipeline
from src.models import ModelFactory
from src.training import Trainer
from src.utils.tokenizer import CharTokenizer

# 1. Preprocess data
pipeline = PreprocessingPipeline('configs/preprocessing.yaml')
splits = pipeline.run_pipeline(
    data=raw_eeg_data,
    labels=text_labels,
    sfreq=500.0,
    ch_names=['Fz', 'Cz', 'Pz', 'Oz'],
    subject_ids=subject_ids
)

# 2. Create tokenizer
tokenizer = CharTokenizer(vocab=list('abcdefghijklmnopqrstuvwxyz '))

# 3. Create model
model = ModelFactory.from_config_file(
    'configs/model.yaml',
    model_key='nest_attention',
    vocab_size=len(tokenizer)
)

# 4. Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(model=model, device=device)
history = trainer.train(
    train_loader=splits['train'],
    val_loader=splits['val'],
    epochs=50
)
```

## Data Preparation

### Loading ZuCo Dataset

```python
from src.data.zuco_dataset import ZucoDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = ZucoDataset(
    data_dir='data/processed/zuco/train',
    tokenizer=tokenizer,
    max_seq_len=512
)

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=dataset.collate_fn
)

# Iterate through batches
for batch in loader:
    eeg_signals = batch['eeg']        # (batch, channels, time)
    text_tokens = batch['tokens']     # (batch, seq_len)
    lengths = batch['lengths']        # (batch,)
    break
```

### Preprocessing Pipeline

#### Configuration-based Preprocessing

Create `configs/preprocessing.yaml`:

```yaml
preprocessing:
  # Filtering parameters
  low_freq: 0.5
  high_freq: 50.0
  notch_freq: 60.0
  
  # ICA parameters
  ica_method: 'fastica'
  n_components: 20
  
  # Electrode selection
  electrode_selection:
    method: 'variance'  # or 'mutual_info', 'correlation'
    n_electrodes: 32
  
  # Augmentation
  augmentation:
    enabled: true
    noise_std: 0.1
    time_shift_max: 50  # ms
    scaling_range: [0.9, 1.1]
  
  # Train/val/test split
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
  
  # Subject-independent evaluation
  subject_independent: true
```

Run preprocessing:

```python
from src.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline('configs/preprocessing.yaml')

# Process raw data
processed_data = pipeline.run_pipeline(
    data=raw_eeg,           # shape: (n_samples, n_channels, n_timepoints)
    labels=text_labels,     # list of strings
    sfreq=500.0,           # sampling frequency in Hz
    ch_names=channel_names, # list of channel names
    subject_ids=subjects    # list of subject IDs
)

# Access splits
train_data = processed_data['train']
val_data = processed_data['val']
test_data = processed_data['test']
```

#### Manual Preprocessing Steps

```python
from src.preprocessing import (
    BandPassFilter,
    ArtifactRemover,
    ElectrodeSelector,
    DataAugmenter
)

# 1. Band-pass filtering
filter = BandPassFilter(low_freq=0.5, high_freq=50.0, sfreq=500.0)
filtered_data = filter.apply(raw_eeg)

# 2. Artifact removal with ICA
artifact_remover = ArtifactRemover(method='ica', n_components=20)
clean_data = artifact_remover.fit_transform(filtered_data)

# 3. Electrode selection
selector = ElectrodeSelector(method='variance', n_electrodes=32)
selected_data, selected_indices = selector.fit_transform(clean_data, ch_names)

# 4. Data augmentation
augmenter = DataAugmenter(
    noise_std=0.1,
    time_shift_max=50,
    scaling_range=(0.9, 1.1)
)
augmented_data = augmenter.augment(selected_data)
```

## Model Training

### Basic Training

```python
from src.models import ModelFactory
from src.training import Trainer, get_optimizer, get_scheduler
import torch

# Create model
model = ModelFactory.from_config_file(
    'configs/model.yaml',
    model_key='nest_rnn_t',
    vocab_size=5000
)

# Setup optimizer and scheduler
optimizer = get_optimizer(model, 'adamw', learning_rate=1e-4, weight_decay=0.01)
scheduler = get_scheduler(optimizer, 'cosine', T_max=100, eta_min=1e-6)

# Create trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=torch.nn.CTCLoss(),
    device=device,
    scheduler=scheduler,
    clip_grad_norm=1.0
)

# Train
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_path='checkpoints/best_model.pt',
    early_stopping_patience=10,
    log_interval=10
)

# Access training history
print(f"Best validation loss: {min(history['val_loss'])}")
print(f"Best WER: {min(history['val_wer'])}")
```

### Model Configuration

Create `configs/model.yaml`:

```yaml
# Model architecture
model:
  # Spatial CNN configuration
  spatial_cnn:
    type: 'eegnet'  # or 'deepconvnet', 'shallow'
    n_channels: 32
    n_timepoints: 1000
    dropout: 0.5
  
  # Temporal encoder configuration
  temporal_encoder:
    type: 'conformer'  # or 'lstm', 'gru', 'transformer'
    d_model: 256
    n_layers: 6
    n_heads: 8
    d_ff: 1024
    dropout: 0.1
  
  # Attention mechanism
  attention:
    type: 'relative_position'  # or 'additive', 'local', 'linear'
    d_model: 256
    n_heads: 8
  
  # Decoder configuration
  decoder:
    type: 'transformer'
    d_model: 256
    n_layers: 6
    n_heads: 8
    d_ff: 1024
    dropout: 0.1
  
  # Loss configuration
  loss:
    type: 'ctc'  # or 'seq2seq', 'rnn_t'
    blank_idx: 0

# Training configuration
training:
  batch_size: 16
  learning_rate: 1e-4
  weight_decay: 0.01
  max_epochs: 100
  early_stopping_patience: 10
  clip_grad_norm: 1.0
  
  # Optimizer
  optimizer:
    type: 'adamw'
    betas: [0.9, 0.999]
    eps: 1e-8
  
  # Scheduler
  scheduler:
    type: 'cosine'
    T_max: 100
    eta_min: 1e-6
```

### Advanced Training Features

#### Subject Adaptation

```python
from src.models.adaptation import SubjectAdapter, DomainAdversarialNetwork

# Method 1: Subject embeddings
adapter = SubjectAdapter(
    base_model=model,
    n_subjects=20,
    embedding_dim=64
)

# Train with subject embeddings
for eeg, text, subject_ids in train_loader:
    output = adapter(eeg, subject_ids)
    loss = criterion(output, text)
    loss.backward()

# Method 2: Domain adversarial training (DANN)
dann = DomainAdversarialNetwork(
    feature_extractor=model.encoder,
    classifier=model.decoder,
    n_domains=20  # number of subjects
)

# Train with domain adaptation
for eeg, text, domain_ids in train_loader:
    class_loss, domain_loss = dann(eeg, text, domain_ids)
    total_loss = class_loss + 0.1 * domain_loss
    total_loss.backward()
```

#### Robustness Training

```python
from src.training.robustness import (
    AdversarialTrainer,
    DenoisingTrainer,
    RobustLoss
)

# Adversarial training
adv_trainer = AdversarialTrainer(
    model=model,
    epsilon=0.01,
    alpha=0.005,
    num_steps=5
)

# Denoising autoencoder pre-training
denoising_trainer = DenoisingTrainer(
    model=model.encoder,
    noise_std=0.2
)
denoising_trainer.pretrain(train_loader, epochs=20)

# Robust loss functions
robust_loss = RobustLoss(
    base_loss=torch.nn.CTCLoss(),
    loss_type='huber',  # or 'cauchy', 'welsch'
    delta=1.0
)
```

#### Language Model Integration

```python
from src.models.language_model import (
    LanguageModelFusion,
    LanguageModelRescorer
)

# Shallow fusion (during decoding)
lm_fusion = LanguageModelFusion(
    seq2seq_model=model,
    lm_model='gpt2',
    lm_weight=0.3
)

# Deep fusion (during training)
from transformers import GPT2LMHeadModel
lm = GPT2LMHeadModel.from_pretrained('gpt2')
model_with_lm = LanguageModelFusion(
    seq2seq_model=model,
    lm_model=lm,
    fusion_type='deep',
    lm_weight=0.3
)

# LM rescoring (post-processing)
rescorer = LanguageModelRescorer(
    lm_model='gpt2',
    lm_weight=0.5
)
rescored_hypotheses = rescorer.rescore(beam_search_results)
```

## Evaluation

### Benchmark Evaluation

```python
from src.evaluation.benchmark import EvaluationPipeline

# Create evaluation pipeline
evaluator = EvaluationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Run comprehensive evaluation
results = evaluator.evaluate(
    test_loader=test_loader,
    beam_size=5,
    max_length=512,
    metrics=['wer', 'cer', 'bleu']
)

print(f"Word Error Rate: {results['wer']:.2f}%")
print(f"Character Error Rate: {results['cer']:.2f}%")
print(f"BLEU Score: {results['bleu']:.2f}")
print(f"Inference Time: {results['inference_time_ms']:.2f} ms")
```

### Command-line Evaluation

```bash
# Basic evaluation
python -m src.evaluation.benchmark \
    --model_config configs/model.yaml \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/zuco/test \
    --tokenizer vocab/tokenizer.json \
    --output results/benchmark.json

# With beam search
python -m src.evaluation.benchmark \
    --model_config configs/model.yaml \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/zuco/test \
    --beam_size 10 \
    --length_penalty 0.6 \
    --coverage_penalty 0.2 \
    --output results/beam_search.json

# Detailed profiling
python -m src.evaluation.profiling \
    --model_config configs/model.yaml \
    --checkpoint checkpoints/best_model.pt \
    --batch_size 1 \
    --profile_memory \
    --profile_flops \
    --output results/profile.json
```

### Beam Search Decoding

```python
from src.evaluation.beam_search import BeamSearchDecoder

# Create beam search decoder
decoder = BeamSearchDecoder(
    model=model,
    tokenizer=tokenizer,
    beam_size=10,
    max_length=512,
    length_penalty=0.6,
    coverage_penalty=0.2,
    eos_token_id=tokenizer.eos_id
)

# Decode with beam search
hypotheses = decoder.decode(
    eeg_input=test_eeg,
    return_scores=True,
    return_attention=True,
    n_best=5
)

# Access results
best_hypothesis = hypotheses[0]
print(f"Text: {best_hypothesis['text']}")
print(f"Score: {best_hypothesis['score']:.3f}")
print(f"Attention weights: {best_hypothesis['attention'].shape}")
```

## Optimization

### Model Pruning

```python
from src.evaluation.pruning import (
    MagnitudePruner,
    StructuredPruner,
    IterativePruner
)

# Magnitude-based pruning
pruner = MagnitudePruner(model, amount=0.5)
pruned_model = pruner.prune()

# Structured pruning (filter-level)
structured_pruner = StructuredPruner(
    model,
    amount=0.3,
    pruning_dim=0  # prune output channels
)
structured_model = structured_pruner.prune()

# Iterative pruning with fine-tuning
iterative_pruner = IterativePruner(
    model=model,
    trainer=trainer,
    train_loader=train_loader,
    val_loader=val_loader,
    pruning_schedule=[0.2, 0.4, 0.5, 0.6],
    finetune_epochs=5
)
final_model = iterative_pruner.prune()
```

### Model Quantization

```python
from src.evaluation.quantization import (
    PostTrainingQuantizer,
    QuantizationAwareTrainer
)

# Post-training quantization (PTQ)
ptq = PostTrainingQuantizer(
    model=model,
    quantization_type='dynamic'  # or 'static', 'qat'
)
quantized_model = ptq.quantize(calibration_loader=val_loader)

# Quantization-aware training (QAT)
qat_trainer = QuantizationAwareTrainer(
    model=model,
    qconfig='fbgemm'  # Facebook quantization config
)
qat_model = qat_trainer.prepare_qat()
# Train normally
qat_trainer.train(train_loader, val_loader, epochs=10)
qat_model = qat_trainer.convert()

# Mixed-precision quantization
from src.evaluation.quantization import MixedPrecisionQuantizer
mixed_quantizer = MixedPrecisionQuantizer(
    model=model,
    sensitive_layers=['encoder.layer.0', 'decoder.layer.0'],
    sensitive_precision='fp16',
    default_precision='int8'
)
mixed_model = mixed_quantizer.quantize()
```

### Inference Optimization

```python
from src.evaluation.inference_optimizer import InferenceOptimizer

# Create optimizer
optimizer = InferenceOptimizer(model)

# Mixed precision (FP16)
fp16_model = optimizer.to_fp16()

# TorchScript compilation
scripted_model = optimizer.to_torchscript(example_input=dummy_eeg)

# ONNX export
optimizer.to_onnx(
    output_path='models/nest.onnx',
    example_input=dummy_eeg,
    opset_version=14
)

# Optimize model (fusion + other optimizations)
optimized_model = optimizer.optimize(fusion=True, inplace=True)

# Benchmark optimizations
results = optimizer.benchmark(
    input_data=test_data,
    batch_sizes=[1, 4, 16],
    num_runs=100
)
print(f"FP32 latency: {results['fp32']['latency_ms']:.2f} ms")
print(f"FP16 latency: {results['fp16']['latency_ms']:.2f} ms")
print(f"Speedup: {results['fp32']['latency_ms'] / results['fp16']['latency_ms']:.2f}x")
```

### Real-time Streaming Inference

```python
from src.evaluation.realtime_inference import StreamingInference

# Create streaming inference engine
streaming = StreamingInference(
    model=optimized_model,
    tokenizer=tokenizer,
    chunk_size=100,  # ms
    overlap=20,      # ms
    device=device
)

# Process streaming EEG
text_stream = []
for eeg_chunk in eeg_stream:
    # Process chunk (should be < 100ms)
    text_token = streaming.process_chunk(eeg_chunk)
    if text_token:
        text_stream.append(text_token)

# Get final result
full_text = streaming.finalize()
print(f"Decoded text: {full_text}")
print(f"Average latency: {streaming.get_avg_latency():.2f} ms")
```

## Deployment

### Model Export

```python
from src.evaluation.deployment import ModelDeployer

# Create deployer
deployer = ModelDeployer(model, tokenizer)

# Export for production
deployer.export(
    output_dir='deployment/',
    formats=['torchscript', 'onnx'],
    quantize=True,
    optimize=True
)

# Generated files:
# - deployment/model.pt (TorchScript)
# - deployment/model.onnx (ONNX)
# - deployment/tokenizer.json
# - deployment/config.yaml
# - deployment/README.md
```

### Loading Deployed Model

```python
import torch

# Load TorchScript model
model = torch.jit.load('deployment/model.pt')
model.eval()

# Run inference
with torch.no_grad():
    output = model(eeg_input)
```

### Production API Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load model once at startup
model = torch.jit.load('deployment/model.pt')
model.eval()

class EEGInput(BaseModel):
    data: list  # EEG data as list
    sampling_rate: float

@app.post("/decode")
async def decode_eeg(input: EEGInput):
    # Convert to tensor
    eeg_tensor = torch.tensor(input.data).float().unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(eeg_tensor)
    
    # Decode tokens to text
    text = tokenizer.decode(output[0].argmax(dim=-1))
    
    return {"text": text, "confidence": output[0].max().item()}

# Run: uvicorn api:app --host 0.0.0.0 --port 8000
```

## Advanced Usage

### Custom Model Architecture

```python
from src.models.nest import NestBase
import torch.nn as nn

class CustomNEST(NestBase):
    def __init__(self, n_channels, vocab_size, d_model=256):
        super().__init__()
        
        # Custom encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.LSTM(128, d_model, num_layers=2, bidirectional=True)
        )
        
        # Custom decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model * 2, nhead=8),
            num_layers=6
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model * 2, vocab_size)
    
    def forward(self, eeg, text=None):
        # Encode EEG
        encoded = self.encoder(eeg)
        
        # Decode to text
        if text is not None:
            decoded = self.decoder(text, encoded)
        else:
            decoded = self.greedy_decode(encoded)
        
        # Project to vocabulary
        logits = self.output_projection(decoded)
        return logits

# Use custom model
custom_model = CustomNEST(n_channels=32, vocab_size=5000)
```

### Custom Metrics

```python
from src.training.metrics import MetricTracker

class CustomMetric:
    def __init__(self, name):
        self.name = name
    
    def __call__(self, predictions, targets):
        # Implement custom metric logic
        score = compute_custom_score(predictions, targets)
        return score

# Register custom metric
tracker = MetricTracker()
tracker.register_metric('custom', CustomMetric('custom'))

# Use during training
tracker.update('custom', predictions, targets)
print(f"Custom metric: {tracker.get_metric('custom')}")
```

### Experiment Tracking

```python
from src.training.trainer import Trainer
import wandb

# Initialize W&B
wandb.init(
    project='nest-eeg-decoding',
    config={
        'learning_rate': 1e-4,
        'batch_size': 16,
        'model': 'nest_conformer'
    }
)

# Train with logging
trainer = Trainer(
    model=model,
    device=device,
    logger=wandb
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)

# W&B will automatically log metrics, gradients, and model weights
```

## Tips and Best Practices

1. **Start small**: Begin with a simple model (e.g., NEST-Attention) before trying complex architectures
2. **Monitor overfitting**: Use validation metrics and early stopping
3. **Subject independence**: Always evaluate on held-out subjects
4. **Augmentation**: Use data augmentation to improve generalization
5. **Hyperparameter tuning**: Use grid search or Bayesian optimization
6. **Model ensemble**: Combine multiple models for better performance
7. **Regular checkpointing**: Save checkpoints frequently during training
8. **Profile first**: Profile your model before optimizing
9. **Incremental optimization**: Apply optimizations one at a time and measure impact
10. **Version control**: Track experiments with W&B or TensorBoard

## Troubleshooting

See [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

## Further Reading

- [Phase 2: Preprocessing](phase2-preprocessing.md)
- [Phase 3: Model Architecture](phase3-models.md)
- [Phase 4: Advanced Features](phase4-advanced-features.md)
- [Phase 5: Evaluation & Optimization](phase5-evaluation-optimization.md)
- [API Reference](API.md)
