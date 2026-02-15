# Phase 3: Model Architecture Development

This document describes the neural architectures implemented for NEST (Neural EEG Sequence Transducer).

## Overview

Phase 3 implements multiple sequence transduction architectures for EEG-to-text decoding. The models combine:
- **Spatial CNNs** for EEG feature extraction
- **Temporal Encoders** for sequence modeling
- **Attention Mechanisms** for alignment
- **Decoders** for text generation

## Architecture Components

### 1. Spatial CNNs (`src/models/spatial_cnn.py`)

Extract spatial features from multi-channel EEG signals.

#### SpatialCNN
Basic CNN with temporal and spatial convolutions:
- Temporal convolution: learns temporal patterns
- Spatial convolution: captures inter-electrode relationships
- Batch normalization and dropout for regularization

#### EEGNet (Lawhern et al., 2018)
Compact CNN designed for BCI:
- Temporal convolution (1×K) across time
- Depthwise spatial convolution across channels
- Separable pointwise convolution
- Very efficient: ~2,500 parameters

#### DeepConvNet (Schirrmeister et al., 2017)
Deep architecture with multiple blocks:
- Multiple convolutional blocks
- Max pooling for dimensionality reduction
- Suitable for complex feature learning

### 2. Temporal Encoders (`src/models/temporal_encoder.py`)

Model temporal dependencies in EEG sequences.

#### LSTMEncoder
Bidirectional LSTM for sequence encoding:
- Captures long-range dependencies
- Bidirectional processing (forward + backward)
- Multiple layers with dropout

#### GRUEncoder
Gated Recurrent Unit variant:
- Similar to LSTM but fewer parameters
- Faster training
- Good for shorter sequences

#### TransformerEncoder
Multi-head self-attention mechanism:
- Parallel processing (vs sequential RNNs)
- Positional encoding for sequence order
- Multi-head attention for different aspects
- Feed-forward networks

#### ConformerEncoder (Gulati et al., 2020)
Combines convolution and attention:
- Multi-head self-attention
- Convolution module for local dependencies
- Feed-forward networks
- State-of-the-art for speech recognition

### 3. Attention Mechanisms (`src/models/attention.py`)

Align EEG features with text outputs.

#### CrossAttention
Scaled dot-product cross-attention:
- Query from decoder, Key/Value from encoder
- Multi-head attention for richer representations
- Used in Transformer models

#### AdditiveAttention (Bahdanau et al., 2015)
Additive attention mechanism:
- Learns alignment through feedforward network
- More flexible than dot-product
- Classic seq2seq attention

#### LocationAwareAttention
Uses previous attention weights:
- Prevents attention collapse
- Better for monotonic alignment (EEG→text)
- Smooths attention over time

### 4. Decoders (`src/models/decoder.py`)

Generate text from encoded EEG features.

#### CTCDecoder
Connectionist Temporal Classification:
- No explicit alignment needed
- Greedy decoding
- Simple and fast
- Good baseline

#### AttentionDecoder
RNN decoder with attention:
- LSTM/GRU for text generation
- Attention over encoder states
- Teacher forcing during training
- Beam search for inference

#### TransducerDecoder
Prediction network for RNN-T:
- Predicts next token from history
- Combined with encoder via JointNetwork
- Streaming-capable

#### JointNetwork
Combines encoder and decoder for RNN-T:
- Merges encoder and decoder representations
- Outputs joint probability distribution
- Enables online decoding

## Complete NEST Architectures (`src/models/nest.py`)

### NEST_RNN_T
RNN-Transducer architecture:
```
EEG → SpatialCNN → LSTM/GRU Encoder → ┐
                                      ├→ JointNetwork → Text
      Text History → TransducerDecoder ┘
```
- Streaming-capable (online decoding)
- No attention mechanism
- Fast inference

### NEST_Transformer_T
Transformer-Transducer variant:
```
EEG → SpatialCNN → Transformer Encoder → ┐
                                         ├→ JointNetwork → Text
         Text History → TransducerDecoder ┘
```
- Parallel processing
- Better for long sequences
- More parameters

### NEST_Attention
Traditional seq2seq with attention:
```
EEG → SpatialCNN → RNN/Transformer → Attention → RNN Decoder → Text
```
- Explicit attention alignment
- Good interpretability
- Most flexible

### NEST_CTC
Simplest architecture (baseline):
```
EEG → SpatialCNN → RNN/Transformer → CTC Decoder → Text
```
- No attention or complex decoding
- Fast and simple
- Good baseline for comparison

## Training Utilities (`src/training/`)

### Trainer (`trainer.py`)
Generic training loop:
- Forward/backward passes
- Gradient clipping
- Learning rate scheduling
- Early stopping
- Checkpoint saving

### CTCTrainer
Specialized for CTC loss:
- Handles variable-length sequences
- CTC-specific loss computation

### Metrics (`metrics.py`)
Evaluation metrics:
- **WER (Word Error Rate)**: primary metric
- **CER (Character Error Rate)**: fine-grained evaluation
- **BLEU**: translation quality
- **Perplexity**: language model quality
- **Accuracy**: token-level correctness

### Checkpoint Management (`checkpoint.py`)
Model persistence:
- Save/load checkpoints
- Keep best N checkpoints
- Early stopping based on metrics
- Configuration saving

## Model Configuration (`configs/model.yaml`)

Pre-configured model variants:
- `nest_rnn_t`: LSTM-based transducer
- `nest_transformer_t`: Transformer transducer
- `nest_attention`: Attention-based seq2seq
- `nest_ctc`: CTC baseline
- `nest_conformer`: Conformer encoder (advanced)

Each config includes:
- Spatial CNN architecture
- Temporal encoder settings
- Decoder parameters
- Training hyperparameters

## Model Factory (`src/models/factory.py`)

Utilities for model creation:
```python
from src.models import ModelFactory

# Create from config file
model = ModelFactory.from_config_file(
    'configs/model.yaml',
    model_key='nest_rnn_t',
    vocab_size=5000
)

# Load pretrained
model = load_pretrained('checkpoints/best_model.pt')

# Count parameters
params = count_parameters(model)
print(f"Total parameters: {params['total']:,}")
```

## Usage Example

```python
import torch
from src.models import ModelFactory
from src.training import Trainer, get_optimizer, get_scheduler

# 1. Create model
model = ModelFactory.from_config_file(
    'configs/model.yaml',
    model_key='nest_rnn_t',
    vocab_size=5000
)

# 2. Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = get_optimizer(model, 'adamw', learning_rate=1e-4)
scheduler = get_scheduler(optimizer, 'cosine', T_max=100)
criterion = torch.nn.CTCLoss()

# 3. Train
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    scheduler=scheduler,
    clip_grad_norm=1.0
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_path='checkpoints/best_model.pt',
    early_stopping_patience=10
)
```

## Implementation Details

### Input Format
- **EEG**: `(batch_size, n_channels, seq_len)` - Raw EEG signals
- **Targets**: `(batch_size, target_len)` - Token IDs
- **Lengths**: `(batch_size,)` - Actual sequence lengths (for CTC)

### Output Format
- **CTC Models**: `(batch_size, time, vocab_size)` - Log probabilities
- **Transducer Models**: `(batch_size, time, vocab_size)` - Joint probabilities
- **Attention Models**: `(batch_size, target_len, vocab_size)` - Token probabilities

### Training Considerations

1. **Gradient Clipping**: Prevent exploding gradients (recommend: 1.0)
2. **Learning Rate**: Start small (1e-4) with warm-up
3. **Batch Size**: Limited by GPU memory (8-32 typical)
4. **Sequence Length**: EEG can be long (10-30s @ 500Hz = 5000-15000 samples)
5. **Data Augmentation**: Use Phase 2 augmentation techniques

## Performance Benchmarks

Expected performance (ZuCo dataset):
- **NEST_CTC**: WER ~60-70% (baseline)
- **NEST_Attention**: WER ~50-60%
- **NEST_RNN_T**: WER ~45-55%
- **NEST_Transformer_T**: WER ~40-50% (best)
- **NEST_Conformer**: WER ~35-45% (state-of-the-art)

Training time (single GPU):
- **NEST_CTC**: ~2-4 hours
- **NEST_RNN_T**: ~4-6 hours
- **NEST_Transformer_T**: ~6-8 hours
- **NEST_Conformer**: ~8-12 hours

## Next Steps

After Phase 3, you can:
1. Train models on preprocessed data (Phase 2 output)
2. Evaluate on test set
3. Tune hyperparameters
4. Implement beam search decoding
5. Add language model fusion
6. Deploy for real-time inference

## References

1. **EEGNet**: Lawhern et al. (2018) - "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces"
2. **DeepConvNet**: Schirrmeister et al. (2017) - "Deep learning with convolutional neural networks for EEG decoding and visualization"
3. **RNN-Transducer**: Graves (2012) - "Sequence Transduction with Recurrent Neural Networks"
4. **Transformer**: Vaswani et al. (2017) - "Attention Is All You Need"
5. **Conformer**: Gulati et al. (2020) - "Conformer: Convolution-augmented Transformer for Speech Recognition"
6. **Attention**: Bahdanau et al. (2015) - "Neural Machine Translation by Jointly Learning to Align and Translate"
