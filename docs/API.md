# NEST API Reference

Complete API documentation for all modules in the NEST framework.

## Table of Contents

1. [Data & Preprocessing](#data--preprocessing)
2. [Models](#models)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Utilities](#utilities)

---

## Data & Preprocessing

### `src.data.zuco_dataset`

#### ZucoDataset

```python
class ZucoDataset(torch.utils.data.Dataset)
```

PyTorch Dataset for loading and processing ZuCo EEG data.

**Parameters:**
- `data_dir` (str): Path to processed data directory
- `tokenizer` (Tokenizer): Tokenizer for text encoding
- `max_seq_len` (int, optional): Maximum sequence length. Default: 512
- `transform` (callable, optional): Optional transform to apply to EEG data

**Methods:**

```python
def __len__() -> int
```
Returns the number of samples in the dataset.

```python
def __getitem__(idx: int) -> Dict[str, torch.Tensor]
```
Returns a sample at the given index.

**Returns:** Dictionary with keys:
- `eeg`: EEG signal tensor (channels, time_points)
- `tokens`: Tokenized text (seq_len,)
- `lengths`: Sequence length
- `subject_id`: Subject identifier

```python
@staticmethod
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]
```
Collate function for batching samples.

**Example:**
```python
from src.data.zuco_dataset import ZucoDataset
from torch.utils.data import DataLoader

dataset = ZucoDataset(
    data_dir='data/processed/zuco/train',
    tokenizer=tokenizer,
    max_seq_len=512
)

loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=dataset.collate_fn
)
```

---

### `src.preprocessing.pipeline`

#### PreprocessingPipeline

```python
class PreprocessingPipeline
```

Complete preprocessing pipeline for EEG data.

**Parameters:**
- `config_path` (str): Path to YAML configuration file

**Methods:**

```python
def run_pipeline(
    data: np.ndarray,
    labels: List[str],
    sfreq: float,
    ch_names: List[str],
    subject_ids: np.ndarray
) -> Dict[str, Any]
```

Execute the complete preprocessing pipeline.

**Parameters:**
- `data`: Raw EEG data (n_samples, n_channels, n_timepoints)
- `labels`: Text labels for each sample
- `sfreq`: Sampling frequency in Hz
- `ch_names`: Channel names
- `subject_ids`: Subject IDs for each sample

**Returns:** Dictionary with train/val/test splits

**Example:**
```python
from src.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline('configs/preprocessing.yaml')
splits = pipeline.run_pipeline(
    data=raw_data,
    labels=text_labels,
    sfreq=500.0,
    ch_names=['Fz', 'Cz', 'Pz'],
    subject_ids=subject_ids
)
```

---

#### BandPassFilter

```python
class BandPassFilter
```

Band-pass filter for EEG signals.

**Parameters:**
- `low_freq` (float): Low cutoff frequency in Hz
- `high_freq` (float): High cutoff frequency in Hz
- `sfreq` (float): Sampling frequency in Hz
- `filter_type` (str, optional): Filter type ('butter', 'fir'). Default: 'butter'
- `order` (int, optional): Filter order. Default: 4

**Methods:**

```python
def apply(data: np.ndarray) -> np.ndarray
```

Apply band-pass filter to EEG data.

---

#### ArtifactRemover

```python
class ArtifactRemover
```

Remove artifacts using ICA.

**Parameters:**
- `method` (str, optional): ICA method ('fastica', 'infomax'). Default: 'fastica'
- `n_components` (int, optional): Number of ICA components. Default: 20
- `random_state` (int, optional): Random seed. Default: 42

**Methods:**

```python
def fit_transform(data: np.ndarray) -> np.ndarray
```

Fit ICA and remove artifacts.

---

#### ElectrodeSelector

```python
class ElectrodeSelector
```

Select most informative electrodes.

**Parameters:**
- `method` (str): Selection method ('variance', 'mutual_info', 'correlation')
- `n_electrodes` (int): Number of electrodes to select

**Methods:**

```python
def fit_transform(
    data: np.ndarray,
    ch_names: List[str]
) -> Tuple[np.ndarray, List[int]]
```

Select electrodes and return selected data and indices.

---

#### DataAugmenter

```python
class DataAugmenter
```

Data augmentation for EEG signals.

**Parameters:**
- `noise_std` (float, optional): Gaussian noise standard deviation. Default: 0.1
- `time_shift_max` (int, optional): Maximum time shift in samples. Default: 50
- `scaling_range` (Tuple[float, float], optional): Range for amplitude scaling. Default: (0.9, 1.1)

**Methods:**

```python
def augment(data: np.ndarray, n_augmentations: int = 1) -> np.ndarray
```

Apply data augmentation.

---

## Models

### `src.models.factory`

#### ModelFactory

```python
class ModelFactory
```

Factory for creating NEST models from configurations.

**Static Methods:**

```python
@staticmethod
def from_config(
    config: Dict[str, Any],
    vocab_size: int
) -> nn.Module
```

Create model from configuration dictionary.

```python
@staticmethod
def from_config_file(
    config_path: str,
    model_key: str,
    vocab_size: int
) -> nn.Module
```

Create model from YAML configuration file.

**Parameters:**
- `config_path`: Path to YAML config
- `model_key`: Key for specific model configuration
- `vocab_size`: Vocabulary size

**Example:**
```python
from src.models import ModelFactory

model = ModelFactory.from_config_file(
    'configs/model.yaml',
    model_key='nest_conformer',
    vocab_size=5000
)
```

**Supported Models:**
- `nest_rnn_t`: RNN Transducer
- `nest_transformer_t`: Transformer Transducer
- `nest_attention`: Attention-based model
- `nest_ctc`: CTC-based model
- `nest_conformer`: Conformer-based model

---

### `src.models.nest`

#### NestRNNTransducer

```python
class NestRNNTransducer(nn.Module)
```

RNN Transducer architecture for EEG-to-text.

**Parameters:**
- `n_channels` (int): Number of EEG channels
- `n_timepoints` (int): Number of time points
- `vocab_size` (int): Size of vocabulary
- `d_model` (int, optional): Model dimension. Default: 256
- `n_encoder_layers` (int, optional): Number of encoder layers. Default: 4
- `n_decoder_layers` (int, optional): Number of decoder layers. Default: 4
- `dropout` (float, optional): Dropout rate. Default: 0.1

**Methods:**

```python
def forward(
    eeg: torch.Tensor,
    text: torch.Tensor = None,
    eeg_lengths: torch.Tensor = None,
    text_lengths: torch.Tensor = None
) -> torch.Tensor
```

Forward pass through the model.

---

#### NestTransformerTransducer

```python
class NestTransformerTransducer(nn.Module)
```

Transformer Transducer architecture.

**Parameters:** Similar to NestRNNTransducer, plus:
- `n_heads` (int, optional): Number of attention heads. Default: 8
- `d_ff` (int, optional): Feed-forward dimension. Default: 1024

---

#### NestAttention

```python
class NestAttention(nn.Module)
```

Attention-based encoder-decoder architecture.

**Parameters:** Similar to NestTransformerTransducer

---

### `src.models.spatial_cnn`

#### SpatialCNN

```python
class SpatialCNN(nn.Module)
```

Base class for spatial CNN models.

---

#### EEGNet

```python
class EEGNet(SpatialCNN)
```

EEGNet architecture for EEG feature extraction.

**Parameters:**
- `n_channels` (int): Number of EEG channels
- `n_timepoints` (int): Number of time points
- `f1` (int, optional): Number of temporal filters. Default: 8
- `d` (int, optional): Depth multiplier. Default: 2
- `f2` (int, optional): Number of pointwise filters. Default: 16
- `kernel_size` (int, optional): Temporal kernel size. Default: 64
- `dropout` (float, optional): Dropout rate. Default: 0.5

---

#### DeepConvNet

```python
class DeepConvNet(SpatialCNN)
```

DeepConvNet architecture.

**Parameters:**
- `n_channels` (int): Number of EEG channels
- `n_timepoints` (int): Number of time points
- `n_filters` (int, optional): Base number of filters. Default: 25
- `dropout` (float, optional): Dropout rate. Default: 0.5

---

### `src.models.temporal_encoder`

#### TemporalEncoder

```python
class TemporalEncoder(nn.Module)
```

Base class for temporal encoders.

---

#### LSTMEncoder

```python
class LSTMEncoder(TemporalEncoder)
```

LSTM-based temporal encoder.

**Parameters:**
- `input_dim` (int): Input dimension
- `hidden_dim` (int): Hidden dimension
- `n_layers` (int, optional): Number of layers. Default: 2
- `bidirectional` (bool, optional): Use bidirectional LSTM. Default: True
- `dropout` (float, optional): Dropout rate. Default: 0.1

---

#### TransformerEncoder

```python
class TransformerEncoder(TemporalEncoder)
```

Transformer-based temporal encoder.

**Parameters:**
- `d_model` (int): Model dimension
- `n_heads` (int): Number of attention heads
- `n_layers` (int): Number of transformer layers
- `d_ff` (int): Feed-forward dimension
- `dropout` (float, optional): Dropout rate. Default: 0.1

---

#### ConformerEncoder

```python
class ConformerEncoder(TemporalEncoder)
```

Conformer encoder (Convolution-augmented Transformer).

**Parameters:** Similar to TransformerEncoder, plus:
- `conv_kernel_size` (int, optional): Convolution kernel size. Default: 31

---

### `src.models.attention`

#### CrossAttention

```python
class CrossAttention(nn.Module)
```

Cross-attention mechanism between EEG and text.

**Parameters:**
- `d_model` (int): Model dimension
- `n_heads` (int, optional): Number of heads. Default: 8
- `dropout` (float, optional): Dropout rate. Default: 0.1

**Methods:**

```python
def forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Compute cross-attention. Returns attention output and weights.

---

### `src.models.advanced_attention`

#### RelativePositionAttention

```python
class RelativePositionAttention(nn.Module)
```

Attention with relative position encoding.

---

#### LocalAttention

```python
class LocalAttention(nn.Module)
```

Local attention with limited receptive field.

**Parameters:**
- `window_size` (int): Attention window size

---

#### LinearAttention

```python
class LinearAttention(nn.Module)
```

Linear complexity attention mechanism.

---

### `src.models.adaptation`

#### SubjectAdapter

```python
class SubjectAdapter(nn.Module)
```

Subject-specific adaptation layer.

**Parameters:**
- `base_model` (nn.Module): Base NEST model
- `n_subjects` (int): Number of subjects
- `embedding_dim` (int, optional): Subject embedding dimension. Default: 64

**Methods:**

```python
def forward(
    eeg: torch.Tensor,
    subject_ids: torch.Tensor
) -> torch.Tensor
```

Forward pass with subject adaptation.

---

#### DomainAdversarialNetwork

```python
class DomainAdversarialNetwork(nn.Module)
```

DANN for domain adaptation.

**Parameters:**
- `feature_extractor` (nn.Module): Feature extraction network
- `classifier` (nn.Module): Classification network
- `n_domains` (int): Number of domains

**Methods:**

```python
def forward(
    x: torch.Tensor,
    labels: torch.Tensor,
    domain_ids: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]
```

Returns (classification_loss, domain_loss).

---

## Training

### `src.training.trainer`

#### Trainer

```python
class Trainer
```

Training manager for NEST models.

**Parameters:**
- `model` (nn.Module): Model to train
- `optimizer` (torch.optim.Optimizer): Optimizer
- `criterion` (nn.Module): Loss function
- `device` (torch.device): Device for training
- `scheduler` (optional): Learning rate scheduler
- `clip_grad_norm` (float, optional): Gradient clipping threshold
- `logger` (optional): Logger (wandb, tensorboard)

**Methods:**

```python
def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    save_path: str = None,
    early_stopping_patience: int = None,
    log_interval: int = 10
) -> Dict[str, List[float]]
```

Train the model.

**Returns:** Training history dictionary

```python
def evaluate(
    test_loader: DataLoader
) -> Dict[str, float]
```

Evaluate the model.

**Returns:** Evaluation metrics

**Example:**
```python
from src.training import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_path='checkpoints/best.pt'
)
```

---

#### get_optimizer

```python
def get_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer
```

Create optimizer.

**Parameters:**
- `model`: Model to optimize
- `optimizer_type`: Type ('adam', 'adamw', 'sgd', 'rmsprop')
- `learning_rate`: Learning rate
- `**kwargs`: Additional optimizer arguments

---

#### get_scheduler

```python
def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler
```

Create learning rate scheduler.

**Parameters:**
- `optimizer`: Optimizer
- `scheduler_type`: Type ('step', 'cosine', 'plateau', 'exponential')
- `**kwargs`: Additional scheduler arguments

---

### `src.training.metrics`

#### MetricTracker

```python
class MetricTracker
```

Track training metrics.

**Methods:**

```python
def register_metric(name: str, metric_fn: callable)
```

Register a custom metric.

```python
def update(name: str, predictions: torch.Tensor, targets: torch.Tensor)
```

Update metric.

```python
def get_metric(name: str) -> float
```

Get current metric value.

```python
def reset()
```

Reset all metrics.

---

#### compute_wer

```python
def compute_wer(
    predictions: List[str],
    references: List[str]
) -> float
```

Compute Word Error Rate.

---

#### compute_cer

```python
def compute_cer(
    predictions: List[str],
    references: List[str]
) -> float
```

Compute Character Error Rate.

---

#### compute_bleu

```python
def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4
) -> float
```

Compute BLEU score.

---

### `src.training.checkpoint`

#### CheckpointManager

```python
class CheckpointManager
```

Manage model checkpoints.

**Parameters:**
- `save_dir` (str): Directory to save checkpoints
- `max_checkpoints` (int, optional): Maximum checkpoints to keep. Default: 5

**Methods:**

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    is_best: bool = False
)
```

Save a checkpoint.

```python
def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Dict[str, Any]
```

Load a checkpoint.

---

## Evaluation

### `src.evaluation.benchmark`

#### EvaluationPipeline

```python
class EvaluationPipeline
```

Complete evaluation pipeline.

**Parameters:**
- `model` (nn.Module): Model to evaluate
- `tokenizer` (Tokenizer): Tokenizer
- `device` (torch.device): Device

**Methods:**

```python
def evaluate(
    test_loader: DataLoader,
    beam_size: int = 1,
    max_length: int = 512,
    metrics: List[str] = ['wer', 'cer', 'bleu']
) -> Dict[str, float]
```

Run comprehensive evaluation.

---

### `src.evaluation.beam_search`

#### BeamSearchDecoder

```python
class BeamSearchDecoder
```

Beam search decoder.

**Parameters:**
- `model` (nn.Module): Model
- `tokenizer` (Tokenizer): Tokenizer
- `beam_size` (int, optional): Beam size. Default: 5
- `max_length` (int, optional): Maximum length. Default: 512
- `length_penalty` (float, optional): Length penalty. Default: 0.6
- `coverage_penalty` (float, optional): Coverage penalty. Default: 0.0
- `eos_token_id` (int): End-of-sequence token ID

**Methods:**

```python
def decode(
    eeg_input: torch.Tensor,
    return_scores: bool = False,
    return_attention: bool = False,
    n_best: int = 1
) -> List[Dict[str, Any]]
```

Decode with beam search.

---

### `src.evaluation.profiling`

#### ModelProfiler

```python
class ModelProfiler
```

Profile model performance.

**Methods:**

```python
def profile_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...]
) -> int
```

Count FLOPs.

```python
def profile_memory(
    model: nn.Module,
    input_data: torch.Tensor
) -> Dict[str, float]
```

Profile memory usage.

```python
def profile_latency(
    model: nn.Module,
    input_data: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, float]
```

Profile inference latency.

---

### `src.evaluation.pruning`

#### MagnitudePruner

```python
class MagnitudePruner
```

Magnitude-based pruning.

**Parameters:**
- `model` (nn.Module): Model to prune
- `amount` (float): Pruning amount (0-1)

**Methods:**

```python
def prune() -> nn.Module
```

Apply pruning.

---

### `src.evaluation.quantization`

#### PostTrainingQuantizer

```python
class PostTrainingQuantizer
```

Post-training quantization.

**Parameters:**
- `model` (nn.Module): Model to quantize
- `quantization_type` (str): Type ('dynamic', 'static', 'qat')

**Methods:**

```python
def quantize(
    calibration_loader: DataLoader = None
) -> nn.Module
```

Quantize model.

---

## Utilities

### `src.utils.tokenizer`

#### CharTokenizer

```python
class CharTokenizer
```

Character-level tokenizer.

**Parameters:**
- `vocab` (List[str]): Vocabulary

**Methods:**

```python
def encode(text: str) -> List[int]
```

Encode text to token IDs.

```python
def decode(tokens: List[int]) -> str
```

Decode token IDs to text.

---

#### SubwordTokenizer

```python
class SubwordTokenizer
```

Subword tokenizer (BPE/SentencePiece).

**Parameters:**
- `model_path` (str): Path to tokenizer model

**Methods:**

```python
def train(
    texts: List[str],
    vocab_size: int,
    model_type: str = 'bpe'
)
```

Train tokenizer.

```python
def encode(text: str) -> List[int]
```

Encode text.

```python
def decode(tokens: List[int]) -> str
```

Decode tokens.

---

For more examples and usage, see [USAGE.md](USAGE.md).
