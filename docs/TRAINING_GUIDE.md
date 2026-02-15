# Training NEST Models on ZuCo Dataset

This guide provides step-by-step instructions for training NEST models on the ZuCo dataset and reproducing the results from the research paper.

## Quick Start

```bash
# 1. Download and preprocess ZuCo data
python scripts/train_zuco_full.py --config configs/model.yaml --output results/ --download

# 2. Train models (uses cached preprocessed data)
python scripts/train_zuco_full.py --config configs/model.yaml --output results/

# 3. Evaluate models
python scripts/evaluate_models.py --results results/ --output results/evaluation/
```

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended (CUDA 11.7+)
- **RAM**: 32GB+ recommended
- **Disk**: 50GB+ free space for dataset and models

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install NEST package
pip install -e .
```

## Data Preparation

### Option 1: Automatic Download (Recommended)

```bash
python scripts/train_zuco_full.py --download --config configs/model.yaml --output results/
```

This will:
1. Download ZuCo dataset from OSF repository (~5GB)
2. Extract and organize files
3. Preprocess EEG data (filtering, ICA, normalization)
4. Create train/val/test splits
5. Save processed data to `data/processed/zuco/`

**Note**: Download and preprocessing takes ~2-4 hours depending on your system.

### Option 2: Manual Download

If automatic download fails or you have limited bandwidth:

1. Download ZuCo dataset manually from: https://osf.io/q3zws/
2. Extract to `data/raw/zuco/`
3. Run preprocessing only:
   ```bash
   python scripts/preprocess_zuco.py --input data/raw/zuco/ --output data/processed/zuco/
   ```

## Training Configuration

Edit `configs/model.yaml` to customize training:

```yaml
# Model Architecture
model_variants:
  - nest_attention      # Best performance
  - nest_conformer      # State-of-the-art
  - nest_rnn_t          # Streaming capable
  - nest_ctc            # Fastest inference

# Training Hyperparameters
batch_size: 16
learning_rate: 1e-4
epochs: 100
early_stopping_patience: 10

# Data
n_channels: 32
max_seq_len: 256
tokenizer: char  # or 'bpe'

# Optimization
use_mixed_precision: true
gradient_accumulation_steps: 2
clip_grad_norm: 1.0
```

## Training Models

### Train All Variants

```bash
python scripts/train_zuco_full.py \
    --config configs/model.yaml \
    --output results/full_training/
```

This will train all model variants specified in the config and save:
- Model checkpoints: `results/full_training/checkpoints/<variant>/`
- Training history: `results/full_training/checkpoints/<variant>/history.json`
- Best models: `results/full_training/checkpoints/<variant>/best_model.pt`
- Results summary: `results/full_training/results.json`

**Estimated Time**: 8-12 hours per model on NVIDIA A100

### Train Single Model

```bash
python examples/01_basic_training.py \
    --model nest_conformer \
    --output results/single_model/
```

### Resume Training from Checkpoint

```bash
python scripts/train_zuco_full.py \
    --config configs/model.yaml \
    --output results/ \
    --resume results/checkpoints/nest_conformer/checkpoint_epoch_20.pt
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir results/full_training/logs/
```

Open http://localhost:6006 to view:
- Training/validation loss curves
- Learning rate schedule
- Gradient norms
- Attention visualizations

### Weights & Biases (Optional)

```bash
# Login to W&B
wandb login

# Enable W&B logging in config
echo "use_wandb: true" >> configs/model.yaml

# Train with logging
python scripts/train_zuco_full.py --config configs/model.yaml --output results/
```

## Evaluation

### Evaluate on Test Set

```bash
python scripts/evaluate_models.py \
    --checkpoint results/checkpoints/nest_conformer/best_model.pt \
    --test_data data/processed/zuco/test/ \
    --output results/evaluation/
```

Output:
- WER, CER, BLEU scores
- Per-sample predictions: `results/evaluation/predictions.csv`
- Error analysis: `results/evaluation/error_analysis.json`
- Inference time benchmark: `results/evaluation/benchmark.json`

### Benchmark All Models

```bash
python scripts/benchmark_all.py \
    --checkpoints_dir results/checkpoints/ \
    --output results/benchmark/
```

Generates comparison table:
```
| Model           | WER   | CER  | BLEU | Inference (ms) |
|-----------------|-------|------|------|----------------|
| nest_conformer  | 15.8% | 8.3% | 0.76 | 18.2           |
| nest_transformer| 18.1% | 9.6% | 0.71 | 16.5           |
| nest_rnn_t      | 19.7% | 10.4%| 0.67 | 22.1           |
```

## Subject Adaptation

### Train with Subject Embeddings

```bash
python scripts/train_with_adaptation.py \
    --base_model results/checkpoints/nest_conformer/best_model.pt \
    --adaptation_method subject_embeddings \
    --calibration_data data/processed/zuco/calibration/ \
    --output results/adapted_models/
```

### Train with DANN

```bash
python scripts/train_with_adaptation.py \
    --base_model results/checkpoints/nest_conformer/best_model.pt \
    --adaptation_method dann \
    --output results/dann_models/
```

## Model Optimization

### Pruning

```bash
python -m src.evaluation.pruning \
    --model results/checkpoints/nest_conformer/best_model.pt \
    --sparsity 0.4 \
    --output results/pruned_models/
```

### Quantization

```bash
python -m src.evaluation.quantization \
    --model results/checkpoints/nest_conformer/best_model.pt \
    --precision int8 \
    --output results/quantized_models/
```

### Combined Optimization

```bash
python scripts/optimize_for_deployment.py \
    --model results/checkpoints/nest_conformer/best_model.pt \
    --sparsity 0.4 \
    --precision int8 \
    --output results/optimized_models/
```

Result:
- Original: 16.8 MB, 41 ms inference, 15.8% WER
- Optimized: 2.5 MB, 13 ms inference, 16.5% WER

## Expected Results

Reproducing the paper results should yield approximately:

| Metric | Expected Value | Variance |
|--------|----------------|----------|
| WER    | 15.8%          | ±1.2%    |
| CER    | 8.3%           | ±0.8%    |
| BLEU   | 0.76           | ±0.03    |
| Inference | 18 ms       | ±3 ms    |

**Note**: Results may vary slightly due to:
- Hardware differences
- CUDA version
- Random seed variance (despite fixing seeds, some operations are non-deterministic)

To minimize variance:
- Use same CUDA version (11.7)
- Run multiple seeds and average
- Disable cudnn benchmarking: `torch.backends.cudnn.benchmark = False`

## Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Reduce batch size
```yaml
# In configs/model.yaml
batch_size: 8  # Reduce from 16
gradient_accumulation_steps: 4  # Compensate with gradient accumulation
```

**Solution 2**: Enable gradient checkpointing
```python
# In model creation
model = ModelFactory.create(..., use_gradient_checkpointing=True)
```

**Solution 3**: Use CPU offloading
```bash
python scripts/train_zuco_full.py --config configs/model.yaml --cpu_offload
```

### Slow Training

**Solution 1**: Enable mixed precision
```yaml
use_mixed_precision: true
```

**Solution 2**: Use DataLoader num_workers
```yaml
num_workers: 8  # Increase based on CPU cores
```

**Solution 3**: Reduce data augmentation
```yaml
augmentation_prob: 0.1  # Reduce from 0.3
```

### Poor Performance

**Debugging checklist**:
1. Check preprocessed data integrity:
   ```bash
   python scripts/verify_data.py --data data/processed/zuco/
   ```

2. Verify model architecture:
   ```python
   from src.models import ModelFactory
   model = ModelFactory.create('nest_conformer', ...)
   print(model)
   ```

3. Check learning rate:
   ```bash
   # Try lower learning rate
   python scripts/train_zuco_full.py --lr 5e-5
   ```

4. Inspect training curves:
   - Validation loss not decreasing? → Increase model capacity or learning rate
   - Training loss not decreasing? → Check data loading, loss function
   - Overfitting? → Increase dropout, data augmentation

### Data Download Issues

If OSF download fails:

1. **Timeout**: Increase timeout in `src/data/zuco_dataset.py`:
   ```python
   urlretrieve(url, output_path, timeout=3600)  # 1 hour
   ```

2. **Alternative source**: Download from ZuCo official website:
   https://www.zuco.org/

3. **Resume download**: Use `wget` with resume capability:
   ```bash
   wget -c https://osf.io/q3zws/download -O data/raw/zuco.zip
   ```

## Computational Resources

### Estimated Training Costs

**On-Premise GPU**:
- NVIDIA A100 (40GB): 8-10 hours per model, ~$3-4 spot pricing
- NVIDIA RTX 3090 (24GB): 12-16 hours per model
- NVIDIA RTX 3080 (10GB): Requires reduced batch size, 20-24 hours

**Cloud Options**:
- **Google Colab Pro+**: $50/month, A100 access, sufficient for training
- **AWS EC2 p3.2xlarge**: ~$3.06/hour, 16GB V100
- **Lambda Labs**: ~$1.10/hour, A100 instances
- **Vast.ai**: ~$0.60/hour, RTX 3090 spot instances

**Budget Training**:
```bash
# Use smaller model variant
python scripts/train_zuco_full.py \
    --config configs/model_small.yaml \
    --output results/

# Estimated time on RTX 3080: 4-6 hours
```

## Citation

If you use these training scripts or reproduce results, please cite:

```bibtex
@article{nest2026,
  title={NEST: A Neural Sequence Transducer Framework for EEG-to-Text Decoding},
  author={[Authors]},
  journal={arXiv preprint},
  year={2026}
}
```

## Support

For training issues:
- GitHub Issues: https://github.com/wazder/NEST/issues
- Discussions: https://github.com/wazder/NEST/discussions
- Documentation: https://nest-bci.readthedocs.io/

## Advanced Usage

See also:
- [Subject Adaptation Guide](docs/SUBJECT_ADAPTATION.md)
- [Hyperparameter Tuning](docs/HYPERPARAMETER_TUNING.md)
- [Custom Dataset Tutorial](docs/CUSTOM_DATASET.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
