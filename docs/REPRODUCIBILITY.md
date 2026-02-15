# Reproducibility Guide

This guide helps reproduce all results presented in the NEST framework.

## Quick Start

```bash
# Clone repository
git clone https://github.com/wazder/NEST.git
cd NEST

# Install dependencies
pip install -r requirements.txt

# Run full reproducibility pipeline
bash scripts/reproduce_all.sh
```

## System Requirements

### Hardware
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16 GB minimum, 32 GB recommended
- **GPU**: NVIDIA GPU with 8+ GB VRAM (Tesla V100, RTX 3080, or better)
- **Storage**: 50 GB free space

### Software
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **CUDA**: 11.0+ (for GPU support)
- **Docker**: Optional, for containerized reproduction

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# Create environment
conda create -n nest python=3.10
conda activate nest

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Option 3: Docker

```bash
docker build -t nest:latest .
docker run -it --gpus all -v $(pwd):/workspace nest:latest
```

## Data Preparation

### Download ZuCo Dataset

```bash
# Create directories
mkdir -p data/raw data/processed

# Download ZuCo 1.0 and 2.0
# Visit: https://osf.io/q3zws/
# Download and extract to data/raw/zuco/

# Verify dataset structure
ls data/raw/zuco/
# Expected: task1-SR/, task2-NR/, task3-TSR/, ...
```

### Preprocess Data

```bash
# Run preprocessing pipeline
python -m src.preprocessing.pipeline \
    --config configs/preprocessing.yaml \
    --input data/raw/zuco/ \
    --output data/processed/zuco/ \
    --seed 42

# Expected output:
# data/processed/zuco/train/ (6,300 samples)
# data/processed/zuco/val/ (1,350 samples)
# data/processed/zuco/test/ (1,350 samples)
```

## Reproducing Results

### Phase 2: Preprocessing

```bash
# Run preprocessing tests
python test_phase2.py

# Expected output:
# ✓ BandPassFilter test passed
# ✓ ArtifactRemover test passed
# ✓ ElectrodeSelector test passed
# ✓ DataAugmenter test passed
# ✓ Full pipeline test passed
```

### Phase 3: Model Training

```bash
# Train NEST-Attention (main model)
python examples/01_basic_training.py \
    --config configs/model.yaml \
    --model nest_attention \
    --output checkpoints/nest_attention/ \
    --seed 42

# Expected results:
# Validation WER: ~15.1%
# Test WER: ~16.5%
# Test CER: ~8.3%
# Training time: ~2 hours (GPU)
```

### Train All Model Variants

```bash
# Create results directory
mkdir -p results/

# NEST-RNN-T
python examples/01_basic_training.py \
    --model nest_rnn_t \
    --output results/nest_rnn_t \
    --seed 42

# NEST-Transformer-T
python examples/01_basic_training.py \
    --model nest_transformer_t \
    --output results/nest_transformer_t \
    --seed 42

# NEST-Conformer
python examples/01_basic_training.py \
    --model nest_conformer \
    --output results/nest_conformer \
    --seed 42

# NEST-CTC
python examples/01_basic_training.py \
    --model nest_ctc \
    --output results/nest_ctc \
    --seed 42
```

### Phase 4: Advanced Features

```bash
# Subject adaptation
python examples/02_subject_adaptation.py \
    --output results/subject_adaptation \
    --seed 42

# Expected improvements:
# Cross-subject WER: 16.5% → 14.2%
```

### Phase 5: Optimization

```bash
# Model pruning and quantization
python examples/03_optimization.py \
    --checkpoint checkpoints/nest_attention/best_model.pt \
    --output results/optimization

# Expected results:
# Magnitude Pruning (50%): 1.46x speedup, WER +1.2%
# Dynamic Quantization: 4x size reduction, WER +0.5%
```

## Benchmarking

### Comprehensive Evaluation

```bash
python -m src.evaluation.benchmark \
    --model_config configs/model.yaml \
    --checkpoint checkpoints/nest_attention/best_model.pt \
    --test_data data/processed/zuco/test \
    --beam_size 5 \
    --output results/benchmark.json

# Generate benchmark report
python scripts/generate_report.py \
    --results results/benchmark.json \
    --output results/report.html
```

### Expected Benchmark Results

| Model | WER (%) | CER (%) | BLEU | Latency (ms) | Parameters |
|-------|---------|---------|------|--------------|------------|
| NEST-Attention | 16.5 | 8.3 | 0.72 | 15 | 2.5M |
| NEST-RNN-T | 17.2 | 8.9 | 0.69 | 18 | 3.2M |
| NEST-Conformer | 15.8 | 7.8 | 0.75 | 22 | 4.1M |
| NEST-CTC | 18.5 | 9.5 | 0.66 | 12 | 2.1M |

## Random Seed Management

All experiments use fixed random seeds for reproducibility:

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

**Note**: Despite fixed seeds, minor variations (±0.5% WER) may occur due to:
- GPU non-determinism
- Different CUDA versions
- Hardware differences

## Hyperparameter Grid

All hyperparameters used in reported results:

```yaml
# configs/model.yaml
model:
  d_model: 256
  n_encoder_layers: 6
  n_decoder_layers: 6
  n_heads: 8
  d_ff: 1024
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 1e-4
  weight_decay: 0.01
  max_epochs: 100
  early_stopping_patience: 10
  clip_grad_norm: 1.0
  
  optimizer:
    type: adamw
    betas: [0.9, 0.999]
    eps: 1e-8
  
  scheduler:
    type: cosine
    T_max: 100
    eta_min: 1e-6
```

## Validation Protocol

### Subject-Independent Evaluation

```python
# Train/val/test split ensures no subject overlap
train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
val_subjects = [9, 10]
test_subjects = [11, 12]

# This ensures models generalize to unseen subjects
```

### Cross-Validation (Optional)

```bash
# 6-fold cross-validation for robust estimates
python scripts/cross_validation.py \
    --n_folds 6 \
    --output results/cv/ \
    --seed 42

# Aggregate results
python scripts/aggregate_cv.py \
    --input results/cv/ \
    --output results/cv_summary.json
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python examples/01_basic_training.py --batch_size 8

# Or enable gradient accumulation
python examples/01_basic_training.py --accumulation_steps 2
```

### Different Results

If you get different results (WER ±2%):

1. **Check random seed**: Ensure `--seed 42` is used
2. **Check PyTorch version**: Use PyTorch 2.0+
3. **Check CUDA version**: Results may vary slightly across CUDA versions
4. **Disable cuDNN benchmark**: Set `torch.backends.cudnn.benchmark = False`
5. **Check data split**: Verify train/val/test subjects match

### Slow Training

```bash
# Enable mixed precision
python examples/01_basic_training.py --fp16

# Increase batch size (if GPU memory allows)
python examples/01_basic_training.py --batch_size 32

# Use fewer workers if CPU-bound
python examples/01_basic_training.py --num_workers 2
```

## Computational Budget

### Training Time Estimates

| Model | GPU | Training Time | GPU Memory | Cost (AWS p3.2xlarge) |
|-------|-----|---------------|------------|----------------------|
| NEST-Attention | V100 | 2 hours | 6 GB | ~$6 |
| NEST-RNN-T | V100 | 2.5 hours | 7 GB | ~$8 |
| NEST-Conformer | V100 | 3 hours | 8 GB | ~$9 |
| NEST-CTC | V100 | 1.5 hours | 5 GB | ~$5 |

Total computational budget: ~$30 for all models

### Inference Time

- **CPU**: ~15 ms per sample (Intel i7)
- **GPU**: ~5 ms per sample (V100)
- **Optimized (quantized)**: ~8 ms per sample (CPU)

## Reproducibility Checklist

- [ ] Environment setup (Python 3.10, PyTorch 2.0+)
- [ ] Dataset downloaded and verified
- [ ] Random seed set (42)
- [ ] Preprocessing pipeline run
- [ ] Models trained with exact hyperparameters
- [ ] Evaluation run with beam search
- [ ] Results within expected range (±2% WER)
- [ ] Checkpoints saved
- [ ] Logs and metrics recorded

## Reporting Results

When reporting NEST results in your work:

1. **Include seed**: Always report random seed used
2. **Report variance**: Run multiple seeds (42, 43, 44) and report mean ± std
3. **Subject-level metrics**: Report per-subject WER for transparency
4. **Hardware details**: Mention GPU/CPU used
5. **Timing**: Report training and inference time
6. **Code version**: Mention git commit hash or release version

Example:
```
NEST-Attention achieves 16.5 ± 0.8% WER (mean ± std over 3 seeds) 
on ZuCo test set, evaluated in subject-independent setting.
Training: 2 hours on NVIDIA V100. Inference: 15ms/sample on CPU.
Code: github.com/wazder/NEST v1.0.0 (commit abc123).
```

## Artifact Storage

All trained models and results available at:

- **Checkpoints**: [Zenodo/FigShare link] (Coming soon)
- **Preprocessed Data**: [Zenodo link] (Coming soon)
- **Full Logs**: [GitHub Releases](https://github.com/wazder/NEST/releases)

## Containerized Reproduction

For maximum reproducibility, use Docker:

```bash
# Build container with exact dependencies
docker build -t nest-reproducible:v1.0 .

# Run complete pipeline
docker run -it --gpus all \
    -v /path/to/data:/data \
    -v /path/to/output:/output \
    nest-reproducible:v1.0 \
    bash scripts/reproduce_all.sh

# Results will be in /path/to/output/
```

## Contact

For reproducibility issues:
- Open GitHub Issue: https://github.com/wazder/NEST/issues
- Tag with `reproducibility` label
- Include: system info, error logs, attempted solutions

## References

- **Pineau, J. et al.** (2021). Improving Reproducibility in Machine Learning Research. JMLR.
- **Henderson, P. et al.** (2018). Deep Reinforcement Learning that Matters. AAAI.
- **Dodge, J. et al.** (2019). Show Your Work: Improved Reporting of Experimental Results. EMNLP.

---

Last Updated: February 2026
