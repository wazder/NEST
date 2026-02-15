# NEST Examples

This directory contains practical examples demonstrating key features of the NEST framework.

## Quick Start

Run examples in order to learn the framework:

```bash
# 1. Basic training
python examples/01_basic_training.py

# 2. Advanced subject adaptation
python examples/02_subject_adaptation.py

# 3. Model optimization
python examples/03_optimization.py

# 4. Deployment setup
python examples/04_deployment.py
```

## Examples Overview

### 01_basic_training.py
**Complete end-to-end training workflow**

Demonstrates:
- Loading and preprocessing ZuCo dataset
- Creating a NEST model from configuration
- Training with early stopping
- Evaluation on test set
- Saving checkpoints

Perfect for: Getting started with NEST

Run time: ~2 hours (with GPU)

---

### 02_subject_adaptation.py
**Advanced techniques for cross-subject generalization**

Demonstrates:
- Subject-specific embeddings
- Domain Adversarial Neural Networks (DANN)
- Cross-subject evaluation
- Transfer learning strategies

Perfect for: Improving model generalization across subjects

Run time: ~1 hour (with GPU)

---

### 03_optimization.py
**Model optimization for production deployment**

Demonstrates:
- Magnitude-based pruning
- Structured pruning
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Combined optimization strategies
- Performance benchmarking

Perfect for: Deploying models with reduced size and latency

Run time: ~30 minutes

---

### 04_deployment.py
**Production deployment setup**

Demonstrates:
- TorchScript export
- ONNX export
- FP16 optimization
- REST API creation (FastAPI)
- Model serving

Perfect for: Deploying NEST in production

Run time: ~5 minutes

---

## Prerequisites

Before running examples:

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Download and prepare ZuCo dataset**
```bash
# Create directories
mkdir -p data/raw data/processed

# Download ZuCo dataset from https://osf.io/q3zws/
# Extract to data/raw/zuco/

# Preprocess dataset
python -m src.preprocessing.pipeline \
    --config configs/preprocessing.yaml \
    --input data/raw/zuco/ \
    --output data/processed/zuco/
```

3. **Verify setup**
```bash
python test_phase2.py
```

## Configuration Files

Examples use configuration files in `configs/`:

- `configs/model.yaml` - Model architecture settings
- `configs/preprocessing.yaml` - Preprocessing parameters

Modify these files to experiment with different settings.

## Expected Outputs

### 01_basic_training.py

**Output:**
```
Using device: cuda
Vocabulary size: 30
Train samples: 8532
Val samples: 1828
Test samples: 1828

Creating model...
Model parameters: 2,456,832

Training...
Epoch 1/50 - Train Loss: 3.5421, Val Loss: 3.1234
Epoch 2/50 - Train Loss: 2.9876, Val Loss: 2.7543
...
Best validation WER: 15.23%

Test Results:
Test WER: 16.45%
Test CER: 8.32%
```

**Files created:**
- `checkpoints/best_model.pt` - Best model checkpoint
- `checkpoints/final_model.pt` - Final model with history

---

### 02_subject_adaptation.py

**Output:**
```
Subject Adaptation Training
Training with 12 subjects
Epoch 1/10 - Loss: 2.4532
...

Domain Adversarial Training (DANN)
Training DANN with 12 domains
Epoch 1/10 - Class Loss: 2.3421, Domain Loss: 0.6923, Alpha: 0.000
...
```

**Files created:**
- `checkpoints/subject_adapted.pt` - Subject-adapted model
- `checkpoints/dann_model.pt` - DANN model

---

### 03_optimization.py

**Output:**
```
OPTIMIZATION SUMMARY
-----------------------------------------------------------------------
Method                         Size (MB)       Inference (ms)      Speedup   
-----------------------------------------------------------------------
Base Model                     18.50           15.23               1.00x
Magnitude Pruned (50%)         18.50           10.45               1.46x
Structured Pruned (30%)        18.50           11.23               1.36x
Dynamic Quantized              4.65            13.21               -
Static Quantized               4.65            11.89               -
Pruned + Quantized             4.65            8.92                -
```

**Insights:**
- Pruning improves speed with minimal accuracy loss
- Quantization reduces model size by ~75%
- Combined approach achieves best size/speed tradeoff

---

### 04_deployment.py

**Output:**
```
Model Export for Deployment

1. Exporting to TorchScript...
   Saved to: deployment/nest_model.pt

2. Exporting to ONNX...
   Saved to: deployment/nest_model.onnx

3. Exporting FP16 optimized model...
   Saved to: deployment/nest_model_fp16.pt
```

**Files created:**
- `deployment/nest_model.pt` - TorchScript model
- `deployment/nest_model.onnx` - ONNX model
- `deployment/nest_model_fp16.pt` - FP16 model
- `deployment/tokenizer.json` - Tokenizer config
- `deployment/metadata.json` - Model metadata
- `deployment/api.py` - REST API server
- `deployment/README.md` - Deployment guide

---

## Advanced Usage

### Custom Model Architecture

Create your own NEST variant:

```python
from src.models.nest import NestBase
import torch.nn as nn

class CustomNEST(NestBase):
    def __init__(self, n_channels, vocab_size):
        super().__init__()
        # Define your architecture
        self.encoder = ...
        self.decoder = ...
    
    def forward(self, eeg, text=None):
        # Implement forward pass
        ...

# Use in training
model = CustomNEST(n_channels=32, vocab_size=5000)
```

### Custom Training Loop

For more control over training:

```python
from src.training import Trainer

class CustomTrainer(Trainer):
    def training_step(self, batch):
        # Custom training logic
        ...
    
    def validation_step(self, batch):
        # Custom validation logic
        ...

trainer = CustomTrainer(model=model, device=device)
trainer.train(train_loader, val_loader, epochs=50)
```

### Hyperparameter Tuning

Use Weights & Biases for experiment tracking:

```python
import wandb

# Initialize W&B
wandb.init(project="nest-eeg", config={
    "learning_rate": 1e-4,
    "batch_size": 16,
    "model": "nest_conformer"
})

# Train with logging
trainer = Trainer(model=model, logger=wandb)
history = trainer.train(train_loader, val_loader, epochs=100)
```

## Common Issues

### Out of Memory

Reduce batch size in configuration:

```python
# Use smaller batches
train_loader = DataLoader(dataset, batch_size=8)  # Instead of 16
```

### Slow Training

Enable mixed precision training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(eeg)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Poor Convergence

Try different learning rates or schedulers:

```python
# Warmup + cosine decay
from src.training import get_scheduler

scheduler = get_scheduler(
    optimizer,
    'cosine_with_warmup',
    warmup_steps=1000,
    T_max=10000
)
```

## Performance Tips

1. **Use CUDA if available**: Training is 10-20x faster on GPU
2. **Enable DataLoader workers**: Use `num_workers=4` or more
3. **Pin memory**: Set `pin_memory=True` in DataLoader
4. **Gradient accumulation**: Simulate larger batches:

```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

5. **Mixed precision**: Use torch.amp for faster training

## Further Reading

- [USAGE.md](../docs/USAGE.md) - Comprehensive usage guide
- [API.md](../docs/API.md) - Complete API reference
- [Phase 3 Documentation](../docs/phase3-models.md) - Model architectures
- [Phase 5 Documentation](../docs/phase5-evaluation-optimization.md) - Optimization techniques

## Contributing

Have a useful example? Submit a pull request!

Example template:
```python
#!/usr/bin/env python3
"""
Example: <Title>

This example demonstrates:
1. Feature 1
2. Feature 2
3. Feature 3
"""

def main():
    # Your example code here
    pass

if __name__ == '__main__':
    main()
```

## License

Examples are provided under the same MIT license as the NEST project.
