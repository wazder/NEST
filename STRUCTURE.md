# 📂 NEST Project Structure

> **Quick Navigation Guide for AI Assistants & Developers**

## 🎯 Core Directories

```
NEST/
│
├── src/                 # 🧠 Core Source Code
│   ├── data/            # Dataset loaders (ZuCo)
│   ├── models/          # Neural architectures (LSTM, Transformer, Conformer)
│   ├── preprocessing/   # EEG signal processing pipeline
│   ├── training/        # Training loops, metrics, checkpoints
│   ├── evaluation/      # Inference, benchmarking, deployment
│   └── utils/           # Tokenizers, helpers
│
├── scripts/             # 🔧 Runnable Scripts
│   ├── train_with_real_zuco.py    # Main training script
│   ├── verify_zuco_data.py        # Data validation
│   └── generate_figures.py        # Publication figures
│
├── configs/             # ⚙️ Configuration Files
│   ├── model.yaml       # Model hyperparameters
│   └── preprocessing.yaml
│
├── results/             # 📊 Training Results & Checkpoints
│   └── real_zuco_*/     # Training runs with timestamps
│
├── docs/                # 📖 Documentation
│   ├── guides/          # How-to guides
│   ├── literature-review/
│   └── phase*.md        # Development phase docs
│
├── examples/            # 💡 Example Scripts
│   ├── 01_basic_training.py
│   ├── 02_subject_adaptation.py
│   ├── 03_optimization.py
│   └── 04_deployment.py
│
├── tests/               # 🧪 Test Suite
│   ├── unit/            # Unit tests
│   └── integration/     # End-to-end tests
│
├── data/                # 📁 Dataset Storage
│   ├── raw/zuco/        # Raw ZuCo .mat files
│   └── processed/       # Preprocessed data
│
└── papers/              # 📝 Research Paper Drafts
```

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview & results |
| `ROADMAP.md` | Development phases (1-6) |
| `requirements.txt` | Python dependencies |
| `start_full_training.sh` | One-click training script |
| `pyproject.toml` | Package configuration |

## 🧠 Source Code Overview (`src/`)

### `src/models/` - Neural Architectures
- `nest.py` - Main NEST model classes
- `spatial_cnn.py` - EEGNet, DeepConvNet for spatial features
- `temporal_encoder.py` - LSTM, Transformer, Conformer encoders
- `attention.py` - Cross-attention, self-attention mechanisms
- `decoder.py` - CTC decoder, RNN-T decoder
- `factory.py` - Model creation from YAML configs
- `adaptation.py` - Subject adaptation (DANN, CORAL)

### `src/preprocessing/` - Signal Processing
- `pipeline.py` - Complete preprocessing pipeline
- `filtering.py` - Band-pass filtering (0.5-50 Hz)
- `artifact_removal.py` - ICA-based artifact rejection
- `augmentation.py` - EEG data augmentation

### `src/training/` - Training Utilities
- `trainer.py` - Training loop manager
- `metrics.py` - WER, CER, BLEU calculations
- `checkpoint.py` - Model saving/loading

### `src/evaluation/` - Inference & Deployment
- `benchmark.py` - Comprehensive evaluation
- `beam_search.py` - Beam search decoder
- `pruning.py` - Model compression
- `quantization.py` - INT8/FP16 quantization
- `deployment.py` - ONNX/TorchScript export

## 📊 Results Directory

Each training run creates a timestamped folder:
```
results/real_zuco_20260216_031557/
├── checkpoints/           # Model weights
│   └── best_model.pt
├── results.json           # Training metrics
└── training_curve.pdf     # Loss visualization
```

## 🚀 Quick Start Points

| Goal | Start Here |
|------|------------|
| Train a model | `scripts/train_with_real_zuco.py` |
| Understand the model | `src/models/nest.py` |
| Preprocess EEG data | `src/preprocessing/pipeline.py` |
| Evaluate results | `evaluate_results.py` |
| Learn by example | `examples/01_basic_training.py` |
| Read the paper | `papers/NEST_manuscript.md` |

## 🤖 For AI Assistants

When modifying this project:
1. **Models**: Edit `src/models/` for architecture changes
2. **Training**: Edit `src/training/trainer.py` for training logic
3. **Configs**: Use `configs/model.yaml` for hyperparameters
4. **Scripts**: `scripts/` contains standalone executables
5. **Tests**: Add tests in `tests/unit/` or `tests/integration/`

**Code Style**: Black formatter, PEP 8, type hints required.
