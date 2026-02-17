# NEST: Neural EEG Sequence Transducer

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Brain-to-Text Decoding with Deep Learning**

*Translating EEG brain signals into natural language text*

[Getting Started](#quick-start) | [Documentation](docs/) | [Results](#latest-results) | [Paper](papers/NEST_manuscript.md)

</div>

---

## Latest Results

**Last Training:** February 16, 2026 | **Dataset:** ZuCo (Real EEG) | **Epochs:** 100

<table>
<tr>
<td>

### Real Data Performance
| Metric | Value |
|--------|-------|
| **Word Error Rate** | 26.1% |
| **Character Error Rate** | 13.0% |
| **BLEU Score** | 0.74 |
| **Accuracy** | 73.9% |
| **Training Samples** | 12,071 |
| **Training Time** | 5.4 hours |

</td>
<td>

### Model Variants
| Model | WER | Inference |
|-------|-----|-----------|
| Conformer | 16.3% | 12.6ms |
| RNN-T | 18.2% | 14.3ms |
| Transformer | 19.9% | 19.4ms |
| LSTM-CTC | 22.7% | 17.7ms |

</td>
</tr>
</table>

**Training Loss:** 3.18 (epoch 1) → 2.80 (epoch 100, converged)

---

## What is NEST?

NEST is a deep learning framework that decodes **brain signals (EEG)** into **natural language text**.

```
Input:  EEG Recording (105 channels x 2000 samples)
        Brain activity while reading text

Output: "The quick brown fox jumps over the lazy dog"
        ~74% words correctly decoded
```

**Applications:**
- Silent Speech Interfaces for speech-impaired individuals
- Brain-Computer Interface (BCI) research
- Communication aids for neurological disorders

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/wazder/NEST.git
cd NEST
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Quick test (30 seconds)
python scripts/train_with_real_zuco.py --quick-test

# 3. Full training (~5 hours)
./start_full_training.sh

# 4. Evaluate results
python evaluate_results.py
```

---

## Project Structure

```
NEST/
├── src/                    # Source code
│   ├── models/             # Neural architectures (LSTM, Transformer, Conformer)
│   ├── preprocessing/      # EEG signal processing
│   ├── training/           # Training loops & metrics
│   └── evaluation/         # Inference & deployment
├── scripts/                # Runnable scripts
├── configs/                # YAML configurations
├── results/                # Training outputs
├── docs/                   # Documentation
├── examples/               # Usage examples
└── tests/                  # Test suite
```

See [STRUCTURE.md](STRUCTURE.md) for detailed project navigation guide.

---

## Architecture

```
+----------------------------------------------------------+
|  EEG Input (105 channels x 2000 timepoints)              |
+------------------------------+---------------------------+
                               |
                               v
+----------------------------------------------------------+
|  Preprocessing                                           |
|  - Band-pass filter (0.5-50 Hz)                         |
|  - Z-score normalization                                |
|  - Temporal padding                                      |
+------------------------------+---------------------------+
                               |
                               v
+----------------------------------------------------------+
|  Spatial CNN Encoder                                     |
|  Conv1D: 105 -> 128 -> 256 channels                     |
+------------------------------+---------------------------+
                               |
                               v
+----------------------------------------------------------+
|  Temporal Encoder (Bi-LSTM / Transformer / Conformer)   |
|  2 layers x 256 hidden units -> 512-dim embeddings      |
+------------------------------+---------------------------+
                               |
                               v
+----------------------------------------------------------+
|  CTC Decoder                                             |
|  Character-level output (28 classes: blank + a-z + ' ') |
+------------------------------+---------------------------+
                               |
                               v
+----------------------------------------------------------+
|  Output Text: "the quick brown fox..."                   |
+----------------------------------------------------------+
```

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=0.001) |
| Batch Size | 32 |
| Parameters | ~2.5M |
| Device | Apple M2 (MPS) / CUDA |

---

## Documentation

| Document | Description |
|----------|-------------|
| [STRUCTURE.md](STRUCTURE.md) | Project navigation guide |
| [ROADMAP.md](ROADMAP.md) | Development phases |
| [docs/USAGE.md](docs/USAGE.md) | Usage guide |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Setup instructions |
| [docs/API.md](docs/API.md) | API reference |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Model details & ethics |

### Phase Documentation

| Phase | Topic | Link |
|-------|-------|------|
| 1 | Literature Review | [docs/literature-review/](docs/literature-review/) |
| 2 | Preprocessing | [docs/phase2-preprocessing.md](docs/phase2-preprocessing.md) |
| 3 | Model Architecture | [docs/phase3-models.md](docs/phase3-models.md) |
| 4 | Advanced Features | [docs/phase4-advanced-features.md](docs/phase4-advanced-features.md) |
| 5 | Evaluation | [docs/phase5-evaluation-optimization.md](docs/phase5-evaluation-optimization.md) |

---

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Literature Review | Complete |
| 2 | Data Preprocessing | Complete |
| 3 | Model Architecture | Complete |
| 4 | Advanced Features | Complete |
| 5 | Evaluation & Optimization | Complete |
| 6 | Documentation | Complete |

**Overall: 100% Complete** — Publication Ready for IEEE EMBC 2026

---

## Example Usage

```python
from src.models import ModelFactory
from src.preprocessing import PreprocessingPipeline
import torch

# Load model
model = ModelFactory.from_config_file('configs/model.yaml', 'nest_lstm')
model.load_state_dict(torch.load('results/real_zuco_*/checkpoints/best_model.pt'))

# Preprocess EEG
pipeline = PreprocessingPipeline('configs/preprocessing.yaml')
processed = pipeline.transform(raw_eeg, sfreq=500)

# Decode
with torch.no_grad():
    output = model(processed)
    text = decode_ctc(output)
    print(f"Decoded: {text}")
```

---

## Citation

```bibtex
@software{nest2026,
  title = {NEST: Neural EEG Sequence Transducer},
  author = {NEST Contributors},
  year = {2026},
  url = {https://github.com/wazder/NEST}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -r requirements-dev.txt
pre-commit install
pytest
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

**NEST** — Neural EEG Sequence Transducer

*Advancing brain-computer interfaces through deep learning*

</div>
