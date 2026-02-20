# NEST: Neural EEG Sequence Transducer

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b.svg)](papers/NEST_manuscript.md)

**Open-Vocabulary EEG-to-Text Decoding with Deep Learning**

*Translating EEG brain signals into natural language text*

[Quick Start](#quick-start) | [Architecture](#architecture) | [Training](#training) | [Paper](papers/NEST_manuscript.md)

</div>

---

## Overview

NEST decodes **EEG brain signals** recorded during reading into **natural language text**.
It uses pre-processed frequency-domain EEG features from the ZuCo corpus and a
Transformer + BART architecture to achieve open-vocabulary decoding.

```
Input:  Word-level EEG (105 channels × 8 frequency bands = 840 features per word)
        Brain activity recorded while a subject reads each word

Output: "presents a good case while failing to provide a reason..."
        Full sentence decoded from EEG alone
```

**Applications:**
- Silent speech interfaces for communication-impaired individuals
- Brain-computer interface (BCI) research
- Cognitive neuroscience (reading comprehension, semantic processing)
- Assistive technology for ALS, locked-in syndrome

---

## Dataset

**ZuCo** (Zurich Cognitive Language Processing Corpus):
- 11-12 subjects, 3 reading tasks (SR, NR, TSR)
- ~12,884 sentence-subject pairs
- EEG: 105 channels, pre-processed into 8 frequency bands
- Feature format: 840-dim vector per fixated word (105 ch × 8 bands)

Download from [OSF](https://osf.io/q3zws/) or use the provided download script.

---

## Architecture

NEST v2 uses frequency-domain EEG features — not raw time-series — following
the approach of Wang et al. (2022, ACL):

```
Word-level EEG features: (batch, max_words, 840)
  105 channels × 8 frequency bands (θ1/θ2, α1/α2, β1/β2, γ1/γ2)
         |
         v
EEG Projection: Linear(840 → 768) + LayerNorm + GELU
         |
         v
Positional Encoding (sinusoidal)
         |
         v
Transformer Encoder
  6 layers, d_model=768, 8 heads, pre-norm, GELU FFN
         |
         v
BART Decoder (facebook/bart-base)
  Cross-attention to EEG encoder output
  First 4 layers frozen, last 2 + cross-attention fine-tuned
         |
         v
Text output (beam search, beam_size=4)
```

| Component | Config |
|-----------|--------|
| EEG Input dim | 840 (105ch × 8 bands) |
| d_model | 768 |
| Encoder layers | 6 |
| Attention heads | 8 |
| BART model | facebook/bart-base |
| Trainable params | ~30M |

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/wazder/NEST.git
cd NEST
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify dataset
python src/data/zuco_pickle_dataset.py ZuCo_Dataset/ZuCo

# 3. Quick sanity check (2 epochs, 50 samples)
python scripts/train_nest_v2.py --quick-test --model ctc

# 4. Full training — CTC baseline (fast, CPU/GPU)
python scripts/train_nest_v2.py --model ctc --epochs 200

# 5. Full training — BART (best quality, requires GPU)
python scripts/train_nest_v2.py --model bart --epochs 200 --fp16
```

---

## Training

### Local (GPU recommended)

```bash
python scripts/train_nest_v2.py \
    --model bart \
    --epochs 200 \
    --batch-size 16 \
    --fp16 \
    --d-model 768 \
    --num-layers 6 \
    --tasks task1-SR task2-NR task3-TSR
```

### Cloud (Google Colab / Lambda / Vast.ai)

See [notebooks/NEST_CloudTraining.ipynb](notebooks/NEST_CloudTraining.ipynb) for
a ready-to-run Colab notebook with Google Drive integration.

Recommended cloud config: A100 40GB, ~8-12 hours for 200 epochs.

### Subject-Independent Evaluation

By default, NEST trains on 8 subjects and evaluates on 2 held-out subjects
(ZMG, ZPH) that are never seen during training or validation.
This is the strictest and most realistic evaluation protocol.

---

## Evaluation

```bash
# Evaluate a trained model
python scripts/train_nest_v2.py \
    --resume results/nest_v2_bart_*/best_model.pt \
    --epochs 0  # eval only
```

Results are saved to `results/nest_v2_*/results.json`:
```json
{
  "model": "bart",
  "evaluation": "subject-independent",
  "test_wer": 0.XXX,
  "test_cer": 0.XXX,
  "note": "Real experimental results, honest evaluation"
}
```

---

## Upload to Hugging Face

```bash
python scripts/upload_to_huggingface.py \
    --model-path results/best_model.pt \
    --repo-id your-username/NEST-EEG-to-Text \
    --token $HF_TOKEN
```

---

## Project Structure

```
NEST/
├── src/
│   ├── data/
│   │   ├── zuco_pickle_dataset.py   # ZuCo dataset loader (pickle-based)
│   │   └── zuco_dataset.py          # Legacy MATLAB loader
│   ├── models/
│   │   ├── nest_v2.py               # NEST v2 (primary: CTC + BART)
│   │   ├── nest_bart.py             # NEST-Conformer-BART (raw EEG)
│   │   └── nest.py                  # Legacy NEST models
│   └── training/
│       └── trainer.py               # Training utilities
├── scripts/
│   ├── train_nest_v2.py             # Primary training script
│   ├── train_nest_bart.py           # BART training (raw EEG version)
│   └── upload_to_huggingface.py     # HF Hub upload
├── notebooks/
│   └── NEST_CloudTraining.ipynb     # Google Colab notebook
├── papers/
│   └── NEST_manuscript.md           # Research paper draft
├── configs/                         # YAML model configurations
├── results/                         # Training outputs
└── ZuCo_Dataset/                    # Dataset (not in git)
```

---

## Citation

```bibtex
@article{nest2026,
  title     = {NEST: Open-Vocabulary EEG-to-Text Decoding via Sequence Transduction},
  author    = {},
  journal   = {arXiv preprint},
  year      = {2026},
  url       = {https://github.com/wazder/NEST}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
