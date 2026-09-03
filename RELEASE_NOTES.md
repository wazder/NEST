# NEST v1.0.0 Release Notes

## Overview

First public release of NEST (Neural EEG Sequence Transducer) — an open-source EEG-to-text decoding framework.

## Key Results

**No benchmark results are available yet.** NEST v2 has not been trained to completion and no WER, CER, or latency has been measured. See [docs/STATUS.md](docs/STATUS.md) for the current status and the evaluation protocol (free decoding with a noise-input control, following Jo et al. 2024) that will be used once training completes.

## Architecture (NEST v2)

- Input: Word-level EEG frequency features (840-dim: 105 channels × 8 frequency bands)
- EEG Encoder: Transformer (6 layers, d_model=768, 8 attention heads, pre-norm)
- Text Decoder: BART (facebook/bart-base) with EEG cross-attention
- Dataset: ZuCo (3 tasks, 11 subjects, ~12K sentence-subject pairs)
- Evaluation: Subject-independent (train on 8, test on 2 held-out subjects)

## What's Included

- Model weights: **not yet released** — no NEST-v2-BART checkpoint has completed training
- Complete training code (`scripts/train_nest_v2.py`)
- ZuCo pickle dataset loader (`src/data/zuco_pickle_dataset.py`)
- Google Colab training notebook (`notebooks/NEST_CloudTraining.ipynb`)
- Hugging Face upload script (`scripts/upload_to_huggingface.py`)

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Quick test (2 epochs)
python scripts/train_nest_v2.py --quick-test

# Full training (GPU recommended)
python scripts/train_nest_v2.py --model bart --epochs 200 --fp16 \
    --data-dir /path/to/ZuCo_Dataset/ZuCo

# Cloud training (Google Colab)
# Open: notebooks/NEST_CloudTraining.ipynb
```

## Dataset

ZuCo (Zurich Cognitive Language Processing Corpus). Request access at:
https://osf.io/q3zws/

## Citation

```bibtex
@article{nest2026,
  title={NEST: Neural EEG Sequence Transducer for Brain-to-Text Decoding},
  author={...},
  journal={arXiv},
  year={2026}
}
```

## License

MIT License. See LICENSE file.
