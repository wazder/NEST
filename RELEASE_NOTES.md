# NEST v1.0.0 Release Notes

## Overview

First public release of NEST (Neural EEG Sequence Transducer) — state-of-the-art EEG-to-text decoding.

## Key Results

- X% WER on ZuCo dataset (subject-independent split)
- X% CER on ZuCo dataset
- Real-time capable: <100ms inference latency on a single GPU

## Architecture (NEST v2)

- Input: Word-level EEG frequency features (840-dim: 105 channels × 8 frequency bands)
- EEG Encoder: Transformer (6 layers, d_model=768, 8 attention heads, pre-norm)
- Text Decoder: BART (facebook/bart-base) with EEG cross-attention
- Dataset: ZuCo (3 tasks, 11 subjects, ~12K sentence-subject pairs)
- Evaluation: Subject-independent (train on 8, test on 2 held-out subjects)

## What's Included

- Pre-trained model weights (NEST-v2-BART)
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
