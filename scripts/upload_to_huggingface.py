#!/usr/bin/env python3
"""
Upload NEST model to Hugging Face Hub.

Usage:
    python scripts/upload_to_huggingface.py \
        --model-path results/best_model.pt \
        --repo-id username/NEST-EEG-to-Text \
        --token YOUR_HF_TOKEN
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch


MODEL_CARD_TEMPLATE = """\
---
language: en
tags:
- eeg
- brain-computer-interface
- eeg-to-text
- pytorch
- transformers
license: mit
datasets:
- ZuCo
metrics:
- wer
- cer
---

# NEST v2: Neural EEG Sequence Transducer

EEG-to-text decoding framework using word-level frequency features and BART decoder.
Trained on ZuCo dataset with subject-independent evaluation (WER: {wer}, CER: {cer}).

## Model Description

NEST v2 takes pre-processed EEG frequency features (105 channels x 8 frequency bands = 840
dimensions per word) and decodes them into natural language text using a transformer encoder
and fine-tuned BART decoder with cross-attention.

**Architecture:**
- Input: Word-level EEG frequency features (840-dim: 105 channels x theta/alpha/beta/gamma)
- EEG Projection: Linear(840 -> 768) + LayerNorm + GELU
- EEG Encoder: 6-layer Transformer (d_model=768, 8 heads, pre-norm)
- Text Decoder: facebook/bart-base with cross-attention to EEG encoder
- Evaluation: Subject-independent (train on 8, test on 2 held-out ZuCo subjects)

## Performance

| Dataset | Split | WER | CER |
|---------|-------|-----|-----|
| ZuCo (test subjects ZMG, ZPH) | subject-independent | {wer} | {cer} |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from src.models.nest_v2 import NEST_BART_v2

# Download checkpoint
ckpt_path = hf_hub_download(repo_id="{repo_id}", filename="model.pt")
checkpoint = torch.load(ckpt_path, map_location="cpu")

model = NEST_BART_v2()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# eeg shape: (batch, n_words, 840)
import torch
eeg = torch.randn(1, 17, 840)
text = model.generate(eeg)
print(text)
```

## Data

ZuCo (Zurich Cognitive Language Processing Corpus). Request access at:
https://osf.io/q3zws/

## Citation

```bibtex
@article{{nest2026,
  title={{NEST: Neural EEG Sequence Transducer for Brain-to-Text Decoding}},
  author={{...}},
  journal={{arXiv}},
  year={{2026}}
}}
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Upload NEST model to Hugging Face Hub")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. username/NEST-EEG-to-Text")
    parser.add_argument("--token", default=None, help="HF API token (or set HF_TOKEN env var)")
    parser.add_argument("--results-path", default=None, help="Path to evaluation_results.json")
    parser.add_argument("--vocab-path", default=None, help="Path to vocab/ directory")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    return parser.parse_args()


def load_metrics(results_path: str | None) -> dict:
    if results_path and Path(results_path).exists():
        with open(results_path) as f:
            data = json.load(f)
        wer = data.get("wer", data.get("test_wer", "N/A"))
        cer = data.get("cer", data.get("test_cer", "N/A"))
        if isinstance(wer, float):
            wer = f"{wer * 100:.1f}"
        if isinstance(cer, float):
            cer = f"{cer * 100:.1f}"
        return {"wer": wer, "cer": cer}
    return {"wer": "N/A", "cer": "N/A"}


def load_checkpoint(model_path: str) -> dict:
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    keys = list(checkpoint.keys())
    print(f"Checkpoint keys: {keys}")
    return checkpoint


def build_config(checkpoint: dict) -> dict:
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        if hasattr(cfg, "__dict__"):
            return vars(cfg)
        if isinstance(cfg, dict):
            return cfg
    return {
        "model_type": "nest_bart_v2",
        "architecture": "Transformer-BART",
        "eeg_dim": 840,
        "d_model": 768,
        "num_encoder_layers": 6,
        "nhead": 8,
        "bart_model": "facebook/bart-base",
        "note": "Config not stored in checkpoint; refer to src/models/nest_v2.py for full config.",
    }


def create_model_card(repo_id: str, metrics: dict, config: dict, output_dir: str):
    content = MODEL_CARD_TEMPLATE.format(
        wer=metrics["wer"],
        cer=metrics["cer"],
        repo_id=repo_id,
    )
    card_path = Path(output_dir) / "README.md"
    card_path.write_text(content)
    print(f"Model card written: {card_path}")
    return str(card_path)


def upload(args):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub>=0.19.0")

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Provide --token or set HF_TOKEN environment variable.")

    api = HfApi(token=token)

    print(f"Creating/verifying repo: {args.repo_id}")
    create_repo(
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )

    checkpoint = load_checkpoint(args.model_path)
    metrics = load_metrics(args.results_path)
    config = build_config(checkpoint)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model weights
        model_dest = os.path.join(tmpdir, "model.pt")
        shutil.copy2(args.model_path, model_dest)
        print(f"Copied model weights -> {model_dest}")

        # Save config
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Config written: {config_path}")

        # Model card
        create_model_card(args.repo_id, metrics, config, tmpdir)

        # Vocab/tokenizer
        if args.vocab_path and Path(args.vocab_path).exists():
            vocab_dest = os.path.join(tmpdir, "vocab")
            shutil.copytree(args.vocab_path, vocab_dest)
            print(f"Vocab directory copied: {vocab_dest}")
        else:
            vocab_src = Path("vocab")
            if vocab_src.exists():
                shutil.copytree(str(vocab_src), os.path.join(tmpdir, "vocab"))
                print("Vocab directory copied from ./vocab")

        # Upload entire directory
        print(f"Uploading to {args.repo_id} ...")
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
        )

    print()
    print(f"Upload complete: https://huggingface.co/{args.repo_id}")
    print(f"  WER: {metrics['wer']}  |  CER: {metrics['cer']}")


def main():
    args = parse_args()

    if not Path(args.model_path).exists():
        raise SystemExit(f"Model checkpoint not found: {args.model_path}")

    upload(args)


if __name__ == "__main__":
    main()
