#!/usr/bin/env python3
"""
Noise-Input Control Evaluation for NEST v2

Implements the noise-input control described in Jo et al. 2024,
"Are EEG-to-Text Models Working?" (arXiv:2405.06459). The paper's core
argument: an EEG-to-text model's WER/CER on real EEG is not by itself
evidence that it decodes brain signal, because language-model priors and
dataset artifacts alone can produce low error rates. The control is to
re-run the exact same free-decoding evaluation with the EEG input replaced
by noise that matches the real input's shape and statistics, but carries
no information about the target sentence. If WER/CER on noise is close to
WER/CER on real EEG, the model is not meaningfully using the EEG signal.
Only the *difference* between the two conditions is evidence of decoding.

This script:
  1. Loads a trained NEST v2 checkpoint the same way
     scripts/train_nest_v2.py's ``--resume`` path does.
  2. Reconstructs the same subject-independent held-out test split used
     during training (``get_subject_split`` + ``load_datasets``).
  3. Evaluates the checkpoint on that real test set using the *same*
     free-decoding evaluation functions as training (``eval_ctc`` /
     ``eval_bart`` from scripts/train_nest_v2.py — imported, not
     duplicated).
  4. Builds a noise condition with identical batch shapes:
       - "gaussian": per-feature N(mean, std) noise, with mean/std
         estimated from the real held-out test set itself.
       - "shuffled": the real EEG sequences, permuted across sentences
         (a derangement — no sentence keeps its own EEG) so that EEG and
         label no longer correspond.
  5. Evaluates the same checkpoint on the noise condition with identical
     decoding settings, and reports WER/CER for both conditions plus the
     delta.

BLEU-1/ROUGE-1: only reported if a lightweight dependency (sacrebleu or
rouge_score) is importable AND the underlying hypotheses/references are
available. ``eval_ctc``/``eval_bart`` in scripts/train_nest_v2.py are
reused as-is (per project convention: do not duplicate evaluation logic)
and do not return hypotheses/references, so BLEU-1/ROUGE-1 are not
computed by this script even when those libraries are installed — WER/CER
are reported instead. See the module-level note in the code below.

Usage:
    python scripts/eval_noise_control.py \\
        --checkpoint results/nest_v2_ctc_.../best_model.pt \\
        --noise gaussian

    python scripts/eval_noise_control.py \\
        --checkpoint results/nest_v2_bart_.../best_model.pt \\
        --model bart --noise shuffled --num-beams 4
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.zuco_pickle_dataset import (  # noqa: E402
    get_subject_split,
    collate_word_sequence_ctc,
)
from src.models.nest_v2 import NEST_CTC_v2, NEST_BART_v2  # noqa: E402

import train_nest_v2 as tnv2  # noqa: E402  (reuse eval_ctc/eval_bart/load_datasets/etc.)

logger = logging.getLogger(__name__)

NOISE_CHOICES = ("gaussian", "shuffled")
DEFAULT_TASKS = ["task1-SR", "task2-NR", "task3-TSR"]


# --- Pure helper functions (unit-testable without ZuCo data or a checkpoint) ---


def estimate_feature_stats(eeg_arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-feature (per-dimension) mean and std over a list of
    (n_words_i, eeg_dim) arrays, concatenated along the word axis.

    Args:
        eeg_arrays: non-empty list of arrays, each shaped (n_words_i, eeg_dim),
            all sharing the same eeg_dim.

    Returns:
        (mean, std), each shaped (eeg_dim,), dtype float32.
    """
    if not eeg_arrays:
        raise ValueError("eeg_arrays must contain at least one array")
    stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in eeg_arrays], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def estimate_feature_stats_from_dataset(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-feature mean/std of the "eeg" field over every sample of
    a word-sequence EEG dataset (e.g. ZuCoWordSequenceDataset)."""
    eeg_arrays = [dataset[i]["eeg"] for i in range(len(dataset))]  # type: ignore[arg-type]
    return estimate_feature_stats(eeg_arrays)


def generate_gaussian_noise(
    n_words: int,
    mean: np.ndarray,
    std: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate (n_words, eeg_dim) Gaussian noise, drawn independently per
    feature dimension from N(mean[d], std[d]).

    Args:
        n_words: number of rows (words) to generate — the "matching shape"
            for one sample.
        mean: (eeg_dim,) per-feature mean.
        std: (eeg_dim,) per-feature standard deviation (must be >= 0).
        rng: seeded numpy random Generator, for reproducibility.

    Returns:
        (n_words, eeg_dim) float32 array.
    """
    if mean.shape != std.shape:
        raise ValueError("mean and std must have the same shape")
    eeg_dim = mean.shape[0]
    standard_normal = rng.standard_normal(size=(n_words, eeg_dim))
    noise = standard_normal * std[np.newaxis, :] + mean[np.newaxis, :]
    return noise.astype(np.float32)


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly random derangement (permutation with no fixed
    points) of range(n), via rejection sampling.

    Args:
        n: number of elements to permute; must be >= 2 (a derangement of a
            single element does not exist).
        rng: seeded numpy random Generator.

    Returns:
        (n,) int array — a permutation of range(n) with perm[i] != i for
        every i.
    """
    if n < 2:
        raise ValueError("derangement requires at least 2 elements")
    identity = np.arange(n)
    perm = rng.permutation(n)
    while np.any(perm == identity):
        perm = rng.permutation(n)
    return perm


DELTA_EXCLUDE_KEYS = ("n",)


def compute_delta(
    real_metrics: Dict[str, float],
    noise_metrics: Dict[str, float],
    exclude: Tuple[str, ...] = DELTA_EXCLUDE_KEYS,
) -> Dict[str, float]:
    """Compute noise-minus-real differences for every shared numeric metric.

    Only the delta is evidence of decoding: comparable WER/CER between the
    real and noise conditions means the real-EEG score alone was not
    evidence the model is decoding EEG signal.

    Args:
        real_metrics: metrics dict from the real-EEG condition.
        noise_metrics: metrics dict from the noise condition.
        exclude: keys to skip (e.g. sample counts, which are not "metrics").

    Returns:
        Dict mapping metric name -> (noise_value - real_value), for every
        key present as a numeric value in both inputs and not excluded.
    """
    delta: Dict[str, float] = {}
    for key, real_value in real_metrics.items():
        if key in exclude or key not in noise_metrics:
            continue
        noise_value = noise_metrics[key]
        if isinstance(real_value, bool) or isinstance(noise_value, bool):
            continue
        if isinstance(real_value, (int, float)) and isinstance(noise_value, (int, float)):
            delta[key] = float(noise_value) - float(real_value)
    return delta


# --- Noise-injecting dataset wrappers (reuse the real collate/eval path unchanged) ---


class GaussianNoiseEEGDataset(Dataset):
    """Wraps a word-sequence EEG dataset, replacing each sample's EEG with
    per-feature Gaussian noise of the same (n_words, eeg_dim) shape. Text
    labels and sequence lengths are left untouched — only the EEG content
    is replaced, matching the real batches' shapes exactly.
    """

    def __init__(self, base_dataset: Dataset, mean: np.ndarray, std: np.ndarray, seed: int):
        self.base_dataset = base_dataset
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.base_dataset[idx])  # type: ignore[index]
        n_words = sample["eeg"].shape[0]
        rng = np.random.default_rng(self.seed + idx)
        sample["eeg"] = generate_gaussian_noise(n_words, self.mean, self.std, rng)
        return sample


class ShuffledEEGDataset(Dataset):
    """Wraps a word-sequence EEG dataset, replacing each sample's EEG
    sequence with another sample's real EEG sequence, per a fixed
    derangement permutation. Text labels stay at their original index, so
    EEG and label no longer correspond for any sample.
    """

    def __init__(self, base_dataset: Dataset, permutation: np.ndarray):
        n = len(base_dataset)  # type: ignore[arg-type]
        if permutation.shape[0] != n:
            raise ValueError("permutation length must match dataset length")
        self.base_dataset = base_dataset
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.base_dataset[idx])  # type: ignore[index]
        source = self.base_dataset[int(self.permutation[idx])]  # type: ignore[index]
        sample["eeg"] = source["eeg"]
        sample["eeg_len"] = source["eeg_len"]
        return sample


# --- Model construction from checkpoint ---


def build_model_from_checkpoint(
    checkpoint: Dict, model_type: str, device: torch.device
) -> nn.Module:
    """Build a NEST v2 model from a checkpoint's stored config and load its
    weights, the same way scripts/train_nest_v2.py's ``--resume`` path
    constructs+loads a model (architecture args -> state_dict)."""
    config = checkpoint.get("config", {})
    d_model = config.get("d_model", 512)
    num_layers = config.get("num_layers", 6)
    nhead = config.get("nhead", 8)
    gaze_dim = config.get("gaze_dim")  # set if the model was trained with gaze input

    if model_type == "ctc":
        model: nn.Module = NEST_CTC_v2(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            vocab_size=tnv2.VOCAB_SIZE,
            gaze_dim=gaze_dim,
            # absent in checkpoints from before output_frames: one frame per word step
            output_frames=config.get("output_frames"),
        )
    elif model_type == "bart":
        model = NEST_BART_v2(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            gaze_dim=gaze_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model"])
    return model.to(device)


# --- CLI ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Noise-input control evaluation for NEST v2 "
            "(Jo et al. 2024, arXiv:2405.06459)"
        )
    )
    p.add_argument("--checkpoint", required=True, help="Path to a trained NEST v2 checkpoint (.pt)")
    p.add_argument("--data-dir", default="ZuCo_Dataset/ZuCo", help="Path to ZuCo dataset")
    p.add_argument(
        "--model",
        choices=["ctc", "bart"],
        default=None,
        help="Model type; inferred from the checkpoint's stored config if omitted",
    )
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="ZuCo tasks to include")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help=(
            "Beam count recorded for reporting/reproducibility. NOTE: "
            "scripts/train_nest_v2.py's eval_bart() (reused here unmodified) "
            "hardcodes num_beams=4 internally and performs greedy decoding "
            "regardless of this value; both the real and noise conditions "
            "use that same fixed decoding path, so the real-vs-noise "
            "comparison is unaffected, but this flag does not currently "
            "change decoding behavior for values other than 4."
        ),
    )
    p.add_argument(
        "--noise",
        choices=NOISE_CHOICES,
        default="gaussian",
        help=(
            "gaussian: N(mean, std) per feature dimension, estimated from "
            "the real held-out test set. shuffled: real EEG sequences "
            "permuted across sentences so labels no longer match."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None, help="Output JSON path")
    p.add_argument(
        "--quick-test",
        action="store_true",
        help="Limit the held-out test set to 50 samples, for a fast smoke run",
    )
    return p.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def print_results_table(
    real_metrics: Dict[str, float],
    noise_metrics: Dict[str, float],
    delta: Dict[str, float],
    noise_type: str,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Noise-Input Control Evaluation — noise type: {noise_type}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<10}{'Real':>12}{'Noise':>12}{'Delta':>12}")
    for key in ("wer", "cer"):
        if key in real_metrics and key in noise_metrics:
            print(
                f"{key.upper():<10}"
                f"{real_metrics[key]:>12.4f}"
                f"{noise_metrics[key]:>12.4f}"
                f"{delta.get(key, float('nan')):>+12.4f}"
            )
    print(f"{'-' * 60}")
    print(f"n: real={real_metrics.get('n')}  noise={noise_metrics.get('n')}")
    print(f"{'=' * 60}")
    print("Only the delta is evidence of decoding.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        fail(f"checkpoint not found: {checkpoint_path}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        fail(f"data directory not found: {data_dir}")

    if args.noise not in NOISE_CHOICES:
        fail(f"unknown --noise value: {args.noise}")

    device = tnv2.get_device()
    torch.manual_seed(args.seed)
    logger.info(f"Device: {device}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    # weights_only=False matches scripts/train_nest_v2.py's own --resume loading
    # path: NEST v2 checkpoints bundle a plain-dict "config" (and, for full
    # checkpoints, optimizer state) alongside the tensors, which the safe
    # weights_only loader cannot unpickle. Only load checkpoints you trust.
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", {})

    model_type = args.model or ckpt_config.get("model")
    if model_type not in ("ctc", "bart"):
        fail(
            "could not determine model type from the checkpoint "
            "(no usable config.model field); pass --model {ctc,bart} explicitly."
        )

    logger.info(f"Model type: {model_type}")

    logger.info("Reconstructing subject-independent held-out test split ...")
    _, _, test_subjects = get_subject_split(str(data_dir), tasks=args.tasks)
    if not test_subjects:
        fail("no test subjects found in the dataset — check --data-dir and --tasks.")

    max_samples = 50 if args.quick_test else None
    try:
        test_ds = tnv2.load_datasets(
            str(data_dir), args.tasks, subjects=test_subjects, max_samples=max_samples
        )
    except RuntimeError as exc:
        fail(f"failed to load held-out test data: {exc}")
        return  # unreachable; keeps type checkers happy

    if len(test_ds) == 0:
        fail("held-out test split is empty — check --data-dir and --tasks.")

    logger.info(f"Held-out test subjects: {test_subjects}")
    logger.info(f"Held-out test samples: {len(test_ds)}")

    model = build_model_from_checkpoint(checkpoint, model_type, device)

    tokenizer = None
    if model_type == "bart":
        from transformers import BartTokenizer

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        collate_fn = tnv2.collate_bart_v2
    else:
        collate_fn = collate_word_sequence_ctc

    def evaluate(dataset: Dataset) -> Dict[str, float]:
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )
        if model_type == "ctc":
            metrics = tnv2.eval_ctc(model, loader, device, fp16=False)
        else:
            metrics = tnv2.eval_bart(model, loader, device, fp16=False, tokenizer=tokenizer)
        metrics.pop("hypotheses", None)  # keep this script's output format
        metrics.pop("references", None)
        metrics["n"] = len(dataset)  # type: ignore[arg-type]
        return metrics

    logger.info("Evaluating on REAL EEG (held-out test set) ...")
    real_metrics = evaluate(test_ds)

    logger.info(f"Building '{args.noise}' noise condition ...")
    if args.noise == "gaussian":
        mean, std = estimate_feature_stats_from_dataset(test_ds)
        noise_ds: Dataset = GaussianNoiseEEGDataset(test_ds, mean=mean, std=std, seed=args.seed)
    else:
        perm = derangement(len(test_ds), np.random.default_rng(args.seed))
        noise_ds = ShuffledEEGDataset(test_ds, perm)

    logger.info(f"Evaluating on NOISE EEG ({args.noise}) ...")
    noise_metrics = evaluate(noise_ds)

    delta = compute_delta(real_metrics, noise_metrics)

    print_results_table(real_metrics, noise_metrics, delta, args.noise)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"noise_control_{model_type}_{args.noise}_{timestamp}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "real": real_metrics,
        "noise": noise_metrics,
        "delta": delta,
        "config": {
            "checkpoint": str(checkpoint_path),
            "data_dir": str(data_dir),
            "model": model_type,
            "tasks": args.tasks,
            "test_subjects": test_subjects,
            "batch_size": args.batch_size,
            "num_beams": args.num_beams,
            "noise": args.noise,
            "seed": args.seed,
            "quick_test": args.quick_test,
            "device": str(device),
            "checkpoint_config": ckpt_config,
            "reference": "Jo et al. 2024, arXiv:2405.06459",
            "timestamp": timestamp,
        },
        "note": "Only the delta is evidence of decoding.",
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
