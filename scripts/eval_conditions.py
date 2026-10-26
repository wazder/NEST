#!/usr/bin/env python3
"""
Three-condition evaluation for NEST v2: real EEG vs gaze-only vs noise.

Loads one checkpoint and decodes the held-out test subjects freely (no
teacher forcing) under three inputs, with the same decoding for each:
  eeg:   the real word-level EEG features
  gaze:  word-level eye-tracking measures (nFixations, FFD, GD, TRT, GPT),
         through the model's gaze projection
  noise: Gaussian noise with the per-feature mean and std of the test EEG

The gaze condition needs a checkpoint trained with gaze input, e.g.
``train_nest_v2.py --input-mode eeg gaze``. All conditions use the same
sentences, sequence lengths and padding masks, so the comparison is paired.

For a checkpoint trained with --heldout-sentences, --sentences picks the test
set: 'heldout' (default; test subjects x sentences never seen in training) or
'seen' (test subjects x training sentences, a memorization diagnostic).

Reported:
  - corpus WER and CER per condition;
  - for each pair of conditions and each metric, a two-sided paired
    permutation test with one random sign flip per unique sentence. The
    held-out subjects read the same sentences, so subject-sentence pairs
    are not independent; errors are summed over all copies of a sentence;
  - Holm-corrected p values over all of these tests (pairs x metrics).

Decoding reuses eval_ctc/eval_bart from scripts/train_nest_v2.py and the
model is built by build_model_from_checkpoint from
scripts/eval_noise_control.py, so the numbers match the training-time
evaluation.

Usage:
    python scripts/eval_conditions.py \\
        --checkpoint results/nest_v2_bart_eeg_gaze/best_model.pt \\
        --data-dir ZuCo_Dataset/ZuCo --output results/conditions_bart.json
"""

import sys
import json
import argparse
import itertools
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.zuco_pickle_dataset import (  # noqa: E402
    INPUT_MODES,
    get_subject_split,
    collate_word_sequence_ctc,
)

import train_nest_v2 as tnv2  # noqa: E402
from eval_noise_control import build_model_from_checkpoint, fail  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TASKS = ["task1-SR", "task2-NR", "task3-TSR"]
METRICS = {"wer": "word", "cer": "char"}


# --- Pure helper functions (unit-testable without ZuCo data or a checkpoint) ---


def sentence_errors(
    hypotheses: Sequence[str], references: Sequence[str], unit: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sentence edit distances and reference lengths, counted exactly as
    compute_wer ('word') and compute_cer ('char') in train_nest_v2.py count
    them, so errors.sum() / lengths.sum() equals the corpus rate."""
    errors, lengths = [], []
    for hyp, ref in zip(hypotheses, references):
        if unit == "word":
            hyp_units, ref_units = hyp.split(), ref.split()
        elif unit == "char":
            hyp_units, ref_units = list(hyp), list(ref)
        else:
            raise ValueError(f"Unknown unit: {unit}")
        errors.append(tnv2._levenshtein(hyp_units, ref_units))
        lengths.append(max(len(ref_units), 1))
    return np.array(errors, dtype=np.int64), np.array(lengths, dtype=np.int64)


def aggregate_by_sentence(
    errors: np.ndarray, lengths: np.ndarray, references: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum errors and lengths over all copies of the same reference sentence
    (across subjects and tasks). Corpus rates do not change."""
    _, sentence_ids = np.unique(np.asarray(references), return_inverse=True)
    n_sentences = int(sentence_ids.max()) + 1
    summed_errors = np.zeros(n_sentences, dtype=np.int64)
    summed_lengths = np.zeros(n_sentences, dtype=np.int64)
    np.add.at(summed_errors, sentence_ids, errors)
    np.add.at(summed_lengths, sentence_ids, lengths)
    return summed_errors, summed_lengths


def dataset_tasks(dataset) -> set:
    """Tasks present in a dataset built by train_nest_v2.load_datasets."""
    dataset = getattr(dataset, "ds", dataset)  # --quick-test wrapper
    parts = getattr(dataset, "datasets", [dataset])  # ConcatDataset of tasks
    return {sample["task"] for part in parts for sample in part.samples}


def paired_permutation_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    chunk_size: int = 1000,
) -> float:
    """Two-sided paired permutation test for a difference in corpus error rate.

    Under the null hypothesis the two conditions are exchangeable within each
    sentence, so each per-sentence difference gets a random sign. The
    statistic is the summed difference: the corpus error-rate difference
    times the shared number of reference units.

    Returns:
        (1 + #{|T_perm| >= |T_obs|}) / (1 + n_permutations)
    """
    diff = np.asarray(errors_a, dtype=np.int64) - np.asarray(errors_b, dtype=np.int64)
    observed = abs(int(diff.sum()))
    exceed = 0
    for start in range(0, n_permutations, chunk_size):
        size = min(chunk_size, n_permutations - start)
        signs = rng.integers(0, 2, size=(size, diff.shape[0]), dtype=np.int64) * 2 - 1
        exceed += int((np.abs(signs @ diff) >= observed).sum())
    return (exceed + 1) / (n_permutations + 1)


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p values, in the input order."""
    m = len(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(np.argsort(p_values, kind="stable")):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running_max
    return adjusted


def compare_conditions(
    errors: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    n_permutations: int,
    seed: int,
) -> List[Dict]:
    """Paired permutation tests for every pair of conditions and metric.

    Args:
        errors: errors[condition][metric] = (errors, lengths) per unit of the
            sign flip (per unique sentence, see aggregate_by_sentence)
        n_permutations: random sign-flip permutations per test
        seed: seed for the permutations

    Returns:
        One dict per (pair, metric): rates, difference (a - b), p_value and
        p_holm (Holm over all returned tests).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in itertools.combinations(errors, 2):
        for metric in METRICS:
            errors_a, lengths = errors[a][metric]
            errors_b, _ = errors[b][metric]
            total = int(lengths.sum())
            rows.append({
                "a": a,
                "b": b,
                "metric": metric,
                "rate_a": int(errors_a.sum()) / total,
                "rate_b": int(errors_b.sum()) / total,
                "difference": int(errors_a.sum() - errors_b.sum()) / total,
                "p_value": paired_permutation_test(errors_a, errors_b, n_permutations, rng),
            })
    for row, p_holm in zip(rows, holm_correction([r["p_value"] for r in rows])):
        row["p_holm"] = p_holm
    return rows


# --- CLI ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEST v2 evaluation under EEG, gaze-only and noise input")
    p.add_argument("--checkpoint", required=True, help="Path to a trained NEST v2 checkpoint (.pt)")
    p.add_argument("--data-dir", default="ZuCo_Dataset/ZuCo", help="Path to ZuCo dataset")
    p.add_argument("--model", choices=["ctc", "bart"], default=None,
                   help="Model type; inferred from the checkpoint's stored config if omitted")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="ZuCo tasks to include (default: the checkpoint's training tasks)")
    p.add_argument("--conditions", nargs="+", choices=INPUT_MODES, default=list(INPUT_MODES),
                   help="Input conditions to decode (at least two for the tests)")
    p.add_argument("--sentences", choices=["heldout", "seen", "all"], default=None,
                   help="Test sentences: heldout (unseen in training), seen (training "
                        "sentences), all. Default: heldout if the checkpoint was trained "
                        "with --heldout-sentences, else all")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42, help="Seed for the permutation tests")
    p.add_argument("--output", type=str, default=None, help="Output JSON path")
    p.add_argument("--quick-test", action="store_true",
                   help="Limit the held-out test set to 50 samples, for a fast smoke run")
    return p.parse_args()


def print_results(conditions: Dict[str, Dict], comparisons: List[Dict]) -> None:
    print(f"\n{'=' * 72}")
    print("Free decoding on held-out subjects, same checkpoint and decoding")
    print(f"{'=' * 72}")
    print(f"{'Condition':<12}{'WER':>10}{'CER':>10}{'n':>8}{'sentences':>11}")
    for name, m in conditions.items():
        print(f"{name:<12}{m['wer']:>10.4f}{m['cer']:>10.4f}{m['n']:>8}{m['n_sentences']:>11}")
    print(f"{'-' * 72}")
    print(f"{'Pair':<16}{'Metric':<8}{'Diff (a-b)':>12}{'p':>10}{'p (Holm)':>10}")
    for r in comparisons:
        pair = f"{r['a']} - {r['b']}"
        print(f"{pair:<16}{r['metric'].upper():<8}{r['difference']:>+12.4f}"
              f"{r['p_value']:>10.4f}{r['p_holm']:>10.4f}")
    print(f"{'=' * 72}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    conditions = list(dict.fromkeys(args.conditions))
    if len(conditions) < 2:
        fail("need at least two conditions to compare")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        fail(f"checkpoint not found: {checkpoint_path}")
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        fail(f"data directory not found: {data_dir}")

    device = tnv2.get_device()
    torch.manual_seed(args.seed)
    logger.info(f"Device: {device}")

    # Load to CPU: build_model_from_checkpoint moves the model to the device
    # afterwards (loading MPS tensors into a CPU model fails on torch 2.14).
    # The plain-dict config and optimizer state load with weights_only=True.
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    ckpt_config = checkpoint.get("config", {})
    model_type = args.model or ckpt_config.get("model")
    if model_type not in ("ctc", "bart"):
        fail("could not determine model type from the checkpoint; pass --model {ctc,bart}.")
    if "gaze" in conditions and not ckpt_config.get("gaze_dim"):
        fail("the checkpoint has no gaze projection; train with --input-mode eeg gaze, "
             "or drop 'gaze' from --conditions.")
    if ckpt_config.get("no_subject_split"):
        fail("the checkpoint was trained with --no-subject-split; it has no held-out subjects.")
    tasks = args.tasks or ckpt_config.get("tasks") or DEFAULT_TASKS
    trained_heldout = bool(ckpt_config.get("heldout_sentences"))
    sentences = args.sentences or ("heldout" if trained_heldout else "all")
    if sentences != "all" and not trained_heldout:
        fail("--sentences heldout/seen needs a checkpoint trained with --heldout-sentences.")
    sentence_splits = {"heldout": ("test",), "seen": ("train",), "all": None}[sentences]

    _, _, test_subjects = get_subject_split(str(data_dir), tasks=tasks)
    if not test_subjects:
        fail("no test subjects found in the dataset; check --data-dir and --tasks.")
    trained_split = ckpt_config.get("subject_split")
    if trained_split and trained_split["test"] != test_subjects:
        fail(f"test subjects {test_subjects} differ from the checkpoint's {trained_split['test']}.")
    try:
        test_ds = tnv2.load_datasets(
            str(data_dir), tasks, subjects=test_subjects,
            max_samples=50 if args.quick_test else None, input_modes=conditions,
            sentence_splits=sentence_splits,
        )
    except RuntimeError as exc:
        fail(f"failed to load held-out test data: {exc}")
        return
    missing_tasks = set(tasks) - dataset_tasks(test_ds)
    if missing_tasks:
        fail(f"no held-out data for tasks {sorted(missing_tasks)}; check the pickles.")
    logger.info(f"Held-out test subjects: {test_subjects}, sentences: {sentences}, samples: {len(test_ds)}")

    model = build_model_from_checkpoint(checkpoint, model_type, device)
    tokenizer = None
    if model_type == "bart":
        from transformers import BartTokenizer

        tokenizer = (
            BartTokenizer.from_pretrained("facebook/bart-base")
        )
        collate_fn = tnv2.collate_bart_v2
    else:
        collate_fn = collate_word_sequence_ctc
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    decoded = {}
    for condition in conditions:
        logger.info(f"Decoding with {condition} input ...")
        if model_type == "ctc":
            decoded[condition] = tnv2.eval_ctc(model, loader, device, fp16=False, input_mode=condition)
        else:
            decoded[condition] = tnv2.eval_bart(
                model, loader, device, fp16=False, tokenizer=tokenizer, input_mode=condition
            )

    references = decoded[conditions[0]]["references"]
    if any(decoded[c]["references"] != references for c in conditions):
        fail("conditions decoded different sentences; the paired tests would be invalid.")

    errors = {
        c: {
            metric: aggregate_by_sentence(
                *sentence_errors(decoded[c]["hypotheses"], references, unit), references
            )
            for metric, unit in METRICS.items()
        }
        for c in conditions
    }
    n_sentences = len(set(references))
    results_by_condition = {
        c: {"wer": decoded[c]["wer"], "cer": decoded[c]["cer"], "n": len(references), "n_sentences": n_sentences}
        for c in conditions
    }
    comparisons = compare_conditions(errors, args.n_permutations, args.seed)
    print_results(results_by_condition, comparisons)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output) if args.output
        else Path("results") / f"conditions_{model_type}_{timestamp}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "conditions": results_by_condition,
        "comparisons": comparisons,
        "hypotheses": {c: decoded[c]["hypotheses"] for c in conditions},
        "references": references,
        "config": {
            "checkpoint": str(checkpoint_path),
            "data_dir": str(data_dir),
            "model": model_type,
            "tasks": tasks,
            "test_subjects": test_subjects,
            "sentences": sentences,
            "conditions": conditions,
            "batch_size": args.batch_size,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "quick_test": args.quick_test,
            "device": str(device),
            "checkpoint_config": ckpt_config,
            "decoding": "free (no teacher forcing): greedy for BART, greedy CTC",
            "test": "two-sided paired sign-flip permutation test, one sign per unique sentence",
            "correction": "Holm over all pairs and metrics",
            "timestamp": timestamp,
        },
    }
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
