#!/usr/bin/env python3
"""
NEST v2 Training Script

Trains NEST_CTC_v2 or NEST_BART_v2 on ZuCo EEG frequency features.

Architecture:
  - Input: Pre-processed EEG (105 channels × 8 frequency bands = 840-dim) per word
  - Encoder: Transformer (4-8 layers, d_model=512-768)
  - Decoder: CTC (fast) or BART (high quality)

Dataset:
  - ZuCo 1.0: 11 subjects × ~400 sentences per task
  - Subject-independent split: test on ZMG, ZPH (never seen during training)

Usage:
  # CTC model (fast, good baseline)
  python scripts/train_nest_v2.py --model ctc

  # BART model (best quality, requires more memory)
  python scripts/train_nest_v2.py --model bart --fp16

  # Quick test
  python scripts/train_nest_v2.py --quick-test

Cloud training (A100):
  python scripts/train_nest_v2.py --model ctc --epochs 200 --fp16 --batch-size 32
"""

import sys
import os
import math
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.zuco_pickle_dataset import (
    ZuCoWordSequenceDataset,
    get_subject_split,
    collate_word_sequence_ctc,
)
from src.models.nest_v2 import NEST_CTC_v2, build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Vocabulary ---
BLANK = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
CHAR_TO_IDX[" "] = 27
PAD_IDX = 28
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}
IDX_TO_CHAR[BLANK] = ""
IDX_TO_CHAR[PAD_IDX] = ""
VOCAB_SIZE = 29


# --- WER/CER computation ---

def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[len(b)]


def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    total_words = total_errors = 0
    for hyp, ref in zip(hypotheses, references):
        ref_w = ref.split()
        hyp_w = hyp.split()
        total_words += max(len(ref_w), 1)
        total_errors += _levenshtein(hyp_w, ref_w)
    return total_errors / max(total_words, 1)


def compute_cer(hypotheses: List[str], references: List[str]) -> float:
    total_chars = total_errors = 0
    for hyp, ref in zip(hypotheses, references):
        total_chars += max(len(ref), 1)
        total_errors += _levenshtein(list(hyp), list(ref))
    return total_errors / max(total_chars, 1)


def ctc_greedy_decode(log_probs: torch.Tensor) -> List[str]:
    """Greedy CTC decoding. log_probs: (batch, time, vocab)"""
    preds = log_probs.argmax(dim=-1)
    results = []
    for seq in preds.tolist():
        chars, prev = [], -1
        for idx in seq:
            if idx != prev and idx not in (BLANK, PAD_IDX):
                c = IDX_TO_CHAR.get(idx, "")
                if c:
                    chars.append(c)
            prev = idx
        results.append("".join(chars))
    return results


# --- Learning rate schedule ---

def cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int
):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- CTC training ---

def train_epoch_ctc(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    grad_accum: int,
    fp16: bool,
    epoch: int,
) -> float:
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)
    total_loss = n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        eeg = batch["eeg"].to(device)                      # (B, T, 840)
        eeg_lengths = batch["eeg_lengths"].to(device)      # (B,)
        targets = batch["targets"].to(device)              # (B, U)
        target_lengths = batch["target_lengths"].to(device) # (B,)

        with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
            log_probs = model(eeg, eeg_lengths)  # (B, T, vocab)
            log_probs_t = log_probs.permute(1, 0, 2)  # (T, B, vocab)
            T = log_probs_t.shape[0]
            input_lengths = torch.clamp(eeg_lengths, max=T)
            loss = ctc_loss_fn(log_probs_t, targets, input_lengths, target_lengths)
            loss = loss / grad_accum

        if fp16 and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if fp16 and device.type == "cuda":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if fp16 and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_ctc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> Dict[str, float]:
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)
    total_loss = n_batches = 0
    all_hyps, all_refs = [], []

    for batch in loader:
        if batch is None:
            continue

        eeg = batch["eeg"].to(device)
        eeg_lengths = batch["eeg_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        texts = batch["texts"]

        with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
            log_probs = model(eeg, eeg_lengths)

        log_probs_t = log_probs.permute(1, 0, 2)
        T = log_probs_t.shape[0]
        input_lengths = torch.clamp(eeg_lengths, max=T)
        loss = ctc_loss_fn(log_probs_t, targets, input_lengths, target_lengths)

        total_loss += loss.item()
        n_batches += 1

        hyps = ctc_greedy_decode(log_probs.cpu())
        all_hyps.extend(hyps)
        all_refs.extend(texts)

    val_loss = total_loss / max(n_batches, 1)
    wer = compute_wer(all_hyps, all_refs)
    cer = compute_cer(all_hyps, all_refs)

    return {"loss": val_loss, "wer": wer, "cer": cer}


# --- BART training ---

def train_epoch_bart(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    grad_accum: int,
    fp16: bool,
    tokenizer,
    epoch: int,
) -> float:
    model.train()
    total_loss = n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        eeg = batch["eeg"].to(device)
        eeg_lengths = batch["eeg_lengths"].to(device)
        texts = batch["texts"]

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        labels = encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
            loss, _ = model(
                eeg=eeg,
                eeg_lengths=eeg_lengths,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=labels,
            )
            loss = loss / grad_accum

        if fp16 and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if fp16 and device.type == "cuda":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if fp16 and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_bart(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
    tokenizer,
) -> Dict[str, float]:
    model.eval()
    total_loss = n_batches = 0
    all_hyps, all_refs = [], []

    for batch in loader:
        if batch is None:
            continue

        eeg = batch["eeg"].to(device)
        eeg_lengths = batch["eeg_lengths"].to(device)
        texts = batch["texts"]

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        labels = encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
            loss, _ = model(
                eeg=eeg,
                eeg_lengths=eeg_lengths,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=labels,
            )

        total_loss += loss.item()
        n_batches += 1

        generated = model.generate(eeg, eeg_lengths, max_length=64, num_beams=4)
        hyps = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_hyps.extend(hyps)
        all_refs.extend(texts)

    val_loss = total_loss / max(n_batches, 1)
    wer = compute_wer(all_hyps, all_refs)
    cer = compute_cer(all_hyps, all_refs)

    return {"loss": val_loss, "wer": wer, "cer": cer}


# --- Collate for BART (no char encoding) ---

def collate_bart_v2(batch: List[Dict]) -> Optional[Dict]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    eeg_list, eeg_lengths, texts = [], [], []
    for sample in batch:
        eeg = sample["eeg"].astype(np.float32) if isinstance(sample["eeg"], np.ndarray) else sample["eeg"]
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
        eeg_list.append(eeg)
        eeg_lengths.append(sample["eeg_len"])
        texts.append(sample["text"].lower().strip())

    max_words = max(e.shape[0] for e in eeg_list)
    eeg_dim = eeg_list[0].shape[1]
    padded_eeg = np.zeros((len(eeg_list), max_words, eeg_dim), dtype=np.float32)
    for i, e in enumerate(eeg_list):
        padded_eeg[i, : e.shape[0]] = e

    return {
        "eeg": torch.from_numpy(padded_eeg),
        "eeg_lengths": torch.tensor(eeg_lengths, dtype=torch.long),
        "texts": texts,
    }


# --- Dataset loading ---

def load_datasets(
    data_dir: str,
    tasks: List[str],
    subjects: Optional[List[str]] = None,
    exclude_subjects: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> ZuCoWordSequenceDataset:
    datasets = []
    for task in tasks:
        try:
            ds = ZuCoWordSequenceDataset(
                data_dir=data_dir,
                tasks=[task],
                subjects=subjects,
                exclude_subjects=exclude_subjects,
            )
            if len(ds) > 0:
                datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to load {task}: {e}")

    if not datasets:
        raise RuntimeError(f"No data loaded from {data_dir}")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    if max_samples is not None and hasattr(combined, "__len__"):
        class SubsetWrapper:
            def __init__(self, ds, n):
                self.ds = ds
                self.n = min(n, len(ds))
            def __len__(self): return self.n
            def __getitem__(self, i):
                if i >= self.n: raise IndexError
                return self.ds[i]
        combined = SubsetWrapper(combined, max_samples)

    return combined


# --- Argument parser ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NEST v2 on ZuCo")
    p.add_argument("--model", choices=["ctc", "bart"], default="ctc",
                   help="Model type: ctc (fast) or bart (best quality)")
    p.add_argument("--data-dir", default="ZuCo_Dataset/ZuCo",
                   help="Path to ZuCo dataset")
    p.add_argument("--tasks", nargs="+",
                   default=["task1-SR", "task2-NR", "task3-TSR"],
                   help="ZuCo tasks to include")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=512,
                   help="Transformer hidden size (512 or 768)")
    p.add_argument("--num-layers", type=int, default=6,
                   help="Number of Transformer layers")
    p.add_argument("--nhead", type=int, default=8,
                   help="Number of attention heads")
    p.add_argument("--fp16", action="store_true",
                   help="Use mixed precision (requires CUDA)")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (epochs)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--fixation", default="GD",
                   choices=["FFD", "TRT", "GD"],
                   help="EEG fixation measure to use")
    p.add_argument("--quick-test", action="store_true",
                   help="2 epochs, 50 samples, skip test eval")
    p.add_argument("--no-subject-split", action="store_true",
                   help="Use random 80/10/10 split instead of subject-independent")
    return p.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Main ---

def main():
    args = parse_args()
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"results/nest_v2_{args.model}_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"NEST v2 Training")
    logger.info(f"  Device: {device}")
    logger.info(f"  Model:  {args.model.upper()}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Tasks:  {args.tasks}")

    if args.quick_test:
        args.epochs = 2
        max_samples = 50
        logger.info("Quick test mode: 2 epochs, 50 samples")
    else:
        max_samples = None

    # Subject-independent split
    if args.no_subject_split:
        from torch.utils.data import random_split
        full_ds = load_datasets(
            args.data_dir, args.tasks, max_samples=max_samples
        )
        n = len(full_ds)
        n_test = max(1, int(0.1 * n))
        n_val = max(1, int(0.1 * n))
        n_train = n - n_test - n_val
        train_ds, val_ds, test_ds = random_split(
            full_ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_subjs, val_subjs, test_subjs = get_subject_split(
            args.data_dir, tasks=args.tasks
        )
        train_ds = load_datasets(
            args.data_dir, args.tasks, subjects=train_subjs, max_samples=max_samples
        )
        val_ds = load_datasets(
            args.data_dir, args.tasks, subjects=val_subjs
        )
        test_ds = load_datasets(
            args.data_dir, args.tasks, subjects=test_subjs
        )

    logger.info(
        f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    if args.model == "ctc":
        collate_fn = collate_word_sequence_ctc
    else:
        collate_fn = collate_bart_v2

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # Build model
    if args.model == "ctc":
        model = NEST_CTC_v2(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            dim_feedforward=args.d_model * 4,
            vocab_size=VOCAB_SIZE,
        ).to(device)
        tokenizer = None
    else:
        from src.models.nest_v2 import NEST_BART_v2
        from transformers import BartTokenizer
        model = NEST_BART_v2(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
        ).to(device)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-2,
        betas=(0.9, 0.98),
    )

    total_steps = (len(train_loader) // max(args.grad_accum, 1)) * args.epochs
    warmup_steps = max(100, int(0.1 * total_steps))
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    start_epoch = 0
    best_wer = float("inf")
    patience_counter = 0
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_wer = ckpt.get("best_wer", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}, best WER={best_wer:.4f}")

    # Save config
    config = {
        "model": args.model,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "tasks": args.tasks,
        "fixation": args.fixation,
        "n_params": n_params,
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val WER':>8}  {'Val CER':>8}  {'LR':>9}")
    print(f"{'='*70}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        if args.model == "ctc":
            train_loss = train_epoch_ctc(
                model, train_loader, optimizer, scheduler, scaler,
                device, args.grad_accum, args.fp16, epoch,
            )
            val_metrics = eval_ctc(model, val_loader, device, args.fp16)
        else:
            train_loss = train_epoch_bart(
                model, train_loader, optimizer, scheduler, scaler,
                device, args.grad_accum, args.fp16, tokenizer, epoch,
            )
            val_metrics = eval_bart(model, val_loader, device, args.fp16, tokenizer)

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        val_wer = val_metrics["wer"]
        val_cer = val_metrics["cer"]
        val_loss = val_metrics["loss"]

        print(
            f"{epoch+1:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}  "
            f"{val_wer:>8.4f}  {val_cer:>8.4f}  {current_lr:>9.2e}  ({elapsed:.0f}s)"
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_wer": round(val_wer, 5),
            "val_cer": round(val_cer, 5),
            "lr": current_lr,
        }
        history.append(row)

        is_best = val_wer < best_wer
        if is_best:
            best_wer = val_wer
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_wer": best_wer,
                    "config": config,
                },
                output_dir / "best_model.pt",
            )
            print(f"         *** New best WER: {best_wer:.4f} ***")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "best_wer": best_wer, "config": config},
                output_dir / f"checkpoint_epoch{epoch+1}.pt",
            )

        # Save history after each epoch
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= args.patience and not args.quick_test:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    print(f"{'='*70}")
    logger.info(f"Training complete. Best val WER: {best_wer:.4f}")

    # --- Final test evaluation ---
    if not args.quick_test:
        logger.info("Running final test evaluation on held-out subjects...")

        # Load best model
        best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])

        if args.model == "ctc":
            test_metrics = eval_ctc(model, test_loader, device, args.fp16)
        else:
            test_metrics = eval_bart(model, test_loader, device, args.fp16, tokenizer)

        print(f"\n{'='*70}")
        print("FINAL TEST RESULTS (held-out subjects, honest evaluation):")
        print(f"  WER: {test_metrics['wer']:.4f}  ({test_metrics['wer']*100:.1f}%)")
        print(f"  CER: {test_metrics['cer']:.4f}  ({test_metrics['cer']*100:.1f}%)")
        print(f"{'='*70}")

        results = {
            "model": args.model,
            "dataset": "ZuCo",
            "evaluation": "subject-independent",
            "best_val_wer": best_wer,
            "test_wer": test_metrics["wer"],
            "test_cer": test_metrics["cer"],
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "epochs_trained": len(history),
            "config": config,
            "note": "Real experimental results, honest evaluation",
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_dir / 'results.json'}")

    logger.info(f"All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
