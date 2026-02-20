"""
Training script for NEST-BART and NEST_ConformerCTC_Large.

Usage:
    python scripts/train_nest_bart.py --model ctc_large --epochs 200
    python scripts/train_nest_bart.py --model bart --epochs 100 --fp16
"""

import sys
import os
import argparse
import math
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.zuco_dataset import ZuCoTorchDataset
from src.models.nest_bart import NEST_ConformerCTC_Large

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Vocabulary (CTC mode) ---
BLANK = 0
CHARS = "abcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX: Dict[str, int] = {c: i + 1 for i, c in enumerate(CHARS)}
CHAR_TO_IDX[" "] = 27
PAD_IDX = 28
IDX_TO_CHAR: Dict[int, str] = {v: k for k, v in CHAR_TO_IDX.items()}
IDX_TO_CHAR[BLANK] = ""
IDX_TO_CHAR[PAD_IDX] = ""
VOCAB_SIZE = 29  # blank + 26 letters + space + pad


def text_to_indices(text: str) -> List[int]:
    text = text.lower()
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def ctc_greedy_decode(log_probs: torch.Tensor) -> List[str]:
    """
    log_probs: (batch, time, vocab_size)
    Returns list of decoded strings.
    """
    preds = log_probs.argmax(dim=-1)  # (batch, time)
    results = []
    for seq in preds:
        chars = []
        prev = -1
        for idx in seq.tolist():
            if idx != prev and idx != BLANK and idx != PAD_IDX:
                c = IDX_TO_CHAR.get(idx, "")
                if c:
                    chars.append(c)
            prev = idx
        results.append("".join(chars))
    return results


def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    """Compute Word Error Rate."""
    total_words = 0
    total_errors = 0
    for hyp, ref in zip(hypotheses, references):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_words += len(ref_words)
        total_errors += _levenshtein(hyp_words, ref_words)
    if total_words == 0:
        return 0.0
    return total_errors / total_words


def compute_cer(hypotheses: List[str], references: List[str]) -> float:
    """Compute Character Error Rate."""
    total_chars = 0
    total_errors = 0
    for hyp, ref in zip(hypotheses, references):
        total_chars += len(ref)
        total_errors += _levenshtein(list(hyp), list(ref))
    if total_chars == 0:
        return 0.0
    return total_errors / total_chars


def _levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


# --- Collate functions ---

def collate_ctc(batch: list) -> Optional[Dict]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    eeg_list, target_list, text_list = [], [], []
    for sample in batch:
        eeg = torch.tensor(sample["eeg"], dtype=torch.float32)
        # eeg from dataset: (channels, time) or (time, channels)
        if eeg.dim() == 2 and eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T  # ensure (channels, time)
        indices = text_to_indices(sample["text"])
        if len(indices) == 0:
            continue
        eeg_list.append(eeg)
        target_list.append(torch.tensor(indices, dtype=torch.long))
        text_list.append(sample["text"].lower())

    if not eeg_list:
        return None

    # Pad EEG along time dimension
    max_time = max(e.shape[-1] for e in eeg_list)
    n_channels = eeg_list[0].shape[0]
    padded_eeg = torch.zeros(len(eeg_list), n_channels, max_time)
    eeg_lengths = []
    for i, e in enumerate(eeg_list):
        t = e.shape[-1]
        padded_eeg[i, :, :t] = e
        eeg_lengths.append(t)

    # Pad targets
    max_tgt = max(t.shape[0] for t in target_list)
    padded_targets = torch.full((len(target_list), max_tgt), PAD_IDX, dtype=torch.long)
    target_lengths = []
    for i, t in enumerate(target_list):
        padded_targets[i, : t.shape[0]] = t
        target_lengths.append(t.shape[0])

    return {
        "eeg": padded_eeg,
        "eeg_lengths": torch.tensor(eeg_lengths, dtype=torch.long),
        "targets": padded_targets,
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": text_list,
    }


def collate_bart(batch: list) -> Optional[Dict]:
    try:
        from transformers import BartTokenizer
    except ImportError:
        raise RuntimeError("transformers required for BART mode")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    eeg_list, text_list = [], []
    for sample in batch:
        eeg = torch.tensor(sample["eeg"], dtype=torch.float32)
        if eeg.dim() == 2 and eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T
        eeg_list.append(eeg)
        text_list.append(sample["text"])

    if not eeg_list:
        return None

    max_time = max(e.shape[-1] for e in eeg_list)
    n_channels = eeg_list[0].shape[0]
    padded_eeg = torch.zeros(len(eeg_list), n_channels, max_time)
    for i, e in enumerate(eeg_list):
        t = e.shape[-1]
        padded_eeg[i, :, :t] = e

    encoded = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    labels = encoded["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "eeg": padded_eeg,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "texts": text_list,
    }


# --- Scheduler ---

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- Training ---

def train_epoch_ctc(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    grad_accum: int,
    fp16: bool,
) -> float:
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        eeg = batch["eeg"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        with torch.cuda.amp.autocast(enabled=fp16):
            log_probs = model(eeg)  # (batch, T, vocab)
            # CTCLoss expects (T, batch, vocab)
            log_probs_t = log_probs.permute(1, 0, 2)
            T = log_probs_t.shape[0]
            input_lengths = torch.full((eeg.shape[0],), T, dtype=torch.long, device=device)
            loss = ctc_loss_fn(log_probs_t, targets, input_lengths, target_lengths)
            loss = loss / grad_accum

        if fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_epoch_ctc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> Tuple[float, float, float]:
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)
    total_loss = 0.0
    n_batches = 0
    all_hyps, all_refs = [], []

    for batch in loader:
        if batch is None:
            continue

        eeg = batch["eeg"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"]
        texts = batch["texts"]

        with torch.cuda.amp.autocast(enabled=fp16):
            log_probs = model(eeg)

        log_probs_t = log_probs.permute(1, 0, 2)
        T = log_probs_t.shape[0]
        input_lengths = torch.full((eeg.shape[0],), T, dtype=torch.long, device=device)
        loss = ctc_loss_fn(log_probs_t, targets.to(device), input_lengths, target_lengths.to(device))

        total_loss += loss.item()
        n_batches += 1

        hyps = ctc_greedy_decode(log_probs.cpu())
        all_hyps.extend(hyps)
        all_refs.extend(texts)

    val_loss = total_loss / max(1, n_batches)
    wer = compute_wer(all_hyps, all_refs)
    cer = compute_cer(all_hyps, all_refs)
    return val_loss, wer, cer


def train_epoch_bart(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    grad_accum: int,
    fp16: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        eeg = batch["eeg"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=fp16):
            loss, _ = model(
                eeg_data=eeg,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = loss / grad_accum

        if fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_epoch_bart(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
    tokenizer,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_hyps, all_refs = [], []

    for batch in loader:
        if batch is None:
            continue

        eeg = batch["eeg"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        texts = batch["texts"]

        with torch.cuda.amp.autocast(enabled=fp16):
            loss, _ = model(
                eeg_data=eeg,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss += loss.item()
        n_batches += 1

        generated = model.generate(eeg, max_length=50, num_beams=5)
        hyps = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_hyps.extend(hyps)
        all_refs.extend(texts)

    val_loss = total_loss / max(1, n_batches)
    wer = compute_wer(all_hyps, all_refs)
    cer = compute_cer(all_hyps, all_refs)
    return val_loss, wer, cer


# --- Main ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NEST-BART or NEST_ConformerCTC_Large")
    p.add_argument("--model", choices=["bart", "ctc_large"], default="ctc_large")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data-dir", type=str, default="ZuCo_Dataset/ZuCo")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--tasks", nargs="+", default=["task1-SR", "task2-NR"])
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_datasets(data_dir: str, tasks: List[str], max_samples: Optional[int] = None):
    from torch.utils.data import ConcatDataset

    datasets = []
    data_path = Path(data_dir)
    for task in tasks:
        # Try both underscore and hyphen variants
        task_us = task.replace("-", "_")
        task_hy = task.replace("_", "-")
        for variant in [task_us, task_hy]:
            task_path = data_path / variant
            if task_path.exists():
                logger.info(f"Loading task: {variant}")
                ds = ZuCoTorchDataset(
                    root_dir=str(data_path),
                    task=variant,
                    max_samples=max_samples,
                )
                if len(ds) > 0:
                    datasets.append(ds)
                    break
        else:
            logger.warning(f"Task folder not found: {task} (tried {task_us}, {task_hy})")

    if not datasets:
        raise RuntimeError(f"No data loaded from {data_dir} for tasks {tasks}")

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def detect_n_channels(dataset) -> int:
    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        if sample is None:
            continue
        eeg = np.array(sample["eeg"])
        if eeg.ndim == 2:
            # Assume (channels, time) or (time, channels) — pick smaller dim
            return min(eeg.shape)
    return 105


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"results/nest_bart_{args.model}_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {output_dir}")

    if args.quick_test:
        args.epochs = 2
        max_samples = 20
        logger.info("Quick test mode: 2 epochs, 20 samples")
    else:
        max_samples = None

    # Load data
    dataset = load_datasets(args.data_dir, args.tasks, max_samples=max_samples)
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    logger.info(f"Dataset: {n_total} samples | train={n_train}, val={n_val}")

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    collate_fn = collate_ctc if args.model == "ctc_large" else collate_bart

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    n_channels = detect_n_channels(dataset)
    logger.info(f"Detected EEG channels: {n_channels}")

    # Build model
    if args.model == "ctc_large":
        model = NEST_ConformerCTC_Large(
            n_channels=n_channels,
            vocab_size=VOCAB_SIZE,
            d_model=768,
            num_layers=12,
            nhead=12,
        ).to(device)
        tokenizer = None
    else:
        from src.models.nest_bart import NEST_ConformerBART
        from transformers import BartTokenizer
        model = NEST_ConformerBART(n_channels=n_channels).to(device)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Optimizer — lower LR for BART decoder
    if args.model == "bart":
        bart_params = list(model.bart.parameters())
        bart_ids = {id(p) for p in bart_params}
        enc_params = [p for p in model.parameters() if id(p) not in bart_ids and p.requires_grad]
        dec_params = [p for p in bart_params if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": enc_params, "lr": args.lr},
            {"params": dec_params, "lr": args.lr * (1e-4 / 3e-4)},
        ], weight_decay=1e-2)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-2
        )

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    start_epoch = 0
    best_wer = float("inf")
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_wer = ckpt.get("best_wer", float("inf"))
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        if args.model == "ctc_large":
            train_loss = train_epoch_ctc(
                model, train_loader, optimizer, scheduler,
                scaler, device, args.grad_accum, args.fp16,
            )
            val_loss, val_wer, val_cer = eval_epoch_ctc(
                model, val_loader, device, args.fp16,
            )
        else:
            train_loss = train_epoch_bart(
                model, train_loader, optimizer, scheduler,
                scaler, device, args.grad_accum, args.fp16,
            )
            val_loss, val_wer, val_cer = eval_epoch_bart(
                model, val_loader, device, args.fp16, tokenizer,
            )

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Val WER: {val_wer:.3f} | "
            f"Val CER: {val_cer:.3f} | "
            f"LR: {current_lr:.3e} | "
            f"Time: {elapsed:.1f}s"
        )

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
                    "args": vars(args),
                },
                output_dir / "best_model.pt",
            )
            logger.info(f"  New best WER: {best_wer:.4f} — checkpoint saved")
        else:
            patience_counter += 1

        # Save latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_wer": best_wer,
                    "args": vars(args),
                },
                output_dir / f"checkpoint_epoch{epoch + 1}.pt",
            )

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch + 1} (patience={args.patience})")
            break

    logger.info(f"Training complete. Best val WER: {best_wer:.4f}")
    logger.info(f"Best model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
