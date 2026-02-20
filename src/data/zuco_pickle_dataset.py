"""
ZuCo Pickle Dataset Loader

Loads pre-processed EEG features from ZuCo pickle files.
EEG data is in frequency-domain feature format:
  - 8 frequency bands: theta1, theta2, alpha1, alpha2, beta1, beta2, gamma1, gamma2
  - 105 channels each
  - Total: 840-dim feature vector per word or sentence

Reference:
  Wang et al. (2022) "Open Vocabulary EEG-to-Text Decoding and Zero-Shot
  Sentiment Classification", ACL 2022.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FREQ_BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
N_CHANNELS = 105
N_BANDS = len(FREQ_BANDS)
EEG_DIM = N_CHANNELS * N_BANDS  # 840


def _get_sentence_eeg(sentence_level_eeg: Dict) -> np.ndarray:
    """
    Concatenate all frequency bands into a single 840-dim vector.

    Args:
        sentence_level_eeg: dict with keys 'mean_t1', ..., 'mean_g2'

    Returns:
        ndarray of shape (840,)
    """
    bands = []
    for band in FREQ_BANDS:
        key = f"mean_{band}"
        if key in sentence_level_eeg:
            arr = np.array(sentence_level_eeg[key], dtype=np.float32)
            if arr.shape == (N_CHANNELS,):
                bands.append(arr)
            else:
                bands.append(np.zeros(N_CHANNELS, dtype=np.float32))
        else:
            bands.append(np.zeros(N_CHANNELS, dtype=np.float32))
    return np.concatenate(bands)  # (840,)


def _get_word_eeg(word_level_eeg: Dict, measure: str = "FFD") -> np.ndarray:
    """
    Get EEG features for a single word from a fixation measure.

    Args:
        word_level_eeg: dict with keys 'FFD', 'TRT', 'GD'
        measure: 'FFD' (First Fixation Duration), 'TRT' (Total Reading Time),
                 or 'GD' (Gaze Duration)

    Returns:
        ndarray of shape (840,) — concatenated frequency bands
    """
    if measure not in word_level_eeg:
        return np.zeros(EEG_DIM, dtype=np.float32)

    measure_dict = word_level_eeg[measure]
    bands = []
    for band in FREQ_BANDS:
        key = f"{measure}_{band}"
        if key in measure_dict:
            arr = np.array(measure_dict[key], dtype=np.float32)
            if arr.shape == (N_CHANNELS,):
                bands.append(arr)
            else:
                bands.append(np.zeros(N_CHANNELS, dtype=np.float32))
        else:
            bands.append(np.zeros(N_CHANNELS, dtype=np.float32))
    return np.concatenate(bands)  # (840,)


class ZuCoSentenceDataset(Dataset):
    """
    Sentence-level ZuCo dataset.

    Each sample is:
        eeg: (840,) concatenated mean EEG features across all 8 frequency bands
        text: str sentence

    This is suitable for sentence-classification or simple seq2seq tasks.
    """

    def __init__(
        self,
        data_dir: str,
        tasks: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        exclude_subjects: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Path to ZuCo_Dataset/ZuCo directory
            tasks: List of task names, e.g. ['task1-SR', 'task2-NR']
            subjects: If set, only include these subjects
            exclude_subjects: If set, exclude these subjects (useful for test split)
            max_samples: Limit total samples (for quick testing)
        """
        if tasks is None:
            tasks = ["task1-SR", "task2-NR", "task3-TSR"]

        self.samples: List[Dict] = []
        data_path = Path(data_dir)

        for task in tasks:
            pkl_path = data_path / task / "pickle" / f"{task}-dataset.pickle"
            if not pkl_path.exists():
                logger.warning(f"Pickle not found: {pkl_path}")
                continue

            logger.info(f"Loading {pkl_path}")
            with open(pkl_path, "rb") as f:
                task_data = pickle.load(f)

            if not isinstance(task_data, dict):
                logger.warning(f"Unexpected format in {pkl_path}")
                continue

            for subj_id, sentence_list in task_data.items():
                if subjects is not None and subj_id not in subjects:
                    continue
                if exclude_subjects is not None and subj_id in exclude_subjects:
                    continue

                for sent in sentence_list:
                    if sent is None or not isinstance(sent, dict):
                        continue
                    text = sent.get("content", "")
                    if not text or not isinstance(text, str):
                        continue

                    eeg_dict = sent.get("sentence_level_EEG", {})
                    eeg = _get_sentence_eeg(eeg_dict)

                    # Skip samples with all-zero EEG (missing data)
                    if np.all(eeg == 0):
                        continue

                    self.samples.append({
                        "eeg": eeg,
                        "text": text.strip(),
                        "subject": subj_id,
                        "task": task,
                    })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info(f"ZuCoSentenceDataset: {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class ZuCoWordSequenceDataset(Dataset):
    """
    Word-sequence-level ZuCo dataset (recommended for EEG-to-text).

    Each sample is:
        eeg: (max_words, 840) word-level EEG feature sequences
        eeg_len: int actual number of words
        text: str sentence

    Input is a sequence of word-level EEG features (one per fixated word).
    This directly follows Wang et al. 2022 (ACL) and subsequent SOTA papers.
    """

    def __init__(
        self,
        data_dir: str,
        tasks: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        exclude_subjects: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        max_words: int = 56,
        fixation_measure: str = "GD",
        min_word_count: int = 3,
    ):
        """
        Args:
            data_dir: Path to ZuCo_Dataset/ZuCo directory
            tasks: List of task names
            subjects: If set, only include these subjects
            exclude_subjects: If set, exclude these subjects
            max_samples: Limit total samples
            max_words: Maximum words per sentence (longer sentences truncated)
            fixation_measure: Which EEG measure to use ('FFD', 'TRT', 'GD')
            min_word_count: Minimum words required per sentence
        """
        if tasks is None:
            tasks = ["task1-SR", "task2-NR", "task3-TSR"]

        self.max_words = max_words
        self.fixation_measure = fixation_measure
        self.samples: List[Dict] = []
        data_path = Path(data_dir)

        for task in tasks:
            pkl_path = data_path / task / "pickle" / f"{task}-dataset.pickle"
            if not pkl_path.exists():
                logger.warning(f"Pickle not found: {pkl_path}")
                continue

            logger.info(f"Loading {pkl_path}")
            with open(pkl_path, "rb") as f:
                task_data = pickle.load(f)

            if not isinstance(task_data, dict):
                continue

            for subj_id, sentence_list in task_data.items():
                if subjects is not None and subj_id not in subjects:
                    continue
                if exclude_subjects is not None and subj_id in exclude_subjects:
                    continue

                for sent in sentence_list:
                    if sent is None or not isinstance(sent, dict):
                        continue
                    text = sent.get("content", "")
                    if not text or not isinstance(text, str):
                        continue

                    words = sent.get("word", [])
                    if len(words) < min_word_count:
                        continue

                    word_eegs = []
                    word_texts = []
                    for word in words[:max_words]:
                        weeg = _get_word_eeg(
                            word.get("word_level_EEG", {}),
                            measure=fixation_measure,
                        )
                        word_eegs.append(weeg)
                        word_texts.append(word.get("content", ""))

                    if not word_eegs:
                        continue

                    eeg_seq = np.stack(word_eegs, axis=0)  # (n_words, 840)

                    # Skip if all zeros (missing EEG for all words)
                    if np.all(eeg_seq == 0):
                        continue

                    self.samples.append({
                        "eeg": eeg_seq,
                        "eeg_len": len(word_eegs),
                        "text": text.strip(),
                        "words": word_texts,
                        "subject": subj_id,
                        "task": task,
                    })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info(
            f"ZuCoWordSequenceDataset: {len(self.samples)} samples, "
            f"measure={fixation_measure}, max_words={max_words}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def get_subject_split(
    data_dir: str,
    tasks: Optional[List[str]] = None,
    test_subjects: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Get train/val/test subject splits for subject-independent evaluation.

    Default split (11 subjects from task1-SR):
        Test:  ZMG, ZPH  (held-out, never seen during training)
        Val:   ZKH       (used for early stopping only)
        Train: remaining 8 subjects

    Returns:
        (train_subjects, val_subjects, test_subjects)
    """
    if tasks is None:
        tasks = ["task1-SR"]

    data_path = Path(data_dir)
    all_subjects = set()

    for task in tasks:
        pkl_path = data_path / task / "pickle" / f"{task}-dataset.pickle"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            task_data = pickle.load(f)
        if isinstance(task_data, dict):
            all_subjects.update(task_data.keys())

    all_subjects = sorted(all_subjects)

    if test_subjects is None:
        # Default: last 2 subjects as test
        test_subjects = all_subjects[-2:]

    val_subjects = [all_subjects[-3]] if len(all_subjects) > 3 else []
    train_subjects = [s for s in all_subjects if s not in test_subjects and s not in val_subjects]

    logger.info(f"Subject split — train: {train_subjects}, val: {val_subjects}, test: {test_subjects}")
    return train_subjects, val_subjects, test_subjects


def collate_word_sequence_ctc(
    batch: List[Dict],
    blank_id: int = 0,
    pad_id: int = 28,
) -> Optional[Dict]:
    """
    Collate function for ZuCoWordSequenceDataset with CTC training.

    Returns dict with:
        eeg: (batch, max_words, 840) padded EEG sequences
        eeg_lengths: (batch,) actual word counts
        targets: (batch, max_target_len) character indices
        target_lengths: (batch,) actual target lengths
        texts: list of reference strings
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    char_to_idx = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    char_to_idx[" "] = 27

    eeg_list = []
    eeg_lengths = []
    targets = []
    target_lengths = []
    texts = []

    for sample in batch:
        eeg = sample["eeg"].astype(np.float32) if isinstance(sample["eeg"], np.ndarray) else sample["eeg"]
        eeg_list.append(eeg)
        eeg_lengths.append(sample["eeg_len"])

        text = sample["text"].lower().strip()
        indices = [char_to_idx[c] for c in text if c in char_to_idx]
        if not indices:
            indices = [pad_id]
        targets.append(indices)
        target_lengths.append(len(indices))
        texts.append(text)

    # Pad EEG (batch, max_words, 840)
    max_words = max(e.shape[0] for e in eeg_list)
    eeg_dim = eeg_list[0].shape[1]
    padded_eeg = np.zeros((len(eeg_list), max_words, eeg_dim), dtype=np.float32)
    for i, e in enumerate(eeg_list):
        padded_eeg[i, : e.shape[0]] = e

    # Pad targets
    max_tgt = max(len(t) for t in targets)
    padded_targets = np.full((len(targets), max_tgt), pad_id, dtype=np.int64)
    for i, t in enumerate(targets):
        padded_targets[i, : len(t)] = t

    return {
        "eeg": torch.from_numpy(padded_eeg),
        "eeg_lengths": torch.tensor(eeg_lengths, dtype=torch.long),
        "targets": torch.from_numpy(padded_targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
    }


def collate_sentence_ctc(
    batch: List[Dict],
    blank_id: int = 0,
    pad_id: int = 28,
) -> Optional[Dict]:
    """
    Collate function for ZuCoSentenceDataset with CTC training.

    Input EEG shape: (840,) per sample.
    Batched as (batch, 840) — no sequence dimension.

    For CTC, we treat the 840-dim vector as a single-step sequence:
        (batch, 1, 840)
    This is a degenerate case; word-sequence is preferred.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    char_to_idx = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    char_to_idx[" "] = 27

    eeg_list = []
    targets = []
    target_lengths = []
    texts = []

    for sample in batch:
        eeg = np.array(sample["eeg"], dtype=np.float32)
        eeg_list.append(eeg)

        text = sample["text"].lower().strip()
        indices = [char_to_idx[c] for c in text if c in char_to_idx]
        if not indices:
            indices = [pad_id]
        targets.append(indices)
        target_lengths.append(len(indices))
        texts.append(text)

    eeg_array = np.stack(eeg_list, axis=0)  # (batch, 840)

    max_tgt = max(len(t) for t in targets)
    padded_targets = np.full((len(targets), max_tgt), pad_id, dtype=np.int64)
    for i, t in enumerate(targets):
        padded_targets[i, : len(t)] = t

    return {
        "eeg": torch.from_numpy(eeg_array),
        "targets": torch.from_numpy(padded_targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "ZuCo_Dataset/ZuCo"

    print("=== ZuCoWordSequenceDataset ===")
    ds = ZuCoWordSequenceDataset(
        data_dir=data_dir,
        tasks=["task1-SR"],
        max_samples=100,
    )
    print(f"Samples: {len(ds)}")
    sample = ds[0]
    print(f"EEG shape: {sample['eeg'].shape}")
    print(f"EEG length: {sample['eeg_len']}")
    print(f"Text: {repr(sample['text'])}")
    print(f"Subject: {sample['subject']}")

    print()
    print("=== Subject split ===")
    train_s, val_s, test_s = get_subject_split(data_dir)
    print(f"Train: {train_s}")
    print(f"Val:   {val_s}")
    print(f"Test:  {test_s}")

    print()
    print("=== Collate test ===")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_word_sequence_ctc)
    batch = next(iter(loader))
    print(f"Batched EEG: {batch['eeg'].shape}")
    print(f"EEG lengths: {batch['eeg_lengths']}")
    print(f"Targets: {batch['targets'].shape}")
    print(f"Texts: {batch['texts'][:2]}")
