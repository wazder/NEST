#!/usr/bin/env python3
"""
Convert ZuCo 1.0 .mat files to the pickle format read by
src/data/zuco_pickle_dataset.py.

The output keeps the layout of the Wang & Ji (AAAI 2022) converter
(util/construct_dataset_mat_to_pickle_v1.py in MikeWangWZHL/EEG-To-Text),
so pickles from either tool load the same way. Differences:
  - every fixated word also keeps its eye-tracking measures
    (FFD, GD, TRT, GPT; nFixations was already kept), which the gaze-only
    input condition needs;
  - the .mat and output folders are arguments instead of fixed paths;
  - unfixated words (empty nFixations in the .mat) are detected by size,
    not by truth-testing an empty array.

Like the original, words without a fixation are skipped and sentences
without word data are stored as None.

Usage:
    python scripts/zuco_mat_to_pickle.py --task task1-SR \\
        --mat-dir ZuCo_Dataset/ZuCo/task1-SR/Matlab_files \\
        --out-dir ZuCo_Dataset/ZuCo/task1-SR/pickle
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.io as io

FREQ_BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
EEG_MEASURES = ["FFD", "TRT", "GD"]
GAZE_MEASURES = ["FFD", "GD", "TRT", "GPT"]


def is_fixated(word) -> bool:
    """True if the word received at least one fixation."""
    return np.size(word.nFixations) > 0 and word.nFixations > 0


def convert_word(word) -> Dict:
    """Word dict with band-power EEG per fixation measure and gaze measures."""
    word_obj = {"content": word.content, "nFixations": int(word.nFixations)}
    word_obj["word_level_EEG"] = {
        measure: {f"{measure}_{band}": getattr(word, f"{measure}_{band}") for band in FREQ_BANDS}
        for measure in EEG_MEASURES
    }
    for measure in GAZE_MEASURES:
        word_obj[measure] = float(getattr(word, measure))
    return word_obj


def convert_sentence(sent, task_name: str) -> Optional[Dict]:
    """Sentence dict in the Wang & Ji layout, or None if word data is missing."""
    if isinstance(sent.word, float):
        return None

    sent_obj = {
        "content": sent.content,
        "sentence_level_EEG": {f"mean_{band}": getattr(sent, f"mean_{band}") for band in FREQ_BANDS},
    }
    if task_name == "task1-SR":
        sent_obj["answer_EEG"] = {
            f"answer_mean_{band}": getattr(sent, f"answer_mean_{band}") for band in FREQ_BANDS
        }

    sent_obj["word"] = []
    tokens_has_fixation, tokens_with_mask, tokens_all = [], [], []
    for word in sent.word:
        tokens_all.append(word.content)
        if is_fixated(word):
            sent_obj["word"].append(convert_word(word))
            tokens_has_fixation.append(word.content)
            tokens_with_mask.append(word.content)
        else:
            tokens_with_mask.append("[MASK]")

    sent_obj["word_tokens_has_fixation"] = tokens_has_fixation
    sent_obj["word_tokens_with_mask"] = tokens_with_mask
    sent_obj["word_tokens_all"] = tokens_all
    return sent_obj


def subject_name(mat_path: Path) -> str:
    """'resultsZAB_SR.mat' -> 'ZAB'."""
    return mat_path.name.split("_")[0].replace("results", "").strip()


def convert_task(mat_dir: Path, task_name: str) -> Dict[str, List[Optional[Dict]]]:
    """Convert every .mat file in mat_dir to {subject: [sentence dict or None]}."""
    mat_files = sorted(mat_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {mat_dir}")

    dataset: Dict[str, List[Optional[Dict]]] = {}
    for mat_path in mat_files:
        print(f"Reading {mat_path.name}")
        sentences = io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["sentenceData"]
        dataset[subject_name(mat_path)] = [convert_sentence(s, task_name) for s in sentences]
    return dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Convert ZuCo 1.0 .mat files to pickle, keeping gaze measures")
    p.add_argument("--task", required=True, choices=["task1-SR", "task2-NR", "task3-TSR"])
    p.add_argument("--mat-dir", required=True, type=Path, help="Folder with results*.mat files")
    p.add_argument("--out-dir", required=True, type=Path, help="Output folder for <task>-dataset.pickle")
    args = p.parse_args()

    dataset = convert_task(args.mat_dir, args.task)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.task}-dataset.pickle"
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    for subject, sentences in dataset.items():
        present = [s for s in sentences if s is not None]
        n_words = sum(len(s["word"]) for s in present)
        print(f"{subject}: {len(sentences)} sentences ({len(sentences) - len(present)} missing), "
              f"{n_words} fixated words")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
