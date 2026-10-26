"""Synthetic ZuCo pickles in the layout of scripts/zuco_mat_to_pickle.py.

No real data: random band powers and gaze measures that grow with word
length, enough to exercise the data, training and evaluation code paths.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

FREQ_BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
VOCABULARY = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "far", "away", "big", "red"]


def make_word(rng: np.random.Generator, content: str, with_gaze: bool = True) -> Dict:
    word = {"content": content, "nFixations": int(rng.integers(1, 4))}
    word["word_level_EEG"] = {
        measure: {f"{measure}_{band}": rng.standard_normal(105) for band in FREQ_BANDS}
        for measure in ["FFD", "TRT", "GD"]
    }
    if with_gaze:
        ffd = 80.0 + 20.0 * len(content) + rng.integers(0, 40)
        gd = ffd + rng.integers(0, 60)
        word.update({"FFD": ffd, "GD": gd, "TRT": gd + rng.integers(0, 200), "GPT": gd + rng.integers(0, 300)})
    return word


def make_sentence(rng: np.random.Generator, tokens: List[str], with_gaze: bool = True) -> Dict:
    return {
        "content": " ".join(tokens) + ".",
        "sentence_level_EEG": {f"mean_{band}": rng.standard_normal(105) for band in FREQ_BANDS},
        "word": [make_word(rng, t, with_gaze) for t in tokens],
        "word_tokens_has_fixation": tokens,
        "word_tokens_with_mask": tokens,
        "word_tokens_all": tokens,
    }


def write_synthetic_zuco(
    root: Path,
    task: str = "task1-SR",
    subjects: Sequence[str] = ("S01", "S02", "S03", "S04", "S05"),
    n_sentences: int = 6,
    seed: int = 0,
    with_gaze: bool = True,
) -> Path:
    """Write <root>/<task>/pickle/<task>-dataset.pickle and return its path.

    As in ZuCo, every subject reads the same n_sentences sentences (3 to 8
    words, own EEG and gaze per subject) plus one missing sentence (None).
    """
    rng = np.random.default_rng(seed)
    sentences = [[str(t) for t in rng.choice(VOCABULARY, size=int(rng.integers(3, 9)))] for _ in range(n_sentences)]
    dataset = {
        subject: [make_sentence(rng, tokens, with_gaze) for tokens in sentences] + [None]
        for subject in subjects
    }
    out_dir = root / task / "pickle"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}-dataset.pickle"
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)
    return out_path
