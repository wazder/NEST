# NEST Project Status

This document is the single source of truth for what NEST currently is, what has
actually been verified, and what has not. Where other documents in this
repository (status reports, phase summaries, the model card, the manuscript)
disagree with this file, this file is correct.

Last updated: September 2026.

## What exists

- **NEST v2 architecture** (`src/models/nest_v2.py`): a Transformer EEG encoder
  cross-attended by a BART (`facebook/bart-base`) decoder (`NEST_BART_v2`:
  184,424,592 total parameters, 64,697,232 trainable), alongside a CTC baseline
  (`NEST_CTC_v2`: 45,310,637 parameters). Both are implemented and instantiate
  correctly.
- **Training/evaluation code** (`scripts/train_nest_v2.py`): implements honest
  evaluation — `eval_bart()` decodes with `model.generate()` (free autoregressive
  decoding, no teacher forcing; `num_beams=4` is passed but the current
  implementation decodes greedily, see the comment in `nest_v2.py`), `eval_ctc()` performs greedy CTC decoding,
  and WER/CER are computed via word/character Levenshtein distance. This code
  has not yet been run to completion on NEST v2.
- **ZuCo integration** (`src/data/zuco_pickle_dataset.py`): loads ZuCo 1.0 (12
  subjects, 105-channel EEG, 1,107 sentences, 3 reading tasks — SR, NR, TSR),
  with a subject-independent split (8 train / 2 validation / 2 test subjects).
  Only reading paradigms exist in this data — there is no listening or imagery
  condition.
- **A completed training run** (February 16, 2026): a small, ad hoc
  `SimpleLSTMDecoder` (~2.78M parameters, defined inline in
  `scripts/train_with_real_zuco.py` — not the NEST v2 architecture above) was
  trained for 100 epochs on 12,071 ZuCo sentence-subject pairs, taking
  approximately 5.4 hours on an Apple M2 laptop. Training loss decreased over
  the run.

## What is verified

- The architectures above import and run (forward pass, parameter counts).
- The evaluation code (free autoregressive decoding, currently greedy / greedy CTC decoding, Levenshtein-based
  WER/CER) is real, implemented, and not teacher-forced.
- The one completed training run's loss curve decreased over 100 epochs.

## What is NOT verified

- **No word error rate (WER) exists for any NEST model.** No decoding was ever
  run and compared against reference text for the completed training run, and
  NEST v2 (`NEST_CTC_v2` / `NEST_BART_v2`) has not been trained to completion —
  it needs a GPU that has not yet been allocated to this project.
- No CER, BLEU, ROUGE, or perplexity number has been measured for any NEST
  model.
- No latency, throughput, pruning, or quantization benchmark has been measured.
- The test suite (80 test functions across 5 files) does not pass — it fails
  to import due to code/test drift.
- `pyproject.toml` has no `[project]`/`[build-system]` table, so
  `pip install -e .` fails. There is no `NESTDecoder` class and no PyPI package
  `nest-eeg`.

## Where the withdrawn "26.1% WER" figure came from

A figure of "26.1% WER" (with "BLEU 0.74" and "73.9% accuracy") previously
appeared in project documentation and was described as a result. It was not a
measurement. It came from `estimate_wer_from_ctc_loss()` in
`detailed_analysis.py`, a hand-written formula that converts the final CTC
training loss of the February 2026 `SimpleLSTMDecoder` run into a WER
percentage using empirical bucket ranges (e.g. loss < 2.5 → 15-20% WER). No
text was ever decoded from EEG and compared against a reference sentence to
produce this number. It has been withdrawn and must not be cited as a result.

Separately, WER figures such as "16.3% WER" that appear in some archived status
documents (`docs/status/`, `docs/phases/`, `docs/turkish/`) came from runs on
**synthetic, randomly generated data** (`scripts/generate_synthetic_data.py`),
not real EEG, and have no scientific meaning as a performance measurement.

## Evaluation commitment

Jo et al. (2024, *Are EEG-to-Text Models Working?*, arXiv:2405.06459) showed
that much of the accuracy reported by prior ZuCo-based EEG-to-text work was an
artifact of teacher forcing at evaluation time — feeding reference tokens back
into the decoder lets a language-model prior reconstruct much of the sentence
largely independent of the EEG input. NEST commits to reporting only:

1. **Free-decoding metrics**: free autoregressive decoding (greedy today; beam search planned) for NEST-BART-v2,
   greedy decoding for NEST-CTC-v2 — no reference text supplied at inference
   time.
2. **A noise-input control**: the same model and evaluation run with the EEG
   input replaced by uninformative noise, so that any gap between real-EEG and
   noise-input performance is visible alongside the headline number.
3. **Subject-independent evaluation**: held-out test subjects (ZMG, ZPH) never
   seen during training or validation.
4. **WER, CER, BLEU, and ROUGE**, reported together rather than cherry-picked.

Results will be published here, in the model card, and in the manuscript's
Results section once a full training run completes.

## Related documents

- [../README.md](../README.md) — project overview and quick start
- [MODEL_CARD.md](MODEL_CARD.md) — intended use, ethics, limitations (performance
  sections marked pending)
- [../papers/NEST_manuscript.md](../papers/NEST_manuscript.md) — draft
  manuscript; see its Section 5 for the full evaluation protocol
- [../ROADMAP.md](../ROADMAP.md) — Phase 7 tracks the validated benchmark
- [../CHANGELOG.md](../CHANGELOG.md) — Known issues
- [../TECHNICAL_AUDIT_REPORT.md](../TECHNICAL_AUDIT_REPORT.md) — original
  technical audit that flagged these issues
