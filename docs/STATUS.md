# NEST Project Status

This document is the single source of truth for what NEST currently is, what has
actually been verified, and what has not. Where other documents in this
repository (status reports, phase summaries, the model card, the manuscript)
disagree with this file, this file is correct.

Last updated: 27 October 2026.

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
  ran to completion on NEST v2 for the first time in October 2026.
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

- **The first NEST v2 benchmark run** (October 27, 2026): `NEST_BART_v2` trained
  on real ZuCo EEG and gaze, evaluated against real EEG, gaze-only and noise
  input. See [First measured result](#first-measured-result-october-2026).

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
- **No NEST model decodes text from EEG.** The first full NEST v2 run (below)
  measured a WER near 1.0 for real EEG, gaze-only and noise input alike. The
  February 2026 `SimpleLSTMDecoder` run was never decoded, and `NEST_CTC_v2`
  has not been trained to completion.
- No BLEU, ROUGE, or perplexity number has been measured for any NEST model.
- No latency, throughput, pruning, or quantization benchmark has been measured.
- The test suite (80 test functions across 5 files) does not pass — it fails
  to import due to code/test drift.
- `pyproject.toml` has no `[project]`/`[build-system]` table, so
  `pip install -e .` fails. There is no `NESTDecoder` class and no PyPI package
  `nest-eeg`.

## First measured result (October 2026)

The first full NEST v2 run finished on 27 October 2026. **NEST-BART-v2 does not
decode text from EEG under honest evaluation.** Real EEG does not beat noise
input, and it does worse than gaze-only input.

**Setup**

- Code: branch `fix/bart-encoder-scale` (commit `59c133f`), which includes the
  encoder output-scale fix. An earlier run without that fix collapsed to one
  repeated token; it is not reported as a result.
- Model: `NEST_BART_v2` (`facebook/bart-base` decoder, 768-d EEG encoder,
  6 layers), 65,888,400 trainable parameters, trained on real EEG and gaze-only
  input in alternating batches (`--input-mode eeg gaze`).
- Training: AdamW, learning rate 3e-4 for the EEG encoder and 3e-5 for the BART
  weights (`--decoder-lr`), fp16, batch 16 with gradient accumulation 4. Early
  stopping ended training at epoch 51 (patience 20); best validation WER 0.9637.
- Data: ZuCo 1.0, tasks SR, NR and TSR. Subjects split 8/2/2 (test: ZMG, ZPH).
  Sentences are also split 80/10/10 by a hash of the normalized text
  (`--heldout-sentences`), so no test sentence appears in training.
- Evaluation: `scripts/eval_conditions.py`, free greedy decoding, the same
  checkpoint for all three inputs. Noise input is Gaussian noise matched to the
  per-feature mean and standard deviation of the EEG. Paired sign-flip
  permutation tests (10,000 permutations) use one unit per unique sentence, with
  Holm correction over all six tests.
- Hardware: NVIDIA Jetson AGX Orin. Training and evaluation together took about
  88 minutes.

**Held-out subjects, held-out sentences** (205 samples, 108 sentences)

| Input | WER | CER |
|---|---|---|
| Real EEG | 0.9992 | 0.8144 |
| Gaze only | 0.9699 | 0.7780 |
| Noise | 0.9846 | 0.7903 |

| Comparison | WER difference | p (Holm) | CER difference | p (Holm) |
|---|---|---|---|---|
| EEG minus gaze | +0.0293 | 0.0006 | +0.0365 | 0.0006 |
| EEG minus noise | +0.0147 | 0.0881 | +0.0242 | 0.0252 |
| Gaze minus noise | -0.0147 | 0.0006 | -0.0123 | 0.0006 |

A positive difference means the first input scores worse.

**Held-out subjects, sentences seen in training** (1,636 samples, 818
sentences): real EEG 0.9744 WER / 0.7983 CER, gaze 0.9526 / 0.7740, noise
0.9675 / 0.7868. EEG is worse than gaze and noise here too (all p (Holm)
0.0023 or lower).

**Shuffled-EEG control** (`scripts/eval_noise_control.py --noise shuffled`,
held-out subjects, all 2,047 samples): real EEG 0.9793 WER, shuffled EEG
0.9844, difference +0.0052.

**What the model does.** The decoder writes fluent sentences, but mostly the
same few training sentences whatever the input. For the 205 held-out test
samples it produced 13 distinct outputs from real EEG, 3 from gaze and 3 from
noise. The most frequent output, "he was a member of the labour party for the
eastern shore.", appears for 79 EEG, 200 gaze and 142 noise inputs. This is the
language-prior failure that Jo et al. (2024) describe.

**What this shows and what it does not.** This is one run with one seed and
one configuration, on a small corpus (825 training sentences). It does not prove
that the EEG carries no linguistic information. It shows that this model,
evaluated without teacher forcing on unseen subjects and sentences, does not
decode text from EEG, and that its small differences between inputs are not
evidence of decoding. Gaze-only input beats noise by 1.5 WER points, so the gaze
measures (fixation counts and durations) may carry some coarse information,
such as word or sentence length.

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

The first results are in [First measured result](#first-measured-result-october-2026)
above and in the model card. The manuscript's Results section is not updated yet.

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
