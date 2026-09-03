# NEST Project Roadmap

## Phase 1: Literature Review & Foundation
- [ ] Comprehensive survey of Sequence Transducers (RNN-T, Neural Transducers) in BCI applications
- [ ] Analysis of existing EEG-to-text decoding approaches and their limitations
- [ ] Review of attention mechanisms for neural signal processing
- [ ] Study of Silent Speech Interface (SSI) methodologies
- [ ] Documentation of state-of-the-art benchmarks and evaluation metrics

**Status:** To be re-implemented with automated pipeline  
**Documentation Target:** [docs/literature-review/](docs/literature-review/)

## Phase 2: Data Acquisition & Preprocessing
- [x] ZuCo dataset acquisition and exploratory data analysis
- [x] Implementation of band-pass filtering (0.5–50 Hz) for artifact removal
- [x] Development of Independent Component Analysis (ICA) pipeline for eye-blink/muscle artifact rejection
- [x] Electrode selection and channel optimization strategies
- [x] Data augmentation techniques for limited EEG samples
- [x] Train/validation/test split with subject-independent evaluation protocol

**Status:** ✅ Complete  
**Documentation:** [docs/phase2-preprocessing.md](docs/phase2-preprocessing.md)

## Phase 3: Model Architecture Development
- [x] Implementation of CNN-based spatial feature extractor (SpatialCNN, EEGNet, DeepConvNet)
- [x] Development of Temporal Encoder (LSTM/GRU/Transformer/Conformer variants)
- [x] Design of cross-attention mechanism between EEG embeddings and text tokens
- [x] Integration of Connectionist Temporal Classification (CTC) loss
- [x] Implementation of RNN-Transducer and Transformer-Transducer architectures
- [x] Training utilities and metrics (WER, CER, BLEU, Perplexity)
- [x] Model factory and configuration system
- [x] Checkpoint management utilities

**Status:** ✅ Complete  
**Documentation:** [docs/phase3-models.md](docs/phase3-models.md)

## Phase 4: Advanced Model Features & Robustness
- [x] Implementation of advanced attention mechanisms (relative position, local, linear)
- [x] Development of robust subword vocabularies using BPE/SentencePiece
- [x] Subject-independent generalization and cross-subject transfer learning (DANN, CORAL)
- [x] Handling of noisy EEG signals and artifact robustness (adversarial training, denoising)
- [x] Integration of pre-trained language models for improved decoding (fusion, rescoring)
- [x] Subject adaptation techniques (embeddings, adaptive batch norm)
- [x] Fine-tuning strategies for new subjects

**Status:** ✅ Complete  
**Documentation:** [docs/phase4-advanced-features.md](docs/phase4-advanced-features.md)

## Phase 5: Evaluation & Optimization
- [x] Benchmark evaluation code using Word Error Rate (WER), CER, and BLEU scores — implemented, unbenchmarked (no trained checkpoint to evaluate yet)
- [x] Beam search decoder with length normalization and coverage penalty — implemented, unbenchmarked
- [x] Latency and real-time inference optimization (ONNX, TorchScript, FP16) — implemented, unbenchmarked
- [x] Model compression through pruning (magnitude, structured, iterative, sensitivity) — implemented, unbenchmarked
- [x] Model quantization (PTQ, QAT, dynamic, mixed-precision) — implemented, unbenchmarked
- [x] Real-time streaming inference pipeline targeting <100ms latency — implemented, unbenchmarked; no measured latency exists
- [x] Comprehensive profiling tools (FLOPs, memory, throughput, layer timing) — implemented, unbenchmarked
- [x] Deployment utilities (model export, packaging, configuration) — implemented, unbenchmarked
- [ ] User study design for practical SSI applications

**Status:** Code implemented; not yet benchmarked. See `docs/STATUS.md`.  
**Documentation:** [docs/phase5-evaluation-optimization.md](docs/phase5-evaluation-optimization.md)

## Phase 6: Documentation & Dissemination
- [x] Preparation of reproducible codebase with documentation
- [x] Research paper outline prepared for BCI/AI conferences (NeurIPS, EMNLP, IEEE EMBC) — implemented, unbenchmarked (manuscript is a draft with no results yet, see `papers/NEST_manuscript.md`)
- [x] Open-source release preparation with examples and model cards — implemented; model card numbers marked pending, no trained checkpoint released
- [ ] Submission of research paper to relevant BCI/AI conferences
- [ ] User study design for practical SSI applications

**Status:** Documentation and tooling in place; paper submission blocked on Phase 7 (validated benchmark).  
**Documentation:** [PHASE6_SUMMARY.md](PHASE6_SUMMARY.md)

## Phase 7: Validated Benchmark (2026 Q4)
- [ ] GPU training of NEST v2 (NEST-CTC-v2 and NEST-BART-v2) to completion
- [ ] Free-decoding WER/CER/BLEU/ROUGE on the subject-independent test split, with a noise-input control following Jo et al. (2024)
- [ ] Publish the validated benchmark results and update `docs/STATUS.md`, the model card, and the manuscript's Results section accordingly
- [ ] arXiv technical report with the validated results

**Status:** Not started.

---

## Project Status (honest, September 2026)

The NEST codebase — preprocessing pipeline, multiple model architectures, advanced features, optimization/deployment tooling, and documentation — is implemented (Phases 2-6). It is **not production-ready and has no validated benchmark**:
- No NEST v2 model has completed training to a released checkpoint
- No WER/CER/BLEU number has been measured against reference text
- The test suite currently fails to import and does not pass
- `pip install -e .` does not work (no packaging metadata in `pyproject.toml`)

See [docs/STATUS.md](docs/STATUS.md) for the single source of truth on what exists, what is verified, and what is not.

**Next Steps:**
- Phase 7: validated benchmark (GPU training + free-decoding evaluation with noise-input control)
- Research paper submission, once results exist
- Open-source release of a trained checkpoint
- Community building and maintenance
