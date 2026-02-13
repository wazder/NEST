# NEST Project Roadmap

## Phase 1: Literature Review & Foundation ✓
- [x] Comprehensive survey of Sequence Transducers (RNN-T, Neural Transducers) in BCI applications
- [x] Analysis of existing EEG-to-text decoding approaches and their limitations
- [x] Review of attention mechanisms for neural signal processing
- [x] Study of Silent Speech Interface (SSI) methodologies
- [x] Documentation of state-of-the-art benchmarks and evaluation metrics

**Status:** Completed February 2026  
**Documentation:** See [docs/literature-review/](docs/literature-review/) for comprehensive review covering all five areas.

## Phase 2: Data Acquisition & Preprocessing
- [ ] ZuCo dataset acquisition and exploratory data analysis
- [ ] Implementation of band-pass filtering (0.5–50 Hz) for artifact removal
- [ ] Development of Independent Component Analysis (ICA) pipeline for eye-blink/muscle artifact rejection
- [ ] Electrode selection and channel optimization strategies
- [ ] Data augmentation techniques for limited EEG samples
- [ ] Train/validation/test split with subject-independent evaluation protocol

## Phase 3: Model Architecture Development
- [ ] Implementation of CNN-based spatial feature extractor
- [ ] Development of Temporal Encoder (LSTM/Transformer variants)
- [ ] Design of cross-attention mechanism between EEG embeddings and text tokens
- [ ] Integration of Connectionist Temporal Classification (CTC) loss
- [ ] Experimentation with RNN-Transducer vs. Transformer-Transducer architectures
- [ ] Hyperparameter optimization and ablation studies

## Phase 4: Cross-Lingual Generalization
- [ ] Integration of Turkish morphological analyzer for agglutinative tokenization
- [ ] Development of language-specific subword vocabularies (BPE/SentencePiece)
- [ ] Implementation of multilingual decoder with shared encoder representations
- [ ] Evaluation of zero-shot and few-shot transfer between English and Turkish

## Phase 5: Evaluation & Optimization
- [ ] Benchmark evaluation using Word Error Rate (WER) and BLEU scores
- [ ] Latency and real-time inference optimization
- [ ] Model compression and quantization for edge deployment
- [ ] User study design for practical SSI applications

## Phase 6: Documentation & Dissemination
- [ ] Preparation of reproducible codebase with documentation
- [ ] Submission of research paper to relevant BCI/AI conferences (NeurIPS, EMNLP, IEEE EMBC)
- [ ] Open-source release of preprocessing scripts and pretrained models
