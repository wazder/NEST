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

## Phase 4: Advanced Model Features & Robustness
- [ ] Implementation of advanced attention mechanisms (multi-head, self-attention variants)
- [ ] Development of robust subword vocabularies using BPE/SentencePiece
- [ ] Subject-independent generalization and cross-subject transfer learning
- [ ] Handling of noisy EEG signals and artifact robustness
- [ ] Integration of pre-trained language models for improved decoding

## Phase 5: Evaluation & Optimization
- [ ] Benchmark evaluation using Word Error Rate (WER) and BLEU scores
- [ ] Latency and real-time inference optimization
- [ ] Model compression and quantization for edge deployment
- [ ] User study design for practical SSI applications

## Phase 6: Documentation & Dissemination
- [ ] Preparation of reproducible codebase with documentation
- [ ] Submission of research paper to relevant BCI/AI conferences (NeurIPS, EMNLP, IEEE EMBC)
- [ ] Open-source release of preprocessing scripts and pretrained models
