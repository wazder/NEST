# NEST: Neural EEG Sequence Transducer

## Abstract
The NEST framework aims to develop a novel deep learning architecture capable of decoding non-invasive Electroencephalography (EEG) signals directly into natural language text. By bridging the gap between Neuroscience and Natural Language Processing (NLP), this research focuses on translating neural activity into coherent **English** sentences.

Unlike traditional BCI systems restricted to limited commands, NEST utilizes advanced sequence-to-sequence architectures (Transformers / Transducer-based models) to achieve open-vocabulary speech decoding for Silent Speech Interfaces (SSI).

## Research Objectives
1. **End-to-End Transduction:** Implementation of a pipeline that maps raw EEG waves directly to text tokens without intermediate classification steps.
2. **Robust Generalization:** Investigation of the model's performance across different subjects and conditions for English language decoding.
3. **Reproducibility:** Development of standardized preprocessing scripts for public EEG datasets (e.g., ZuCo).

## Methodology
The proposed NEST architecture consists of three main stages:
- **Signal Preprocessing:** Band-pass filtering and artifact removal strategies.
- **Neural Encoder:** A hybrid CNN-LSTM or Transformer-based encoder to extract spatial-temporal features.
- **Sequence Transducer:** An attention-based decoder aimed at generating contextually accurate text sequences from neural embeddings.

## Roadmap
- ✓ Literature review regarding Sequence Transducers in BCI ([Phase 1 Complete](docs/literature-review/))
- Preprocessing pipeline development for the ZuCo dataset
- Implementation of the NEST Encoder-Decoder architecture
- Advanced feature development and model robustness
- Submission of the research paper to relevant BCI/AI conferences

> See [ROADMAP.md](ROADMAP.md) for detailed project milestones.

## Documentation
- **[Literature Review](docs/literature-review/)**: Comprehensive Phase 1 foundation covering:
  - Sequence Transducers (RNN-T, Neural Transducers) in BCI applications
  - EEG-to-text decoding approaches and their limitations
  - Attention mechanisms for neural signal processing
  - Silent Speech Interface (SSI) methodologies
  - State-of-the-art benchmarks and evaluation metrics