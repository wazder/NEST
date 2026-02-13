# NEST Literature Review: Phase 1 Foundation

## Overview
This literature review provides a comprehensive foundation for the NEST (Neural EEG Sequence Transducer) project. It covers the essential research areas needed to develop an end-to-end EEG-to-text decoding system for Brain-Computer Interfaces (BCI) and Silent Speech Interfaces (SSI).

## Structure
This review is organized into five key areas:

1. **[Sequence Transducers in BCI](./01-sequence-transducers.md)**
   - RNN Transducers (RNN-T)
   - Neural Transducers
   - Applications in BCI systems

2. **[EEG-to-Text Decoding Approaches](./02-eeg-to-text-decoding.md)**
   - Existing methodologies
   - Current limitations
   - State-of-the-art systems

3. **[Attention Mechanisms for Neural Signals](./03-attention-mechanisms.md)**
   - Self-attention in neural signal processing
   - Cross-attention for sequence transduction
   - Temporal attention for EEG

4. **[Silent Speech Interface Methodologies](./04-silent-speech-interfaces.md)**
   - SSI paradigms
   - Neural correlates of speech
   - Decoding strategies

5. **[Benchmarks and Evaluation Metrics](./05-benchmarks-metrics.md)**
   - Standard datasets (ZuCo, etc.)
   - Evaluation metrics (WER, BLEU, CER)
   - Comparative analysis frameworks

## Key Findings Summary

### Sequence Transducers
- RNN-T architecture provides streaming capability essential for real-time BCI
- Neural Transducers offer superior alignment learning compared to traditional approaches
- Attention-based transducers show promise for handling variable-length EEG sequences

### EEG-to-Text Decoding
- Most existing approaches rely on intermediate classification steps
- End-to-end models are emerging but remain in early stages
- Cross-subject generalization remains a significant challenge

### Attention Mechanisms
- Self-attention effectively captures spatial dependencies between EEG channels
- Temporal attention is crucial for modeling time-varying neural dynamics
- Cross-attention bridges the gap between neural and linguistic representations

### Silent Speech Interfaces
- Imagined speech shows detectable EEG patterns
- Multi-band spectral features improve decoding accuracy
- Subject-specific calibration significantly impacts performance

### Evaluation Standards
- Word Error Rate (WER) is the primary metric for speech decoding
- BLEU scores assess semantic coherence
- Real-time latency requirements: <300ms for natural interaction

## Research Gaps Identified

1. **Limited Open-Vocabulary Systems**: Most BCI systems are restricted to command-based interfaces
2. **Cross-Lingual Models**: Minimal research on non-English languages, especially agglutinative languages like Turkish
3. **End-to-End Learning**: Few studies attempt direct EEG-to-text mapping without intermediate steps
4. **Standardized Benchmarks**: Lack of consensus on evaluation protocols for EEG-based text generation

## Implications for NEST

The literature review validates the NEST approach and identifies several opportunities:

1. **Architectural Innovation**: Combining Transformer-based encoding with transducer decoding
2. **Cross-Lingual Capability**: Turkish language support addresses an underserved research area
3. **End-to-End Pipeline**: Direct transduction without intermediate phoneme/word classification
4. **Standardized Preprocessing**: Reproducible pipelines for public datasets (ZuCo)

## References
See individual section documents for detailed citations and bibliography.

## Last Updated
February 2026
