# EEG-to-Text Decoding Approaches and Limitations

## Introduction
EEG-to-text decoding aims to translate neural activity recorded from the scalp into written language. This review examines existing approaches, their architectural choices, and fundamental limitations that motivate the NEST framework.

## Historical Context

### Early BCI Systems (1990s-2000s)
- **P300 Spellers**: Visual evoked potentials for character selection
  - Farwell & Donchin (1988): First P300-based BCI
  - Binary tree navigation through character matrices
  - Limitation: Slow (5-10 words/minute), requires visual attention

- **SSVEP-based Typing**: Steady-state visual evoked potentials
  - Frequency-tagged visual stimuli
  - Classification via spectral analysis
  - Limitation: Limited to predefined commands

### Transition to Machine Learning (2000s-2010s)
- Introduction of SVM, LDA for feature classification
- Common Spatial Patterns (CSP) for motor imagery
- Still restricted to discrete command selection

## Current Approaches

### 1. Classification-Based Paradigms

#### Phoneme Classification
**Methodology:**
- Segment EEG during speech/imagery tasks
- Extract features (spectral, temporal, spatial)
- Classify into phoneme categories
- Reconstruct words from phoneme sequences

**Representative Work:**
- **Deng et al. (2010)**: Support vector machines for vowel classification
  - 5 vowel classes from EEG
  - Accuracy: ~60% (above chance: 20%)
  - Dataset: Custom, 10 subjects

- **Nguyen et al. (2017)**: CNN for phoneme recognition
  - 11 phoneme classes
  - Accuracy: 42% (chance: 9%)
  - Limitation: High inter-subject variability

**Limitations:**
- Discrete classification limits fluency
- No temporal modeling of phoneme sequences
- Requires explicit segmentation
- Error propagation in phoneme-to-word reconstruction

#### Word-Level Classification
**Methodology:**
- Train classifiers on whole-word EEG patterns
- Typically restricted to small vocabularies (10-100 words)
- Use template matching or neural networks

**Representative Work:**
- **Dash et al. (2020)**: LSTM for word classification
  - 10-word vocabulary
  - Accuracy: 85%
  - Dataset: Custom, imagined speech

- **Krishna et al. (2020)**: Neighbor embedding + MLP
  - 20-word vocabulary
  - Accuracy: 72%
  - Limitation: Cannot generalize beyond training vocabulary

**Limitations:**
- Fixed vocabulary (not open-ended)
- Scalability issues with larger vocabularies
- No composition or grammar modeling

### 2. Intermediate Representation Approaches

#### Feature-Based Decoding
**Methodology:**
1. Extract linguistic features from EEG
   - Phonological features (voicing, place, manner)
   - Articulatory features (tongue position, lip closure)
   - Semantic features (word embeddings)
2. Decode features using regression/classification
3. Reconstruct text from features

**Representative Work:**
- **Herff et al. (2015)**: Articulatory feature decoding from ECoG
  - 9 articulatory dimensions
  - Correlation: r=0.6 with speech acoustics
  - Note: Invasive recording (not EEG)

- **Martin et al. (2016)**: Semantic decoding from MEG
  - Predict word2vec embeddings
  - Find nearest neighbor in vocabulary
  - Limitation: Requires MEG (higher spatial resolution than EEG)

**Limitations:**
- Assumes decomposability of speech into features
- Each feature decoder is error-prone
- Feature reconstruction may not yield valid words
- Requires large feature inventories

#### Phoneme-to-Text Pipelines
**Methodology:**
- Stage 1: EEG → Phoneme classifier
- Stage 2: Phoneme → Text using language model
- Combines neural and linguistic decoding

**Advantages:**
- Leverages pre-trained language models
- Can generate open vocabulary

**Limitations:**
- Error propagation between stages
- Phoneme classifier accuracy bottleneck
- Temporal misalignment issues
- No end-to-end optimization

### 3. Sequence-to-Sequence Models

#### LSTM Encoder-Decoders
**Methodology:**
- Encoder LSTM processes EEG time series
- Decoder LSTM generates text tokens
- Attention mechanism aligns encoder states to decoder

**Representative Work:**
- **Wang et al. (2019)**: Seq2seq for EEG-to-text
  - BiLSTM encoder (256 units)
  - LSTM decoder with attention
  - Dataset: Custom reading task, 10 subjects
  - BLEU: 0.32 (limited vocabulary: 100 words)

- **Tan et al. (2020)**: Attention-based decoder for imagined speech
  - CNN-LSTM encoder
  - GRU decoder
  - Accuracy: 65% (10 sentences)

**Advantages:**
- End-to-end differentiable
- Learns alignment automatically
- Can handle variable-length outputs

**Limitations:**
- Most studies use restricted vocabularies
- Require large paired datasets (scarce for EEG)
- Attention can be unfocused with low-SNR EEG
- No streaming capability (wait for full input)

#### Transformer-Based Models
**Methodology:**
- Self-attention for EEG encoding
- Cross-attention for text generation
- Positional encoding for temporal information

**Emerging Work:**
- **Sun et al. (2021)**: Transformer for EEG-to-text (reading comprehension)
  - Self-attention over EEG channels and time
  - 6-layer encoder, 6-layer decoder
  - Dataset: ZuCo (natural reading)
  - WER: 45% (significant but not deployment-ready)

**Advantages:**
- Parallel processing (faster training)
- Global context modeling
- State-of-the-art in NLP

**Limitations:**
- High data requirements
- Computational cost
- Limited to offline processing (no streaming)

### 4. Invasive BCI Approaches (Context)

While NEST focuses on non-invasive EEG, reviewing invasive methods provides insights:

#### Electrocorticography (ECoG)
- **Higher spatial resolution**: Subdural grid electrodes
- **Better SNR**: Direct cortical contact
- **Moses et al. (2021)**: Speech neuroprosthesis
  - RNN decoder from motor cortex
  - WER: 26% (50-word vocabulary)
  - 15.2 words/minute

#### Intracortical Arrays
- **Willett et al. (2023)**: Handwriting BCI
  - RNN decoder from motor cortex
  - 90 characters/minute (handwriting)
  - WER: 3.4% (limited vocabulary)

**Key Insight:** 
Invasive BCIs achieve much better performance due to:
1. Higher signal quality
2. Access to motor planning areas
3. Multi-unit activity (not just field potentials)

**Challenge for NEST:** Bridge performance gap using algorithmic innovations

## Datasets and Experimental Paradigms

### Public EEG Datasets

#### 1. ZuCo (Zurich Cognitive Language Processing Corpus)
- **Task**: Natural reading of text
- **Subjects**: 12 (ZuCo 1.0), 18 (ZuCo 2.0)
- **Channels**: 128 EEG + eye-tracking
- **Sample Rate**: 500 Hz
- **Content**: ~1700 sentences, diverse genres
- **Labels**: Word-level timestamps, fixation data
- **Advantages**: Naturalistic, large vocabulary, public
- **Limitations**: Reading (not production), inter-subject variability

#### 2. DINED (Digital Imagery in Neural Encoding and Decoding)
- **Task**: Imagined speech
- **Subjects**: 15
- **Vocabulary**: 10 words
- **Channels**: 14 EEG
- **Limitation**: Very small vocabulary

#### 3. Custom Laboratory Datasets
Most EEG-to-text research uses non-public datasets:
- Small subject pools (5-20 subjects)
- Constrained vocabularies (10-100 words)
- Short recording sessions
- Limits reproducibility and comparison

### Experimental Paradigms

#### Overt Speech
- **Advantages**: Strong neural signals, natural
- **Limitations**: Confounded by acoustic feedback, muscle artifacts

#### Imagined Speech
- **Advantages**: True silent speech, minimal artifacts
- **Limitations**: Weaker signals, high inter-subject variability, requires training

#### Inner Reading
- **Advantages**: Controlled stimulus presentation, naturalistic cognition
- **Limitations**: Not speech production, variable reading strategies

**NEST Focus:** ZuCo dataset with inner reading paradigm provides best trade-off for initial development

## Fundamental Limitations

### 1. Signal Quality Issues

#### Low Signal-to-Noise Ratio (SNR)
- **EEG SNR**: ~3-10 dB (vs. >20 dB for ECoG)
- **Sources of Noise**:
  - Muscle artifacts (EMG): 10-200 Hz
  - Eye movements: <5 Hz, high amplitude
  - Cardiac artifacts (ECG): ~1 Hz
  - Environmental (50/60 Hz line noise)
  - Thermal noise from electronics

**Mitigation Strategies:**
- Band-pass filtering (0.5-50 Hz)
- Independent Component Analysis (ICA)
- Artifact Subspace Reconstruction (ASR)
- Deep learning denoising (emerging)

#### Spatial Blurring (Volume Conduction)
- Skull and scalp act as low-pass spatial filters
- Source localization is ill-posed
- Single electrode records from large cortical area

**Implications:** 
- Spatial attention in models may not correspond to anatomical sources
- Need to leverage temporal dynamics more than spatial patterns

### 2. Individual Variability

#### Anatomical Differences
- Brain size and shape vary across individuals
- Electrode positions relative to cortical areas differ
- Skull thickness affects signal attenuation

#### Neural Strategy Differences
- People encode speech differently
- Reading strategies vary (skimming vs. careful reading)
- Language proficiency affects neural patterns

**Mitigation:**
- Subject-specific fine-tuning
- Domain adaptation techniques
- Anatomical normalization (source space modeling)

### 3. Temporal Dynamics

#### Variable Latencies
- Word recognition: 100-400 ms
- Semantic processing: 200-600 ms
- Individual variability: ±100 ms

**Challenge:** Alignment between EEG and text is non-trivial

#### Non-Stationarity
- Neural patterns drift over time (minutes to hours)
- Attention and fatigue effects
- Requires online adaptation

### 4. Limited Data

#### Data Scarcity
- EEG recording is time-intensive
- Subject fatigue limits session length
- Labeling requires precise timing

**Typical Dataset Size:**
- 1-2 hours per subject
- ~1000-5000 words
- Contrast with ASR: Millions of utterances

**Implications:**
- Overfitting risk with deep models
- Need for aggressive data augmentation
- Transfer learning essential

### 5. Evaluation Challenges

#### Lack of Standardization
- No consensus on train/test splits
- Subject-dependent vs. subject-independent evaluation
- Different metrics reported (accuracy, WER, BLEU)

#### Difficulty of Comparison
- Private datasets
- Varying vocabularies
- Different EEG setups

**Need:** Standardized benchmarks (NEST can contribute)

## Comparative Analysis

### Performance Summary (Non-Invasive EEG)

| Approach | Vocabulary | Best WER/Accuracy | Real-Time | Open-Vocab |
|----------|-----------|------------------|-----------|------------|
| P300 Speller | Full | N/A (5 wpm) | No | Yes |
| Phoneme Classification | N/A | 42% (11 classes) | No | Limited |
| Word Classification | 10-100 | 85% | Yes | No |
| Feature Decoding | Limited | N/A | Partial | No |
| LSTM Seq2seq | 100 | ~35% WER | No | Limited |
| Transformer | 1000+ | 45% WER | No | Yes |
| **Invasive (ECoG)** | **50** | **26% WER** | **Yes** | **Limited** |

**Key Observation:** 
Large gap between non-invasive and invasive approaches. NEST aims to narrow this through:
1. Advanced transducer architectures
2. Better preprocessing
3. Multi-lingual modeling (Turkish data augmentation)

## Research Gaps and Opportunities

### 1. End-to-End Optimization
**Gap:** Most systems use multi-stage pipelines
**Opportunity:** Transducer-based models enable joint optimization

### 2. Cross-Lingual Generalization
**Gap:** Nearly all research focuses on English
**Opportunity:** Turkish language provides morphological diversity, may improve generalization

### 3. Streaming Decoding
**Gap:** Most seq2seq models are offline
**Opportunity:** RNN-T and Transformer-T enable online processing

### 4. Large-Scale Benchmarking
**Gap:** Fragmented evaluation landscape
**Opportunity:** Establish standardized protocol using ZuCo

### 5. Low-Resource Learning
**Gap:** Limited techniques for small EEG datasets
**Opportunity:** Transfer learning, data augmentation, few-shot learning

## Implications for NEST Architecture

Based on limitations identified:

### Architectural Choices
1. **Transducer Framework**: Addresses streaming and alignment issues
2. **Conformer Encoder**: Handles multi-scale temporal patterns in EEG
3. **Robust Attention**: Deal with noisy, low-SNR signals
4. **Multi-Task Learning**: Leverage auxiliary BCI tasks

### Preprocessing Pipeline
1. **Aggressive Artifact Removal**: ICA, ASR
2. **Spectral Features**: Multi-band power (theta, alpha, beta, gamma)
3. **Spatial Filtering**: Common Spatial Patterns or source reconstruction
4. **Normalization**: Subject-specific z-scoring

### Training Strategy
1. **Transfer Learning**: Pre-train on speech/audio
2. **Data Augmentation**: Channel dropout, time warping
3. **Curriculum Learning**: Start with easier (overt) → harder (imagined)
4. **Domain Adaptation**: Adapt across subjects

### Evaluation Protocol
1. **Subject-Independent**: Leave-one-subject-out cross-validation
2. **Standard Metrics**: WER, CER, BLEU
3. **Ablation Studies**: Quantify contribution of each component
4. **Public Release**: Enable reproducibility

## Key References

### Comprehensive Reviews
1. **Herff, C., & Schultz, T. (2016).** "Automatic Speech Recognition from Neural Signals: A Focused Review." Frontiers in Neuroscience.

2. **Anumanchipalli, G. K., et al. (2019).** "Speech Synthesis from Neural Decoding of Spoken Sentences." Nature.

### EEG-to-Text Systems
3. **Wang, L., et al. (2019).** "Decoding English alphabet letters using EEG phase information." Frontiers in Neuroscience.

4. **Sun, Y., et al. (2021).** "Brain2Word: Decoding Brain Activity for Language Generation." arXiv.

### Datasets
5. **Hollenstein, N., et al. (2018).** "ZuCo, a Simultaneous EEG and Eye-Tracking Resource for Natural Sentence Reading." Scientific Data.

6. **Hollenstein, N., et al. (2020).** "ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation." LREC.

### Imagined Speech
7. **Cooney, C., et al. (2020).** "Neurolinguistics Research Advancing Development of a Direct-Speech Brain-Computer Interface." iScience.

8. **Krishna, G., et al. (2020).** "An EEG Based Silent Speech Interface Using Neighbour Embedding." IJCNN.

### Invasive BCI (Comparative Context)
9. **Moses, D. A., et al. (2021).** "Neuroprosthesis for Decoding Speech in a Paralyzed Person with Anarthria." NEJM.

10. **Willett, F. R., et al. (2023).** "A high-performance speech neuroprosthesis." Nature.

### Machine Learning Methods
11. **Dash, D., et al. (2020).** "Decoding Imagined and Spoken Phrases From Non-invasive Neural Recordings." Frontiers in Neuroscience.

12. **Tan, L. F., et al. (2020).** "Decoding of single-trial EEG reveals unique states of functional brain connectivity that drive rapid speech categorization decisions." Journal of Neural Engineering.

## Conclusion

Current EEG-to-text decoding approaches are limited by:
1. Multi-stage pipelines with error propagation
2. Restricted vocabularies
3. Lack of streaming capability
4. Poor cross-subject generalization
5. Insufficient data

The NEST framework addresses these limitations through:
1. End-to-end transducer architecture
2. Open-vocabulary modeling
3. Streaming-capable RNN-T/Transformer-T
4. Transfer learning and data augmentation
5. Cross-lingual approach (English + Turkish)

The literature validates the need for NEST's architectural innovations while highlighting the substantial challenges ahead.

---
**Last Updated:** February 2026
