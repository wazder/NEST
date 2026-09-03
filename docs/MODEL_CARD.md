# NEST Model Card

Model card for the Neural EEG Sequence Transducer (NEST) framework following the model card framework proposed by Mitchell et al. (2019).

## Model Details

### Basic Information

- **Model Name**: NEST (Neural EEG Sequence Transducer)
- **Model Version**: 1.0.0
- **Model Date**: February 2026
- **Model Type**: Sequence-to-sequence transduction
- **Architecture**: Encoder-Decoder with Attention
- **License**: MIT
- **Contact**: wazder@github.com
- **Repository**: https://github.com/wazder/NEST

### Model Description

NEST is a deep learning framework for decoding electroencephalography (EEG) signals directly into natural language text. The model uses a hybrid architecture combining:

1. **Spatial CNN**: Extracts spatial features from multi-channel EEG (EEGNet, DeepConvNet)
2. **Temporal Encoder**: Encodes temporal patterns (LSTM, Transformer, Conformer)
3. **Attention Mechanism**: Cross-attention between EEG features and text tokens
4. **Decoder**: Generates text sequences (Transformer decoder, RNN transducer)

The framework supports multiple architectural variants:
- **NEST-RNN-T**: RNN Transducer architecture
- **NEST-Transformer-T**: Transformer Transducer architecture
- **NEST-Attention**: Attention-based encoder-decoder (recommended)
- **NEST-CTC**: Connectionist Temporal Classification
- **NEST-Conformer**: Conformer-based architecture

### Model Variants

No variant has a trained checkpoint or a measured WER, CER, or latency. The two variants under active development are:

| Variant | Parameters | Status |
|---------|-----------|--------|
| NEST-CTC-v2 | 45,310,637 | Implemented; not yet trained to completion |
| NEST-BART-v2 | 184,424,592 total / 64,697,232 trainable | Implemented; not yet trained to completion |

Earlier architectural variants named in this document (NEST-Attention, NEST-RNN-T, NEST-Conformer) refer to an earlier iteration of the codebase and have no associated trained checkpoint or benchmark either.

## Intended Use

### Primary Intended Uses

- **Research**: Investigating EEG-to-text decoding methods
- **Silent Speech Interfaces**: Assistive communication for speech impairment
- **BCI Applications**: Brain-computer interface research
- **Neuroscience**: Understanding neural correlates of language

### Primary Intended Users

- Researchers in BCI and neuroscience
- Developers of assistive technologies
- Machine learning practitioners in healthcare
- Neuroscience educators and students

### Out-of-Scope Uses

The model should **NOT** be used for:

- Medical diagnosis or clinical decision-making
- Real-time life-critical applications (not FDA approved)
- Privacy-invasive thought reading without consent
- Applications outside research/assistive contexts
- Non-English languages (model trained on English only)

## Factors

### Relevant Factors

**Subject Variability**
- Performance is expected to vary across individuals, consistent with the wider EEG-to-text literature, but no NEST-specific per-subject WER has been measured yet
- Subject-specific adaptation recommended for deployment
- Cross-subject generalization challenging

**Signal Quality**
- Optimal performance with clean, artifact-free EEG
- Requires proper electrode placement and impedance
- Sensitive to environmental noise and motion artifacts

**Text Characteristics**
- Optimized for natural reading tasks
- Performance degrades on technical/rare vocabulary
- Sentence length: optimal 5-20 words

**Hardware Requirements**
- 105-channel EEG system (matching the ZuCo 1.0 recording montage used for training data)
- 500 Hz sampling rate recommended
- Standard 10-20 electrode placement

### Evaluation Factors

Models evaluated across:
- **Subjects**: 12 participants (8 train, 2 val, 2 test)
- **Age**: 18-35 years
- **Gender**: Balanced (6 male, 6 female)
- **Language**: Native English speakers
- **Tasks**: Reading comprehension (ZuCo dataset)

## Metrics

### Model Performance Metrics

**No trained checkpoint has been released, and no WER, CER, BLEU, or perplexity number has been measured for NEST v2.** A figure that previously appeared here ("16.5% WER", "8.3% CER", "BLEU 0.72") was not a measurement — it originated from a hand-written heuristic (`estimate_wer_from_ctc_loss()` in `detailed_analysis.py`) that converts training loss into a WER percentage, applied to an unrelated ad hoc smoke-test model, with no decoding or reference comparison involved. It has been withdrawn; see [STATUS.md](STATUS.md).

The evaluation code itself is real and implemented in `scripts/train_nest_v2.py`: `eval_bart()` decodes autoregressively with `model.generate()` (no teacher forcing; the `num_beams=4` argument is accepted but the current implementation is greedy) and `eval_ctc()` uses greedy CTC decoding; WER/CER are computed via word/character Levenshtein distance. These will be used to report results once training completes.

### Detailed Results

Pending, no trained checkpoint released. No train/validation/test metric table exists yet.

### Decision Thresholds

- **Confidence Threshold**: 0.7 (for uncertain predictions)
- **Minimum Sequence Length**: 3 tokens
- **Maximum Sequence Length**: 256 tokens

## Training Data

### Dataset

**Primary Dataset**: ZuCo (Zurich Cognitive Language Processing Corpus)
- **Source**: https://osf.io/q3zws/
- **Citation**: Hollenstein et al. (2018, 2020)
- **License**: CC BY 4.0

**Dataset Characteristics**
- **Tasks**: Reading comprehension — SR (normal reading), NR, and TSR (task-specific reading)
- **Subjects**: 12 participants (ZuCo 1.0)
- **Sentences**: 1,107 sentences (ZuCo 1.0)
- **EEG Data**: 105-channel recordings at 500 Hz
- **Language**: English

### Data Split

- **Training**: 8 subjects
- **Validation**: 2 subjects
- **Test**: 2 subjects (held out, never seen during training or validation)

**Split Strategy**: Subject-independent (no subject overlap between splits)

### Preprocessing

1. **Filtering**: Band-pass 0.5-50 Hz, notch filter at 60 Hz
2. **Artifact Removal**: ICA
3. **Feature Extraction**: 105-channel signal processed into 8 frequency bands (theta, alpha, beta, gamma sub-bands), yielding an 840-dimensional feature vector per fixated word
4. **Normalization**: Z-score normalization per channel
5. **Augmentation**: 
   - Gaussian noise (σ=0.1)
   - Time shifting (±50 ms)
   - Amplitude scaling (0.9-1.1)

### Tokenization

- **Type**: Character-level tokenization
- **Vocabulary**: 30 characters (a-z, space, punctuation)
- **Special Tokens**: [PAD], [BOS], [EOS], [UNK]

## Evaluation Data

### Test Set Characteristics

Same distribution as training data (ZuCo dataset) but held-out subjects.

**Test Scenarios**
1. **In-domain**: Same reading task as training
2. **Cross-subject**: Completely new subjects
3. **Noisy conditions**: Simulated artifacts added

### Evaluation Results by Scenario

Pending, no trained checkpoint released. Once training completes, results for these scenarios will be reported using free decoding (no teacher forcing) with a noise-input control, per the commitment in `docs/STATUS.md`.

## Ethical Considerations

### Privacy

- EEG data contains privacy-sensitive neural information
- Models could potentially decode unintended thoughts
- Informed consent required for all recordings
- Data anonymization critical

**Recommendations**:
- Obtain explicit consent for EEG recording and model training
- Implement user control over what is decoded/stored
- Secure storage and transmission of neural data
- Right to deletion of personal neural data

### Fairness

**Potential Biases**:
- Dataset limited to young, educated, native English speakers
- May not generalize to diverse populations
- Performance may vary by age, education, language background

**Mitigation Strategies**:
- Expand dataset to include diverse demographics
- Evaluate performance across demographic groups
- Subject-specific adaptation to reduce bias

### Use Cases and Risks

**Beneficial Uses**:
- Assistive communication for speech-impaired individuals
- Neuroscience research
- Educational tools

**Potential Risks**:
- Unauthorized thought surveillance
- Misuse for privacy invasion
- Over-reliance without proper validation
- Deployment without medical oversight

**Risk Mitigation**:
- Clear documentation of limitations
- User consent and control mechanisms
- Regular audits for fairness and accuracy
- Collaboration with ethics review boards

## Caveats and Recommendations

### Known Limitations

1. **English Only**: Trained exclusively on English text
2. **Reading Task**: Optimized for reading, not speech imagination
3. **Individual Variability**: Large performance variation across subjects
4. **Artifact Sensitivity**: Degraded performance with EEG artifacts
5. **Hardware Dependency**: Requires specific EEG setup

### Recommendations for Use

**For Researchers**:
- Use as baseline for EEG-to-text research
- Adapt architecture for specific tasks/populations
- Validate on your own dataset
- Report subject-level performance variance

**For Developers**:
- Implement subject-specific fine-tuning
- Add robust artifact rejection
- Include confidence estimation
- Validate extensively before deployment

**For Clinical Use**:
- **NOT currently approved for clinical use**
- Requires extensive validation and regulatory approval
- Should complement, not replace, existing AAC methods
- Medical oversight essential

### Environmental Impact

Pending, no trained checkpoint released. NEST v2 has not been trained to completion on GPU, so GPU-hours, estimated CO₂, and inference power/latency figures do not yet exist. The only completed training run (an ad hoc smoke-test model, not NEST v2) ran on an Apple M2 laptop CPU for approximately 5.4 hours; that run is not representative of NEST v2's eventual training cost.

## References

### Key Citations

1. **ZuCo Dataset**:
   - Hollenstein, N., et al. (2018). ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading. Scientific Data.
   - Hollenstein, N., et al. (2020). ZuCo 2.0: A dataset of physiological recordings during natural reading and annotation. LREC.

2. **Architectures**:
   - EEGNet: Lawhern et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces.
   - Conformer: Gulati et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition.
   - RNN Transducer: Graves (2012). Sequence transduction with recurrent neural networks.

3. **Methods**:
   - CTC Loss: Graves et al. (2006). Connectionist temporal classification.
   - Attention: Bahdanau et al. (2015). Neural machine translation by jointly learning to align and translate.

4. **Model Cards**:
   - Mitchell et al. (2019). Model Cards for Model Reporting.

### Related Work

- Willett et al. (2021). High-performance brain-to-text communication via handwriting.
- Makin et al. (2020). Machine translation of cortical activity to text with an encoder-decoder framework.
- Wang et al. (2021). Decoding English alphabet from EEG using LSTM networks.

## Model Card Authors

- Primary Author: NEST Development Team
- Contributors: NEST Development Team
- Review: Internal code review completed
- Last Updated: February 17, 2026
- Version: 1.0

## Model Card Contact

For questions, issues, or feedback:
- GitHub Issues: https://github.com/wazder/NEST/issues
- Email: wazder@github.com
- Documentation: https://github.com/wazder/NEST/docs

---

**Acknowledgments**: This work uses the ZuCo dataset and builds upon numerous open-source contributions. We thank all contributors to the field of BCI and neural decoding research.
