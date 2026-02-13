# Benchmarks and Evaluation Metrics for EEG-Based Text Decoding

## Introduction
Rigorous evaluation is essential for measuring progress in EEG-to-text decoding systems. This document reviews standard datasets (benchmarks), evaluation metrics, and best practices for assessing the NEST framework and comparing it with state-of-the-art approaches.

## Standard Datasets (Benchmarks)

### 1. ZuCo (Zurich Cognitive Language Processing Corpus)

#### ZuCo 1.0 (2018)
**Task:** Natural sentence reading
- **Subjects**: 12 native English speakers (6 female, 6 male)
- **Age**: 23-57 years
- **Channels**: 128 EEG (BioSemi ActiveTwo), eye-tracking (EyeLink 1000)
- **Sampling Rate**: 500 Hz (EEG), 1000 Hz (eye-tracking)
- **Content**: 
  - ~1,100 sentences from movie reviews (Stanford Sentiment Treebank)
  - ~400 sentences from Wikipedia
- **Vocabulary**: ~3,500 unique words
- **Labels**: Word-level timestamps, fixation durations, saccades
- **Access**: Publicly available upon request

**Advantages:**
- Naturalistic reading task
- Large vocabulary, diverse content
- Multi-modal (EEG + eye-tracking)
- Word-level ground truth timing

**Limitations:**
- Reading (receptive), not speech production
- Individual differences in reading strategies
- Some subjects have fewer trials due to fatigue

#### ZuCo 2.0 (2020)
**Task:** Normal reading + task-specific reading

**Additions to ZuCo 1.0:**
- **Subjects**: 18 subjects (12 from 1.0 + 6 new)
- **Tasks**:
  - **Task 1 (NR)**: Normal reading (~700 sentences)
  - **Task 2 (TSR)**: Task-specific reading with questions (~300 sentences)
  - **Task 3 (RD)**: Relation detection (~400 sentence pairs)
- **Total Sentences**: ~2,400 across all tasks
- **Annotations**: Sentiment, semantic relations, syntactic features

**Advantages:**
- Task diversity (different cognitive loads)
- Richer linguistic annotations
- Enables multi-task learning

**Recommended for NEST:** Use ZuCo 2.0 Task 1 (NR) as primary benchmark

### 2. Custom Imagined Speech Datasets

#### Characteristics
Most imagined speech studies use custom, non-public datasets:

**Typical Setup:**
- **Subjects**: 5-20
- **Vocabulary**: 5-100 words
- **Trials**: 20-100 per word per subject
- **Channels**: 14-64 EEG
- **Paradigm**: Visual/auditory cue → imagined articulation

**Examples:**
- **Cooney et al. (2020)**: 15 subjects, 10 words, 64-channel
- **Dash et al. (2020)**: 15 subjects, 11 words, 64-channel
- **Krishna et al. (2020)**: 10 subjects, 20 words, 14-channel

**Limitations:**
- Not publicly available → difficult to compare
- Small vocabulary → not representative of real SSI use
- High inter-study variability in protocols

**Need:** Standardized imagined speech benchmark (opportunity for NEST to contribute)

### 3. BCI Competition Datasets

#### BCI Competition IV (2008)
**Relevant Datasets:**
- **Dataset 2a**: Motor imagery (4 classes)
- **Dataset 2b**: Motor imagery (2 classes)

**Not directly applicable to SSI, but useful for:**
- Benchmarking preprocessing pipelines
- Transfer learning from motor imagery
- Validation of spatial filtering techniques

### 4. Other Cognitive EEG Datasets

#### EEG-based Emotion Recognition
- **DEAP**: 32 subjects, music-induced emotions
- **SEED**: 15 subjects, video-induced emotions

**Relevance:** 
- Similar preprocessing challenges
- Attention mechanisms tested on these datasets
- Not speech-related, limited direct applicability

#### Event-Related Potential (ERP) Datasets
- **P300 Speller Datasets**: Various BCI competitions
- **N400 Datasets**: Semantic incongruity studies

**Relevance:**
- Validate ERP detection capabilities
- Transfer learning for temporal feature extraction

### 5. Speech Datasets (for Transfer Learning)

While not EEG-based, speech corpora can pre-train components of NEST:

#### LibriSpeech
- **Size**: 1,000 hours of read English speech
- **Use**: Pre-train language model (decoder)

#### Common Voice (Mozilla)
- **Languages**: 100+ including Turkish
- **Size**: Thousands of hours
- **Use**: Cross-lingual language model pre-training

#### Turkish Datasets
- **CSS10**: Single speaker Turkish audiobook (~12 hours)
- **Mozilla Common Voice TR**: Crowdsourced Turkish speech

**Strategy:** Pre-train decoder on text/speech, then adapt to EEG encoder

### Benchmark Summary Table

| Dataset | Modality | Task | Subjects | Vocabulary | Public | NEST Priority |
|---------|----------|------|----------|------------|--------|---------------|
| ZuCo 1.0 | EEG+Eye | Reading | 12 | ~3,500 | Yes | High |
| ZuCo 2.0 | EEG+Eye | Reading+Tasks | 18 | ~4,000 | Yes | **Primary** |
| Custom Imagined | EEG | Imagined speech | 5-20 | 5-100 | No | Medium |
| BCI Competition | EEG | Motor imagery | 9 | N/A | Yes | Low |
| LibriSpeech | Audio | Speech | ~2,500 | ~200,000 | Yes | Medium (LM) |

## Evaluation Metrics

### 1. Sequence-Level Metrics (Primary for NEST)

#### Word Error Rate (WER)
**Definition:**
```
WER = (S + D + I) / N × 100%

Where:
S = Number of substitutions
D = Number of deletions
I = Number of insertions
N = Total number of words in reference
```

**Example:**
```
Reference:  "the cat sat on the mat"
Hypothesis: "the cat sit on mat"

S = 1 (sat → sit)
D = 1 (the is missing)
I = 0
N = 6

WER = (1 + 1 + 0) / 6 = 33.3%
```

**Characteristics:**
- **Standard in ASR and SSI**
- Can exceed 100% if insertions are excessive
- Lower is better (0% = perfect)

**Calculation Tools:**
- `jiwer` Python library
- `sclite` from NIST

**Limitations:**
- Treats all errors equally (doesn't distinguish semantic vs. syntactic)
- Sensitive to word order

#### Character Error Rate (CER)
**Definition:**
```
CER = (S_char + D_char + I_char) / N_char × 100%
```

**Use Cases:**
- More granular than WER
- **Critical for agglutinative languages like Turkish**
  - Example: "kitaplarımdan" (from my books) vs. "kitabımdan" (from my book)
  - WER: 100% (completely wrong word)
  - CER: ~20% (few character differences)

**Advantage for NEST:**
- Better reflects partial correctness in Turkish decoding

#### BLEU (Bilingual Evaluation Understudy)
**Definition:**
```
BLEU = BP × exp(Σ_n w_n log p_n)

Where:
p_n = n-gram precision
BP = Brevity penalty (penalizes short outputs)
w_n = Weight for n-gram (typically uniform)
```

**Typically:** BLEU-4 (considers 1-grams to 4-grams)

**Range:** 0 to 1 (higher is better)

**Characteristics:**
- **Standard in machine translation**
- Measures fluency and adequacy
- Multiple reference translations supported

**Use in NEST:**
- Complement to WER
- Assesses semantic preservation
- Useful for cross-lingual evaluation (English ↔ Turkish)

**Limitations:**
- Correlation with human judgment varies
- Not suitable for single-word outputs

#### METEOR (Metric for Evaluation of Translation with Explicit ORdering)
**Features:**
- Considers synonyms and paraphrases
- Aligns hypothesis to reference
- Balances precision and recall

**Advantage:** 
- Better correlation with human judgment than BLEU
- Handles morphological variants (useful for Turkish)

**Limitation:** 
- Requires language-specific resources (stemmer, synonym DB)

### 2. Token-Level Metrics

#### Token Accuracy
**Definition:**
```
Token Accuracy = Correct Tokens / Total Tokens × 100%
```

**Use:**
- Simpler than WER (no alignment needed if segmentation is known)
- Useful for debugging

**Limitation:**
- Doesn't account for sequence errors (deletions, insertions)

#### Perplexity
**Definition:**
```
Perplexity = exp(-1/N Σ log P(w_i | context))
```

**Use:**
- Measures language model quality
- Lower is better
- Useful for evaluating decoder component

**Not a primary metric for NEST** (focuses on generation, not LM alone)

### 3. Alignment Metrics

#### Alignment Error Rate (AER)
**Definition:**
Measures quality of temporal alignment between EEG and text

**Calculation:**
```
AER = (# Misaligned Words) / (# Total Words) × 100%
```

**Ground Truth:** 
- For ZuCo: Word-level fixation timestamps
- For imagined speech: Manually annotated (challenging)

**Use:**
- Diagnostic for transducer alignment learning
- Validate attention mechanism

#### Attention Entropy
**Definition:**
```
H(attention) = -Σ α_i log α_i
```

**Interpretation:**
- Low entropy: Focused attention (sharp peaks)
- High entropy: Diffuse attention (nearly uniform)

**Use:**
- Monitor attention behavior during training
- Identify if model is learning meaningful alignments

**Not a primary metric** (diagnostic tool)

### 4. BCI-Specific Metrics

#### Information Transfer Rate (ITR)
**Definition:**
```
ITR = [log_2(N) + P log_2(P) + (1-P) log_2((1-P)/(N-1))] × (60/T)

Where:
N = Number of possible symbols
P = Accuracy (0 to 1)
T = Time per selection (seconds)
```

**Units:** Bits per minute

**Characteristics:**
- **Standard in BCI literature**
- Balances speed and accuracy
- Higher is better

**Example:**
- N = 26 (alphabet), P = 0.8, T = 2s
- ITR ≈ 60 bits/min

**Use in NEST:**
- Compare with P300 spellers and other BCIs
- Assess practical utility

**Limitation:**
- Assumes discrete symbols (not continuous text generation)
- Approximation for NEST's sequence decoding

#### Real-Time Factor (RTF)
**Definition:**
```
RTF = Processing Time / Input Duration
```

**Interpretation:**
- RTF < 1: Faster than real-time (acceptable)
- RTF = 1: Just keeps up
- RTF > 1: Cannot process in real-time

**Example:**
- 10 seconds of EEG processed in 5 seconds → RTF = 0.5

**Use in NEST:**
- Requirement for deployment
- Target: RTF < 0.5 for responsive interaction

**Measurement:**
- Exclude data loading (only inference time)
- Include all steps (preprocessing, encoding, decoding)

### 5. Cross-Subject Generalization Metrics

#### Leave-One-Subject-Out (LOSO) Accuracy
**Protocol:**
1. Train on N-1 subjects
2. Test on held-out subject
3. Repeat for all subjects
4. Average performance

**Importance:**
- **Critical for practical BCI deployment**
- Most studies report subject-dependent results (overly optimistic)
- NEST should prioritize LOSO evaluation

#### Domain Adaptation Performance
**Metrics:**
- **Source-only**: Train on source subjects, test on target (no adaptation)
- **Target-supervised**: Fine-tune on target subject data
- **Target-unsupervised**: Adapt without target labels (domain adaptation)

**Evaluation:**
```
Adaptation Gain = Performance_adapted - Performance_source-only
```

### 6. Linguistic Metrics

#### Part-of-Speech (POS) Accuracy
**Definition:**
Percentage of words with correct POS tags

**Use:**
- Assess syntactic correctness
- Particularly relevant for Turkish (rich morphology)

**Calculation:**
- Requires POS tagger (spaCy, NLTK)

#### Semantic Similarity
**Metrics:**
- **Cosine Similarity**: Between sentence embeddings (BERT, Sentence-BERT)
- **BERTScore**: Token-level similarity using contextualized embeddings

**Advantage:**
- Captures semantic preservation even with different wording
- Useful when multiple valid paraphrases exist

**Example:**
```
Reference:  "The cat is sleeping on the mat"
Hypothesis: "A cat sleeps on a rug"

WER: ~60% (many substitutions)
Semantic Similarity: ~0.85 (high, meaning preserved)
```

### 7. Statistical Significance Testing

#### Paired t-test
**Use:** Compare two models on same test set
**Null Hypothesis:** No difference in mean performance

#### Wilcoxon Signed-Rank Test
**Use:** Non-parametric alternative to t-test
**Advantage:** Doesn't assume normal distribution

#### McNemar's Test
**Use:** Compare binary outcomes (correct/incorrect per sample)
**Advantage:** More powerful for classification

#### Effect Size
**Cohen's d:**
```
d = (Mean_1 - Mean_2) / Pooled_SD
```

**Interpretation:**
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect

**Importance:** Statistical significance ≠ practical significance

#### Confidence Intervals
**Report:** 95% CI for all metrics
**Example:** WER = 35% ± 3% (95% CI: [32%, 38%])

### 8. Qualitative Evaluation

#### Human Evaluation
**Metrics:**
- **Intelligibility**: Can humans understand the decoded text?
- **Fluency**: Does it sound natural?
- **Adequacy**: Is the meaning preserved?

**Protocol:**
- Present decoded outputs to raters
- 5-point Likert scale
- Multiple raters (inter-rater reliability)

**Use:**
- Final validation before deployment
- Complement quantitative metrics

#### Error Analysis
**Categories:**
- **Substitution Errors**: By phonetic similarity, semantic similarity
- **Deletion Errors**: Function words vs. content words
- **Insertion Errors**: Common vs. rare words

**Insights:**
- Identify systematic weaknesses
- Guide model improvements

## Evaluation Protocols

### 1. Training/Validation/Test Splits

#### Subject-Independent (Recommended)
**Protocol:**
- **Training**: Subjects 1-14 (ZuCo 2.0)
- **Validation**: Subjects 15-16 (hyperparameter tuning)
- **Test**: Subjects 17-18 (held-out, reported once)

**Advantage:**
- Measures generalization to new users
- More realistic for deployment

#### Subject-Dependent (Baseline)
**Protocol:**
- **Training**: 70% of each subject's data
- **Validation**: 15% of each subject's data
- **Test**: 15% of each subject's data

**Advantage:**
- Upper bound on performance
- Useful for ablation studies

**Report both:** Subject-dependent (optimistic) and subject-independent (realistic)

#### Cross-Validation
**k-Fold Cross-Validation:**
- k = 5 or 10
- Report mean and standard deviation

**Leave-One-Subject-Out (LOSO):**
- Special case of k-fold where k = number of subjects
- Maximizes training data per fold

### 2. Hyperparameter Tuning

**Protocol:**
- **Only use validation set** for tuning
- **Never touch test set** until final evaluation
- Report hyperparameter search space

**Best Practices:**
- Grid search for small spaces
- Random search or Bayesian optimization for large spaces
- Document all hyperparameter choices

### 3. Ablation Studies

**Purpose:** Quantify contribution of each component

**Example Ablations for NEST:**
1. **Encoder Variants:**
   - RNN vs. Transformer vs. Conformer
2. **Attention Mechanisms:**
   - No attention vs. cross-attention vs. full multi-head
3. **Preprocessing:**
   - Raw EEG vs. filtered vs. ICA-cleaned
4. **Training Strategies:**
   - From scratch vs. pre-trained vs. transfer learning
5. **Loss Functions:**
   - Transducer loss only vs. + CTC vs. + attention supervision

**Reporting:**
- Table with each ablation and resulting WER/BLEU
- Identify critical components vs. marginal gains

### 4. Baseline Comparisons

**Mandatory Baselines:**
1. **Random Baseline**: Random word selection (sanity check)
2. **Frequency Baseline**: Most frequent word (checks dataset bias)
3. **Prior Work**: Reproduce published results on ZuCo (e.g., Sun et al., 2021)

**Competitive Baselines:**
1. **LSTM Seq2seq**: Standard encoder-decoder with attention
2. **Transformer**: Standard Transformer (no transducer)
3. **RNN-T**: RNN Transducer (if Transformer-T is used)

**Reporting:**
- All baselines on same test set
- Same preprocessing pipeline
- Statistical significance tests

### 5. Robustness Evaluation

#### Noise Robustness
**Protocol:**
- Add synthetic noise (Gaussian, pink noise)
- Vary SNR: -5dB to +10dB
- Measure performance degradation

#### Channel Dropout
**Protocol:**
- Randomly remove 10%, 20%, 30% of channels
- Simulates bad electrodes
- Test generalization

#### Temporal Shift
**Protocol:**
- Shift EEG-text alignment by ±100ms, ±200ms
- Test alignment robustness

### 6. Computational Efficiency

**Metrics to Report:**
- **Model Size**: Number of parameters (millions)
- **FLOPs**: Floating-point operations per sample
- **Inference Time**: Per sample (CPU and GPU)
- **Memory**: Peak GPU memory during inference

**Benchmarking:**
- Standardize hardware (e.g., NVIDIA V100, Intel Xeon)
- Report batch size
- Measure on test set (not training)

## Comparative Analysis Framework

### State-of-the-Art (SOTA) Table

**Format:**
| Model | Dataset | Subjects | WER (%) | BLEU | RTF | Subject-Indep? |
|-------|---------|----------|---------|------|-----|----------------|
| Sun et al. (2021) | ZuCo 1.0 | 12 | 45 | 0.38 | 1.2 | No |
| **NEST (RNN-T)** | ZuCo 2.0 | 18 | **35** | **0.52** | **0.6** | **Yes** |
| **NEST (Conf-T)** | ZuCo 2.0 | 18 | **32** | **0.56** | **0.8** | **Yes** |

**Include:**
- Dataset version (ZuCo 1.0 vs. 2.0 are different)
- Subject-dependent vs. subject-independent
- Comparable preprocessing

### Learning Curves

**Plot:**
- Training data size (x-axis) vs. Performance (y-axis)
- Shows data efficiency

**Insights:**
- Identify data bottlenecks
- Guide future data collection efforts

### Error Analysis by Linguistic Features

**Categories:**
1. **Word Frequency**: Rare vs. common words
2. **Word Length**: Short vs. long words
3. **POS**: Nouns, verbs, function words
4. **Sentence Length**: Short vs. long sentences

**Visualization:**
- Grouped bar charts
- Identify where model struggles

## Reproducibility Checklist

To ensure reproducible research, NEST should report:

### 1. Data
- [ ] Dataset name and version
- [ ] Train/val/test split details
- [ ] Preprocessing steps (with code)
- [ ] Augmentation techniques

### 2. Model
- [ ] Architecture diagram
- [ ] All hyperparameters
- [ ] Initialization strategy
- [ ] Loss function details

### 3. Training
- [ ] Optimizer and learning rate schedule
- [ ] Batch size
- [ ] Number of epochs
- [ ] Early stopping criteria
- [ ] Hardware (GPU model, RAM)

### 4. Evaluation
- [ ] Metrics used
- [ ] Statistical tests
- [ ] Confidence intervals
- [ ] Random seeds

### 5. Code
- [ ] Public GitHub repository
- [ ] Requirements.txt or environment.yml
- [ ] Trained model checkpoints (if permissible)
- [ ] Inference scripts

## Recommended Evaluation Plan for NEST

### Phase 1: Baseline Establishment
1. Reproduce Sun et al. (2021) on ZuCo 2.0
2. Establish LSTM seq2seq baseline
3. Verify preprocessing pipeline

**Metrics:** WER, BLEU, CER
**Split:** Subject-dependent (80/10/10)

### Phase 2: NEST Model Development
1. Implement RNN-T variant
2. Implement Transformer-T variant
3. Implement Conformer-T variant

**Metrics:** WER, BLEU, CER, RTF
**Split:** Subject-dependent and subject-independent

### Phase 3: Ablation Studies
1. Encoder ablations (RNN/Transformer/Conformer)
2. Attention ablations
3. Preprocessing ablations

**Metrics:** WER (primary)
**Analysis:** Statistical significance tests

### Phase 4: Cross-Lingual Evaluation
1. English-only model
2. Turkish-only model
3. Bilingual model

**Metrics:** WER, CER (especially for Turkish), BLEU
**Dataset:** ZuCo (English), future Turkish dataset

### Phase 5: Real-Time Feasibility
1. Model compression (quantization, pruning)
2. Latency optimization
3. Streaming evaluation

**Metrics:** RTF, WER (under streaming constraints)

### Phase 6: User Study
1. Recruit participants (N=10-20)
2. Collect new data with NEST system
3. Qualitative evaluation

**Metrics:** ITR, NASA-TLX, SUS, qualitative feedback

## Key References

### Evaluation Metrics
1. **Papineni, K., et al. (2002).** "BLEU: a Method for Automatic Evaluation of Machine Translation." ACL.

2. **Levenshtein, V. I. (1966).** "Binary Codes Capable of Correcting Deletions, Insertions, and Reversals." Soviet Physics Doklady.

3. **Banerjee, S., & Lavie, A. (2005).** "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments." ACL Workshop.

### BCI Metrics
4. **Wolpaw, J. R., et al. (2000).** "Brain-Computer Interface Technology: A Review of the First International Meeting." IEEE Transactions on Rehabilitation Engineering.

5. **Thompson, D. E., et al. (2014).** "Performance Measurement for Brain-Computer or Brain-Machine Interfaces: A Tutorial." Journal of Neural Engineering.

### Statistical Methods
6. **Demšar, J. (2006).** "Statistical Comparisons of Classifiers over Multiple Data Sets." JMLR.

7. **Dietterich, T. G. (1998).** "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." Neural Computation.

### Reproducibility
8. **Pineau, J., et al. (2021).** "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)." JMLR.

9. **Bouthillier, X., et al. (2021).** "Accounting for Variance in Machine Learning Benchmarks." MLSys.

### ZuCo Dataset
10. **Hollenstein, N., et al. (2018).** "ZuCo, a Simultaneous EEG and Eye-Tracking Resource for Natural Sentence Reading." Scientific Data.

11. **Hollenstein, N., et al. (2020).** "ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation." LREC.

### Comparative EEG Studies
12. **Sun, Y., et al. (2021).** "Brain2Word: Decoding Brain Activity for Language Generation." arXiv.

13. **Lotte, F., et al. (2018).** "A Review of Classification Algorithms for EEG-based Brain-Computer Interfaces: A 10-year Update." Journal of Neural Engineering.

## Summary of Recommended Metrics for NEST

### Primary Metrics (Must Report)
1. **Word Error Rate (WER)**: Compare with ASR and BCI literature
2. **Character Error Rate (CER)**: Critical for Turkish evaluation
3. **BLEU Score**: Semantic adequacy assessment

### Secondary Metrics (Should Report)
4. **Real-Time Factor (RTF)**: Deployment feasibility
5. **Information Transfer Rate (ITR)**: BCI comparison
6. **Subject-Independent vs. Dependent**: Generalization

### Diagnostic Metrics (Internal Use)
7. **Attention Entropy**: Alignment learning
8. **Perplexity**: Language model quality
9. **Alignment Error Rate**: Temporal accuracy

### Qualitative (User Study)
10. **Intelligibility, Fluency, Adequacy**: Human judgment
11. **NASA-TLX**: Cognitive load
12. **Error Analysis**: Systematic patterns

## Conclusion

Rigorous evaluation on standardized benchmarks (ZuCo 2.0) with comprehensive metrics (WER, CER, BLEU) is essential for advancing EEG-to-text decoding. NEST should prioritize subject-independent evaluation, statistical significance testing, and reproducible research practices. Comparisons with both traditional BCI systems (ITR) and modern NLP systems (BLEU) provide context for assessing progress toward practical silent speech interfaces.

---
**Last Updated:** February 2026
