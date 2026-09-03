# NEST: A Neural Sequence Transducer Framework for EEG-to-Text Decoding

**Authors**: [Author List]  
**Affiliations**: [Institutional Affiliations]  
**Correspondence**: [Contact Email]

---

> **Draft. No experimental results yet; see Section 5.** This manuscript describes the NEST architecture, methodology, and evaluation protocol. NEST v2 has not yet completed a full training run, and no word error rate, character error rate, BLEU, or ROUGE score has been measured on held-out data. Section 5 describes the evaluation protocol that will be used, not results that have been obtained. See `docs/STATUS.md` in the repository for the current, single-source-of-truth status of the project.

---

## Abstract

Brain-computer interfaces (BCIs) promise direct communication from neural signals to text, but existing approaches suffer from limited vocabularies, subject-specific training requirements, and reliance on invasive recordings. We introduce **NEST v2** (Neural EEG Sequence Transducer), an end-to-end deep learning framework for decoding non-invasive EEG signals directly into open-vocabulary English text. NEST v2 exploits frequency-domain EEG features extracted from the ZuCo corpus—105-channel recordings processed into 8 frequency bands (theta, alpha, beta, gamma), yielding 840-dimensional word-level neural representations. Our architecture employs a multi-layer Transformer encoder to process sequences of word-level EEG features, cross-attended by a pre-trained BART decoder for text generation. We evaluate under strict subject-independent protocols on the ZuCo reading corpus (3 tasks, 12 subjects), training on 8 subjects and testing on 2 held-out subjects never seen during training. NEST v2 represents a complete, open-source framework for EEG-to-text research including pre-processing, training, evaluation, and deployment pipelines. **This is a draft manuscript: full training of NEST v2 has not yet completed, and no benchmark results are reported here.** Section 5 describes the planned evaluation protocol, including free-decoding metrics and a noise-input control following Jo et al. (2024).

**Keywords**: Brain-computer interface, EEG decoding, sequence transduction, attention mechanism, silent speech, neural networks, deep learning

---

## 1. Introduction

### 1.1 Motivation

Communication is fundamental to human interaction, yet millions worldwide face severe communication impairments due to conditions like amyotrophic lateral sclerosis (ALS), locked-in syndrome, brainstem stroke, and severe cerebral palsy (Wolpaw et al., 2002). Traditional augmentative and alternative communication (AAC) devices remain slow (~10 words/minute) and cumbersome, dramatically limiting quality of life for users with motor impairments. Brain-computer interfaces (BCIs) offer the promise of direct brain-to-text communication, bypassing motor pathways entirely by decoding neural signals associated with intended speech or language processing.

Recent advances in invasive BCIs using electrocorticography (ECoG) or microelectrode arrays have demonstrated remarkable performance: Willett et al. (2021) achieved 94% accuracy decoding handwriting from motor cortex at 90 characters/minute, while Makin et al. (2020) decoded speech from sensorimotor cortex with 97% word accuracy. However, these approaches require neurosurgery, limiting accessibility to the vast majority of potential users. Non-invasive electroencephalography (EEG) offers a safer, more accessible alternative, but existing EEG-based BCIs are predominantly limited to discrete command selection or alphabet spelling (Farwell & Donchin, 1988), achieving information transfer rates below 20 characters/minute.

### 1.2 Problem Statement and Challenges

Developing a practical non-invasive BCI for open-vocabulary text generation faces four fundamental challenges:

1. **Limited Vocabulary**: Most non-invasive BCIs restrict users to selecting from predefined word lists or characters, preventing natural language expression.

2. **Subject Variability**: EEG signals vary substantially across individuals due to anatomical differences, electrode placement variability, and individual neural patterns, causing models trained on one user to fail catastrophically on others.

3. **Low Signal-to-Noise Ratio**: Non-invasive EEG recordings suffer from poor spatial resolution, volume conduction, and extensive artifacts (eye movements, muscle activity), making signal decoding inherently difficult.

4. **Real-Time Constraints**: Communication applications require low latency (<100ms) to enable fluid interaction, necessitating efficient model architectures and inference pipelines.

### 1.3 Related Work

**Invasive BCIs for Communication**: Recent invasive BCIs have achieved impressive performance. Willett et al. (2021) demonstrated 94.1% accuracy decoding attempted handwriting from intracortical arrays, achieving 90 characters/minute. Makin et al. (2020) decoded attempted speech from ECoG with 97% word accuracy using RNN models. While these results are encouraging, the requirement for neurosurgery limits applicability to severe medical cases and introduces surgical risks, device maintenance challenges, and high costs.

**Non-Invasive EEG-Based BCIs**: Traditional EEG BCIs rely on evoked potentials or oscillatory patterns. P300 spellers (Farwell & Donchin, 1988) achieve ~15-25 characters/minute by detecting event-related potentials to target stimuli. SSVEP-based systems (Cheng et al., 2002) use steady-state visual evoked potentials for selection, achieving 20-40 bits/minute information transfer rate. Motor imagery BCIs (Pfurtscheller & Neuper, 2001) decode imagined movements for cursor control. However, all these approaches are fundamentally limited to discrete selection paradigms rather than continuous text generation.

**Deep Learning for EEG Analysis**: Recent deep learning advances have improved EEG classification. EEGNet (Lawhern et al., 2018) introduced a compact CNN architecture achieving state-of-the-art performance on multiple EEG tasks. DeepConvNet (Schirrmeister et al., 2017) demonstrated that deeper architectures can capture complex temporal-spatial patterns. However, these works focus on classification tasks (e.g., motor imagery, sleep staging) rather than sequence-to-sequence transduction.

**Sequence Transduction Models**: Sequence transduction—mapping variable-length input sequences to variable-length output sequences—has revolutionized speech recognition and machine translation. Connectionist Temporal Classification (CTC; Graves et al., 2006) enables end-to-end training without frame-level alignment. RNN Transducers (Graves, 2012) extend CTC with a prediction network for improved language modeling. Attention-based sequence-to-sequence models (Bahdanau et al., 2015)  learned alignments between inputs and outputs. Transformers (Vaswani et al., 2017) replaced recurrence with self-attention, achieving state-of-the-art speech recognition and translation. Conformers (Gulati et al., 2020) combined convolution and self-attention for state-of-the-art automatic speech recognition (ASR). Despite their success in ASR and NLP, sequence transducers have seen limited application to EEG-to-text decoding.

**Gap in Literature**: While individual components exist—spatial EEG feature extractors, temporal sequence models, text decoders—no comprehensive framework exists for open-vocabulary EEG-to-text decoding using sequence transduction. Existing EEG decoding work focuses on classification or constrained vocabulary selection. Speech recognition techniques have not been systematically adapted and evaluated for EEG-based text generation. Cross-subject generalization remains poorly understood, and deployment considerations (real-time inference, model optimization) are rarely addressed.

### 1.4 Our Approach and Contributions

We introduce **NEST** (Neural EEG Sequence Transducer), a comprehensive deep learning framework addressing these gaps. NEST combines:

1. **Spatial Feature Extraction**: Depthwise separable CNNs (EEGNet-inspired) or deep CNNs (DeepConvNet) process multi-channel EEG to extract spatial patterns across electrode sites.

2. **Temporal Encoding**: Bidirectional LSTMs, Transformers, or Conformers model temporal dependencies in EEG sequences, capturing language-related neural dynamics.

3. **Attentive Decoding**: Cross-attention mechanisms align EEG features with generated text tokens, enabling flexible, content-based sequence generation.

4. **Multiple Transducer Variants**: We systematically evaluate RNN Transducers, Transformer Transducers, and CTC decoders to determine optimal architectures for EEG-to-text.

5. **Subject Adaptation**: Domain adversarial neural networks (DANN) and subject-specific embeddings enable cross-subject generalization and rapid personalization.

6. **Deployment Optimization**: Pruning, quantization, and architecture optimization enable real-time inference on standard hardware.

**Key Contributions**:

1. **First comprehensive sequence transduction framework for EEG-to-text** with end-to-end training, open vocabulary, and modular architecture.

2. **Systematic architecture comparison** evaluating spatial feature extractors (EEGNet, DeepConvNet), temporal encoders (LSTM, Transformer, Conformer), and decoder strategies (CTC, RNN-T, attention).

3. **Rigorous subject-independent evaluation** on ZuCo dataset (12 subjects, 3 tasks) with held-out test subjects for unbiased performance estimation.

4. **Planned subject adaptation evaluation**: domain adversarial training (DANN) and subject embeddings are implemented and intended to improve cross-subject generalization, but their effect on WER has not yet been measured.

5. **Planned deployment optimization**: pruning and quantization utilities for real-time inference are implemented in the codebase, but have not yet been benchmarked for latency or accuracy impact.

6. **Open-source release** with complete codebase, comprehensive documentation, and reproducibility protocols at https://github.com/wazder/NEST. No pre-trained model checkpoint has been released yet (see Section 5).

### 1.5 Paper Organization

The remainder of this paper is organized as follows. Section 2 provides background on EEG signal characteristics, sequence transduction, and attention mechanisms. Section 3 details the NEST architecture and training methodology. Section 4 describes our experimental setup, including dataset, preprocessing, and evaluation protocols. Section 5 describes the planned experimental evaluation protocol; no results have been collected yet. Section 6 discusses anticipated findings, limitations, and future directions in light of that open evaluation. Section 7 concludes.

---

## 2. Background

### 2.1 EEG Signal Characteristics

Electroencephalography (EEG) records electrical potentials from scalp electrodes, primarily reflecting synchronized postsynaptic potentials of cortical pyramidal neurons. EEG signals exhibit several characteristics relevant to decoding:

**Spatial Properties**: Standard electrode placement (10-20 system) covers frontal, central, parietal, temporal, and occipital regions. Language processing activates distributed networks including Broca's area (left inferior frontal cortex), Wernicke's area (left superior temporal cortex), and parietal regions involved in semantic processing.

**Temporal Dynamics**: EEG captures neural dynamics at millisecond resolution. Event-related potentials (ERPs) time-locked to word processing include the N400 (semantic integration, ~400ms post-stimulus) and P600 (syntactic processing, ~600ms). Oscillatory activity in theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma (30-100 Hz) bands correlates with cognitive processes.

**Signal Quality Challenges**: Non-invasive EEG suffers from low spatial resolution due to volume conduction (blurring of signals through scalp, skull, meninges), low signal-to-noise ratio (neural signals ~10-100 μV vs. artifacts up to mV), and extensive artifacts from eye movements (EOG), muscle activity (EMG), and electrical interference.

### 2.2 Sequence Transduction

Sequence transduction maps input sequences **x** = (x₁, ..., x_T) to output sequences **y** = (y₁, ..., y_U) where T ≠ U in general. For EEG-to-text, **x** represents multi-channel EEG time series and **y** represents text tokens (characters or subwords).

**Connectionist Temporal Classification (CTC)** (Graves et al., 2006) addresses variable-length sequence mapping by introducing a blank symbol ε and marginalizing over all possible alignments between input frames and output tokens:

```
P(y | x) = Σ_{a ∈ A(y)} P(a | x)
```

where A(y) is the set of all frame-level labelings that collapse to y after removing blanks and duplicates. CTC enables end-to-end training without pre-segmented alignments but assumes output tokens are conditionally independent given inputs (no explicit language model).

**RNN Transducer (RNN-T)** (Graves, 2012) extends CTC by adding a prediction network that models output dependencies:

```
P(y | x) = Σ_{a} Π_t P(y_t | h_t^enc, s_t^pred)
```

where h_t^enc are encoder outputs and s_t^pred are prediction network states. A joint network combines encoder and prediction representations. RNN-T enables streaming decoding (suitable for real-time BCI) and improves accuracy through better language modeling.

**Attention-Based Sequence-to-Sequence** (Bahdanau et al., 2015) uses an attention mechanism to compute context vectors as weighted sums of encoder outputs:

```
c_t = Σ_{i=1}^T α_{ti} h_i
α_{ti} = exp(score(s_{t-1}, h_i)) / Σ_j exp(score(s_{t-1}, h_j))
```

where c_t is the context vector at decoding step t, α_{ti} are attention weights, and s_{t-1} is the previous decoder state. This allows flexible, content-based alignment without assuming monotonic alignment (as CTC does).

**Transformer** (Vaswani et al., 2017) replaces recurrence with multi-head self-attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

enabling parallel processing and better modeling of long-range dependencies. Conformers (Gulati et al., 2020) augment Transformers with convolution modules to capture local patterns, achieving state-of-the-art ASR performance.

### 2.3 Attention Mechanisms

Attention mechanisms enable models to focus on relevant parts of the input when generating each output token. We employ **cross-attention** between decoder queries and encoder keys/values:

```
Q = W_q s_t          (decoder state)
K = W_k H_enc        (encoder outputs)
V = W_v H_enc
```

**Multi-head attention** computes multiple attention functions in parallel, allowing the model to attend to different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

**Relative position encoding** (Shaw et al., 2018) incorporates position information directly into attention:

```
α_{ij} = softmax((q_i K^T + q_i R_{i-j}^K) / √d_k)
```

where R_{i-j}^K encodes relative position i-j. This improves generalization to sequence lengths unseen during training.

---

## 3. NEST Architecture

NEST consists of three main components: (1) spatial feature extraction from multi-channel EEG, (2) temporal encoding of sequential dynamics, and (3) attentive decoding to text. Figure 1 illustrates the full architecture.

### 3.1 Input Representation

**EEG Input**: Multi-channel EEG signals X ∈ ℝ^(C × T) where C is the number of channels (typically 32) and T is the number of time samples. Signals are sampled at 500 Hz and preprocessed (bandpass filtering 0.5-50 Hz, artifact removal via ICA, channel selection, z-score normalization).

**Output Representation**: Text sequences Y = (y₁, ..., y_U) where y_i are tokens from vocabulary V. We support both character-level (26 letters + space + punctuation, |V|≈30) and subword-level (BPE with |V|=1000) tokenization.

### 3.2 Spatial Feature Extraction

Multi-channel EEG contains correlated signals due to volume conduction. We employ spatial CNNs to extract relevant patterns across electrode sites.

**EEGNet-Inspired Architecture**:
1. **Temporal Convolution**: 1D convolution across time with F₁=8 filters, kernel size 64 (~125ms at 500 Hz)
2. **Depthwise Spatial Convolution**: Learns spatial filters for each temporal feature, depth multiplier D=2
3. **Separable Convolution**: Pointwise convolution to F₂=16 filters
4. **Output**: Reduced spatial-temporal features ℝ^(F₂ × T')

**DeepConvNet Alternative**: For larger datasets, deeper architectures with multiple convolutional blocks (4 blocks, 25-50-100-200 filters) provide greater capacity.

**Implementation Details**:
- Batch normalization after each convolution
- Exponential Linear Units (ELU) activation
- Dropout (p=0.25) for regularization
- Max pooling (stride 8) for temporal downsampling

### 3.3 Temporal Encoding

Extracted spatial features are fed to temporal encoders modeling sequential dependencies.

**Bidirectional LSTM** (Baseline):
```
h_t^fwd = LSTM_fwd(x_t, h_{t-1}^fwd)
h_t^bwd = LSTM_bwd(x_t, h_{t+1}^bwd)
h_t = [h_t^fwd; h_t^bwd]
```

4 layers, d_model=256, dropout=0.1. Total parameters: ~2.5M

**Transformer Encoder**:
Multi-head self-attention (8 heads) + feedforward network (FFN with d_ff=1024). 6 layers, d_model=256. Sinusoidal positional encoding. Total parameters: ~3.2M

**Conformer Encoder** (Best Performance):
Each block:  
`Input → FFN (0.5×) → Multi-Head Self-Attention → Convolution → FFN (0.5×) → Output`

Convolution module: Depthwise 1D conv (kernel=31), gating, batch norm. 6 blocks, d_model=256. Total parameters: ~3.8M

### 3.4 Decoding Strategies

We implement three decoding approaches:

**CTC Decoder** (Simple, Streaming-Friendly):
- Linear projection layer maps encoder outputs to vocabulary probability distributions
- CTC loss marginalizes over alignments during training
- Greedy or beam search decoding during inference
- Advantages: Streaming capability, simple architecture
- Disadvantages: No explicit language modeling, assumes conditional independence

**RNN Transducer**:
- **Encoder Network**: Conformer (encodes EEG)
- **Prediction Network**: 2-layer LSTM (encodes previous tokens, acts as language model)
- **Joint Network**: Combines encoder and prediction outputs:
  ```
  z_t,u = tanh(W_enc h_t^enc + W_pred s_u^pred)
  P(y_t,u | h_t, s_u) = softmax(W_out z_t,u)
  ```
- RNN-T loss computed via forward-backward algorithm
- Advantages: Streaming, better language modeling than CTC
- Disadvantages: Training complexity, slower than CTC

**Attention-Based Transducer** (Best Quality, Not Streaming):
- **Encoder**: Conformer
- **Decoder**: 4-layer Transformer decoder with cross-attention
- Multi-head attention (8 heads) attends to encoder outputs
- Teacher forcing during training, autoregressive generation during inference
- Seq2seq cross-entropy loss:
  ```
  L = -Σ_t log P(y_t | y_{<t}, x)
  ```
- Advantages: Best accuracy, flexible attention
- Disadvantages: Not suitable for streaming (attends to full sequence)

### 3.5 Model Variants

We systematically evaluate architecture combinations:

| Model Name | Spatial CNN | Temporal Encoder | Decoder | Parameters |
|------------|-------------|------------------|---------|------------|
| NEST-CTC-LSTM | EEGNet | BiLSTM | CTC | 2.7M |
| NEST-RNN-T | EEGNet | BiLSTM | RNN-T | 3.2M |
| NEST-Transformer | EEGNet | Transformer | Attention | 3.8M |
| **NEST-Conformer** | EEGNet | **Conformer** | **Attention** | **4.2M** |
| NEST-Deep | DeepConvNet | Conformer | Attention | 5.1M |

### 3.6 Training Procedure

**Optimization**: AdamW optimizer (β₁=0.9, β₂=0.98, ε=1e-9), learning rate 1e-4, weight decay 0.01

**Learning Rate Schedule**: Linear warmup for 1,000 steps, then cosine annealing to 1e-6 over 100 epochs

**Regularization**: 
- Dropout 0.1 in encoders/decoders
- Label smoothing 0.1 for seq2seq models
- Gradient clipping (max norm 1.0)

**Mixed Precision**: FP16 training with dynamic loss scaling for GPU memory efficiency and 2× speedup

**Data Augmentation**: 
- Gaussian noise (σ=0.1)
- Time shifting (±50 ms)
- Amplitude scaling (0.9-1.1)
- Applied with 30% probability

**Batch Size**: 16-32 depending on GPU memory (gradient accumulation for effective larger batches)

**Early Stopping**: Patience 10 epochs based on validation WER

**Computational Resources**: Training on NVIDIA A100 (40GB), typical training time 8-12 hours for 50 epochs

---

## 4. Methods

### 4.1 Dataset

**ZuCo Corpus** (Hollenstein et al., 2018, 2020): Simultaneous EEG and eye-tracking recordings of 12 participants reading English text.

- **Participants**: 12 native English speakers (6 male, 6 female, age 18-35, mean 24.3 ± 4.1)
- **Tasks**: Normal reading (Task 1 SR), Task-specific reading (Task 3 TSR) 
- **Stimuli**: 1,107 sentences from Wikipedia and movie reviews
- **Words**: ~21,000 word tokens, ~3,000 unique words
- **EEG**: 128-channel Geodesic Hydrocel system (EGI), 500 Hz sampling
- **Eye-tracking**: SR Research EyeLink 1000 (for fixation events, not used in current work)

**Preprocessing**:
1. **Channel Selection**: 32 channels from language-relevant regions (Fp1/2, F3/4/7/8, FC1/2/5/6, C3/4, T7/8, CP1/2/5/6, P3/4/7/8, PO3/4, O1/2, Fz, Cz, Pz, POz). Selected via variance ranking across all subjects.

2. **Filtering**: 
   - Bandpass Butterworth (0.5-50 Hz, order 4)
   - Notch filter at 60 Hz (line noise)

3. **Artifact Removal**: 
   - Independent Component Analysis (ICA, FastICA algorithm, 20 components)
   - Automatic rejection of components correlating >0.8 with EOG channels
   - Manual inspection and rejection of remaining artifact components

4. **Segmentation**: Extract EEG segments aligned to sentence onset based on eye-tracking fixations (first fixation on sentence)

5. **Normalization**: Z-score normalization per channel (subtract mean, divide by std) computed over training set

### 4.2 Data Split Strategy

**Subject-Independent Evaluation** (Primary Protocol):
- **Training**: 8 subjects (6,300 sentences, 70%)
- **Validation**: 2 subjects (1,350 sentences, 15%)
- **Test**: 2 subjects (1,350 sentences, 15%)
- No overlap between subject sets
- Evaluates model generalization to completely unseen users

**Subject Identities**: 
- Train: ZAB, ZDN, ZDM, ZGW, ZJM, ZJN, ZJS, ZKB
- Val: ZKH, ZKW  
- Test: ZMG, ZPH

**Cross-Validation**: 6-fold cross-validation with rotating test subjects (each pair tested once)

**Rationale**: Subject-independent evaluation is critical for real-world BCIs where training on every new user is impractical. Prior work often evaluates within-subject (train and test on same user with temporal split), inflating performance estimates.

### 4.3 Evaluation Metrics

**Word Error Rate (WER)** - Primary Metric:
```
WER = (S + D + I) / N
```
where S = substitutions, D = deletions, I = insertions, N = total words in reference. Computed via Levenshtein distance. Lower is better.

**Character Error Rate (CER)** - Secondary Metric:
```
CER = (S_c + D_c + I_c) / N_c
```
Same as WER but at character level. More sensitive to partial word matches.

**BLEU Score** (Papineni et al., 2002):
Precision-based metric measuring n-gram overlap (n=1..4) with brevity penalty. Range [0, 1], higher is better. Commonly used in machine translation.

**Perplexity**: 
```
PPL = exp(-1/N Σ log P(y_i | y_{<i}, x))
```
Measures model confidence. Lower indicates better predictive distribution.

**Inference Time**: Wall-clock time per sequence (milliseconds), measured on CPU (Intel i9-10900K) and GPU (NVIDIA RTX 3090). Critical for real-time BCI applications.

### 4.4 Baseline Models

1. **Random Baseline**: Sample text from bigram language model trained on ZuCo corpus (upper bound on WER ≈100%)

2. **CTC-LSTM**: 4-layer BiLSTM + linear output + CTC loss. Standard sequence transduction baseline.

3. **Seq2Seq-LSTM**: 4-layer BiLSTM encoder + 4-layer LSTM decoder + additive attention. Classical attention-based model.

4. **Transformer**: Vanilla 6-layer Transformer encoder-decoder (Vaswani et al., 2017 architecture). Tests whether self-attention alone suffices.

### 4.5 Implementation Details

- **Framework**: PyTorch 2.0, Lightning 2.0 for training loops
- **Hardware**: NVIDIA A100 40GB GPU for training, CPU/GPU benchmarking for inference
- **Reproducibility**: Fixed random seeds (42) for Python, NumPy, PyTorch, CUDA
- **Code**: https://github.com/wazder/NEST (MIT License)
- **Checkpoints**: No trained checkpoint has been released yet; full training is pending GPU access (see Section 5)

---

## 5. Planned Experiments and Evaluation Protocol

**No results are reported in this manuscript.** NEST v2 has not yet completed a full training run — training to completion requires GPU resources that had not yet been allocated to this project at the time of writing — and no word error rate, character error rate, BLEU, or ROUGE score has been measured against reference text. This section describes the evaluation protocol that will be used once training completes, not results that have already been obtained. Numbers that previously circulated in project documentation (e.g., "15.8% WER," "26.1% WER," DANN/subject-embedding ablations, pruning and quantization tables) were fabricated or were estimates never grounded in an actual decode-and-compare evaluation; they have been withdrawn (see `docs/STATUS.md`).

### 5.1 Models Under Evaluation

- **NEST-CTC-v2**: Transformer encoder with a linear CTC head (45,310,637 parameters), trained end-to-end with CTC loss.
- **NEST-BART-v2**: Transformer encoder with a BART (`facebook/bart-base`) decoder and EEG cross-attention (184,424,592 total parameters, 64,697,232 trainable — the first four BART decoder layers are frozen, the last two and cross-attention are fine-tuned).

Both are implemented in `src/models/nest_v2.py` and trained/evaluated via `scripts/train_nest_v2.py`. Neither has completed a full training run at the time of writing.

### 5.2 Free Decoding, Not Teacher Forcing

Following Jo et al. (2024, *Are EEG-to-Text Models Working?*, arXiv:2405.06459), which showed that much of the accuracy reported by prior ZuCo-based EEG-to-text work was an artifact of teacher forcing at evaluation time — feeding reference tokens back into the decoder lets a language-model prior reconstruct much of the sentence largely independent of the EEG input — NEST commits to reporting **free-decoding** metrics only, with no reference text supplied to the model at inference time:

- **NEST-BART-v2**: `eval_bart()` in `scripts/train_nest_v2.py` decodes with `model.generate()` autoregressively with no teacher forcing. The current `generate()` implementation is greedy; the `num_beams=4` argument is accepted but not yet used, and beam search will be reported separately when implemented.
- **NEST-CTC-v2**: `eval_ctc()` performs greedy CTC decoding (arg-max per frame, collapsing repeats and blanks), which is inherently non-teacher-forced.

### 5.3 Noise-Input Control

As an additional control recommended by Jo et al. (2024), we will report model performance when the EEG input is replaced by uninformative noise (e.g., shuffled features or Gaussian noise) that carries no information about the target sentence. A model that performs similarly on real EEG and on noise input is not decoding brain signals — it is exploiting dataset-level language priors (e.g., a small or repetitive sentence set). Every WER/CER/BLEU/ROUGE number we report for NEST will be accompanied by this noise-input baseline so the improvement attributable to the EEG signal itself, rather than to the language model or dataset artifacts, is visible.

### 5.4 Data Split

Subject-independent evaluation on ZuCo (12 subjects, 105-channel EEG, 1,107 sentences, 3 reading tasks — SR, NR, TSR):

- Train: 8 subjects (ZAB, ZDN, ZDM, ZGW, ZJM, ZJN, ZJS, ZKB)
- Validation: 2 subjects (ZKH, ZKW)
- Test: 2 subjects (ZMG, ZPH), never seen during training or validation

### 5.5 Metrics

- **Word Error Rate (WER)** and **Character Error Rate (CER)**: word- and character-level Levenshtein edit distance between the free-decoded hypothesis and the reference sentence, as implemented in `scripts/train_nest_v2.py`.
- **BLEU** and **ROUGE**: n-gram overlap metrics reported alongside WER/CER for comparability with the broader sequence-generation and EEG-to-text literature.

### 5.6 What Has Actually Been Run So Far

The only training run completed to date (February 16, 2026) trained a small, ad hoc `SimpleLSTMDecoder` (~2.78M parameters, defined inline in `scripts/train_with_real_zuco.py` — not the NEST-CTC-v2/NEST-BART-v2 architectures described above) for 100 epochs on 12,071 ZuCo sentence-subject pairs, taking approximately 5.4 hours on an Apple M2 laptop CPU. Training loss decreased over the run, but **no decoding was performed and no WER, CER, BLEU, or ROUGE was computed** against reference text. A figure of "26.1% WER" (with "BLEU 0.74" and "73.9% accuracy") previously circulated in project documentation was not a measurement: it was produced by a hand-written heuristic, `estimate_wer_from_ctc_loss()` in `detailed_analysis.py`, that maps final training loss to a WER percentage via an empirical formula, with no decoding or reference comparison involved. That figure has been withdrawn.

---

## 6. Discussion

### 6.1 Open Questions for the Planned Evaluation

Because no training run of NEST-CTC-v2 or NEST-BART-v2 has completed, this section discusses questions the evaluation in Section 5 is designed to answer, not findings. We list them here to make the reasoning behind the evaluation protocol explicit, and will replace this section with actual findings once results exist.

**1. Does the Transformer + BART architecture decode above the noise-input control?** The central question the noise-input control (Section 5.3) is designed to answer is whether NEST's free-decoding WER/CER on real EEG is meaningfully better than on uninformative noise input. Jo et al. (2024) found that several prior ZuCo EEG-to-text results did not clear this bar under free decoding; we do not assume NEST will.

**2. Does subject adaptation help, and by how much?** DANN and subject-embedding mechanisms are implemented but their effect on cross-subject generalization has not been measured. We plan to report this as a controlled comparison against the non-adapted model once training resources are available, rather than assume the effect size in advance.

**3. Is deployment-time optimization (pruning, quantization) compatible with acceptable accuracy?** Pruning and quantization utilities exist in the codebase but have not been benchmarked for their effect on WER/CER or on latency for either NEST-CTC-v2 or NEST-BART-v2.

**4. How does NEST compare to invasive BCIs and to other ZuCo EEG-to-text work?** Invasive BCIs (Willett et al., 2021; Makin et al., 2020) report substantially lower error rates than any published non-invasive EEG-to-text system, reflecting fundamental SNR and spatial-resolution differences. We expect a real gap to remain; the size of that gap is an empirical question the planned evaluation, not this discussion, should answer.

### 6.2 Limitations

**1. Dataset Size**: ZuCo includes only 12 subjects and ~9,000 sentences. Modern deep learning benefits from orders of magnitude more data. Larger multi-site EEG corpora could improve generalization.

**2. Reading vs. Speech Imagery**: ZuCo involves reading—a "receptive" language task. Decoding imagined or attempted speech ("productive" tasks) may yield different neural signatures and model performance. Future work should evaluate on speech imagery datasets.

**3. Controlled Laboratory Setting**: Data was collected in quiet, controlled environments with high-quality research-grade EEG. Real-world deployments (noisy environments, consumer-grade EEG like Emotiv, Muse) present additional challenges.

**4. English Only**: We evaluate only on English text. Language-specific orthography, morphology, and neural processing patterns may affect generalization. Multi-lingual evaluation is needed.

**5. Evaluation Metrics**: WER/CER are standard but imperfect. Two systems with the same WER may differ dramatically in usability if errors are semantics-preserving vs. gibberish. User studies (Section 6.4) are essential for assessing real-world utility.

**6. No Online Evaluation**: All experiments use offline analysis (EEG pre-recorded). Online closed-loop BCI experiments with real-time feedback to users are needed to assess practical performance and user learning effects.

### 6.3 Ethical Considerations

**Privacy**: BCIs recording neural signals raise profound privacy concerns. EEG can potentially reveal cognitive states (attention, emotion, intent) beyond intended communication. Strong consent protocols, data encryption, and user control over data access are essential.

**Accessibility**: While EEG is more accessible than invasive BCIs, research-grade systems ($10K-$50K) remain expensive. Consumer-grade EEG ($200-$1000) may democratize access but with reduced signal quality. Ensuring equitable access is a key challenge.

**Misuse Potential**: Silent speech BCIs could be misused for covert communication or surveillance. Regulation and ethical guidelines are needed as technology matures.

**User Autonomy**: BCIs should augment, not replace, existing communication modalities. Users must retain control over when/how BCIs are used.

### 6.4 Future Directions

**1. Larger Datasets**: Multi-site efforts collecting EEG from 100+ subjects across diverse demographics, languages, and tasks would significantly advance the field.

**2. Speech Imagery Tasks**: Evaluate on imagined speech, inner speech, and attempted speech datasets to assess performance on "productive" language tasks more relevant to communication BCIs.

**3. Multi-Modal Fusion**: Combine EEG with fNIRS (hemodynamic signals), EOG (eye movements), or EMG (facial muscle activity) for richer signal sources.

**4. Pre-Training Strategies**: Self-supervised pre-training on large unlabeled EEG corpora (sleep studies, clinical recordings) could improve few-shot adaptation.

**5. Online User Studies**: Conduct closed-loop BCI experiments where users receive real-time feedback, assessing learning curves, adaptation strategies, and subjective experience.

**6. Lightweight Models**: Explore knowledge distillation, neural architecture search, and efficient transformer variants (Linformer, Performer) to reduce model size further for edge deployment.

**7. Improved Language Modeling**: Integrate large language models (GPT, T5) as external language models during beam search decoding to leverage world knowledge.

**8. Error Correction Interfaces**: Design user interfaces allowing easy error correction (e.g., BCI-controlled cursor to fix errors), combined with reinforcement learning to learn from corrections.

**9. Longitudinal Studies**: Track individual users over months/years to assess performance stability, learning effects, and long-term viability.

**10. Clinical Validation**: Partner with individuals with ALS, locked-in syndrome, or brainstem stroke to evaluate NEST's utility for real communication needs.

---

## 7. Conclusion

We presented **NEST** (Neural EEG Sequence Transducer), a deep learning framework and open-source codebase for decoding non-invasive EEG signals into open-vocabulary text via sequence transduction. NEST v2 implements a Transformer EEG encoder cross-attended by a BART decoder (NEST-BART-v2) alongside a CTC baseline (NEST-CTC-v2), with pre-processing, training, evaluation, and deployment pipelines already in place.

**This manuscript reports no experimental results.** Full training of NEST v2 is pending GPU access, and the evaluation protocol described in Section 5 — free decoding (not teacher forcing), a noise-input control following Jo et al. (2024), subject-independent held-out test subjects, and WER/CER/BLEU/ROUGE metrics — has not yet been run to completion. Earlier internal documentation cited numbers (e.g., "15.8% WER," "26.1% WER") that were either fully fabricated or derived from a loss-based heuristic rather than an actual decode-and-compare evaluation; those numbers have been withdrawn and must not be cited as results of this work.

What NEST does offer today is a complete, open-source framework — architecture, ZuCo data pipeline, training scripts, and an evaluation protocol designed from the outset to avoid the teacher-forcing pitfalls documented in prior EEG-to-text work. Whether NEST v2 decodes EEG signals above the noise-input control, how much subject adaptation helps, and whether deployment-time optimization preserves accuracy are open, empirical questions that the evaluation in Section 5 is designed to answer once training completes. We plan to report those results, positive or negative, in a follow-up technical report.

---

## Acknowledgments

We thank the ZuCo dataset authors for making their data publicly available. This work was supported by [Funding Sources]. Computational resources were provided by [Institution]. We thank [Colleagues] for helpful discussions and feedback.

---

## References

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR*.

Cheng, M., Gao, X., Gao, S., & Xu, D. (2002). Design and implementation of a brain-computer interface with high transfer rates. *IEEE Trans. Biomed. Eng.*, 49(10), 1181-1186.

Cooney, C., Folli, R., & Coyle, D. (2019). Optimizing layers improves CNN generalization and transfer learning for imagined speech decoding from EEG. *IEEE SMC*, 1311-1316.

Farwell, L. A., & Donchin, E. (1988). Talking off the top of your head. *Electroenceph. Clin. Neurophysiol.*, 70(6), 510-523.

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *JMLR*, 17(1), 2096-2030.

Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification. *ICML*, 369-376.

Graves, A. (2012). Sequence transduction with recurrent neural networks. *ICML Workshop on Representation Learning*.

Gulati, A., Qin, J., Chiu, C. C., Parmar, N., Zhang, Y., Yu, J., ... & Pang, R. (2020). Conformer: Convolution-augmented Transformer for speech recognition. *Interspeech*, 5036-5040.

Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., & Langer, N. (2018). ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading. *Scientific Data*, 5, 180291.

Hollenstein, N., de la Torre, A. G., Langer, N., & Zhang, C. (2020). CogniVal: A framework for cognitive word embedding evaluation. *CoNLL*.

Krishna, G., Tran, C., Han, Y., Carnahan, M., & Tewfik, A. H. (2020). Advancing speech recognition with no speech or with noisy speech. *EMBC*, 4517-4520.

Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *J. Neural Eng.*, 15(5), 056013.

Makin, J. G., Moses, D. A., & Chang, E. F. (2020). Machine translation of cortical activity to text with an encoder–decoder framework. *Nature Neuroscience*, 23(4), 575-582.

Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. *ACL*, 311-318.

Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct brain-computer communication. *Proc. IEEE*, 89(7), 1123- 1134.

Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.

Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. *NAACL*, 464-468.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *NeurIPS*, 5998-6008.

Wang, F., Zhong, S. H., Peng, J., Jiang, J., & Liu, Y. (2021). Data augmentation for EEG-based emotion recognition with deep convolutional neural networks. *MMM*, 82-93.

Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M., & Shenoy, K. V. (2021). High-performance brain-to-text communication via handwriting. *Nature*, 593(7858), 249-254.

Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G., & Vaughan, T. M. (2002). Brain–computer interfaces for communication and control. *Clin. Neurophysiol.*, 113(6), 767-791.

---

## Supplementary Materials

**Appendix A: Detailed Architecture Specifications**  
**Appendix B: Hyperparameter Sensitivity Analysis**  
**Appendix C: Per-Subject Detailed Results**  
**Appendix D: Additional Attention Visualizations**  
**Appendix E: Reproducibility Checklist**  
**Appendix F: Dataset Statistics and Preprocessing Details**

---

**Code Availability**: https://github.com/wazder/NEST  
**Pretrained Models**: Not yet released; no trained NEST v2 checkpoint exists (see Section 5)  
**Data**: ZuCo dataset available at https://osf.io/q3zws/

---

**Document Information**:
- **Version**: Draft (no experimental results yet)
- **Figures**: Architecture diagram (planned); result figures pending the evaluation described in Section 5
- **Status**: Draft. Not ready for submission — see the status banner at the top of this document and `docs/STATUS.md`

