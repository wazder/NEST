# Attention Mechanisms for Neural Signal Processing

## Introduction
Attention mechanisms have revolutionized sequence modeling by enabling neural networks to dynamically focus on relevant parts of the input. In the context of neural signal processing, attention provides interpretable and effective methods for handling the spatial, temporal, and spectral complexity of EEG data.

## Fundamentals of Attention

### Basic Attention Mechanism
Introduced by Bahdanau et al. (2015) for machine translation:

```
Attention Score: e_ij = a(s_i, h_j)
Attention Weights: α_ij = softmax(e_ij)
Context Vector: c_i = Σ_j α_ij · h_j
```

Where:
- `s_i`: Decoder state at step i
- `h_j`: Encoder hidden state at position j
- `a(·)`: Alignment function (typically MLP or dot product)

**Key Insight:** Instead of compressing entire input into fixed vector, decoder accesses all encoder states selectively.

### Self-Attention (Intra-Attention)
Introduced in the Transformer (Vaswani et al., 2017):

```
Q = X · W_Q  (Queries)
K = X · W_K  (Keys)
V = X · W_V  (Values)

Attention(Q,K,V) = softmax(QK^T / √d_k) · V
```

**Advantages:**
- Captures dependencies regardless of distance
- Parallelizable (no sequential dependency)
- Explicit relationship modeling

### Multi-Head Attention
Allows model to attend to different representation subspaces:

```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
where head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

**Benefits:**
- Different heads can specialize (e.g., local vs. global patterns)
- Increases model capacity
- Improves robustness

## Attention for EEG Processing

### Challenges of EEG Data
1. **Multi-Channel Structure**: 64-128 electrodes with spatial relationships
2. **Temporal Dynamics**: Oscillations at multiple time scales (theta, alpha, beta, gamma)
3. **Spectral Content**: Information distributed across frequency bands
4. **Low SNR**: Attention must be robust to noise
5. **Subject Variability**: Patterns vary across individuals

### Spatial Attention

#### Channel-Wise Attention
**Objective:** Identify which EEG channels are most relevant

**Architecture:**
```python
# Pseudo-code
channel_features = GlobalAveragePooling(EEG_data)  # [batch, channels, features]
attention_weights = softmax(MLP(channel_features))  # [batch, channels]
attended_features = EEG_data * attention_weights.unsqueeze(-1)
```

**Applications:**
- **Yang et al. (2018)**: Channel attention for motor imagery
  - Automatically identifies motor cortex channels
  - Improves classification accuracy by 8%
  
- **Song et al. (2019)**: Spatial attention for emotion recognition
  - Learns task-specific channel importance
  - Interpretable: Highlights frontal regions for valence

**Benefits:**
- Handles electrode variability across subjects
- Reduces dimensionality
- Provides interpretability (which brain regions matter)

**Limitations:**
- Assumes channel independence
- May miss spatial patterns (e.g., left vs. right hemisphere)

#### Graph Attention Networks (GAT)
**Objective:** Model relationships between EEG channels based on spatial topology

**Architecture:**
```
# Channel graph: Nodes = electrodes, Edges = spatial proximity
Graph Attention: α_ij = softmax(LeakyReLU(a^T [W·h_i || W·h_j]))
Updated Features: h_i' = σ(Σ_{j∈N(i)} α_ij · W · h_j)
```

**Applications:**
- **Song et al. (2020)**: EEG Graph Neural Networks
  - Constructs electrode graph based on 10-20 system
  - Attention weights indicate functional connectivity
  - Outperforms CNN baselines on emotion recognition

- **Zheng et al. (2021)**: Dynamic graph attention
  - Updates graph structure based on learned attention
  - Captures task-dependent connectivity

**Benefits:**
- Incorporates spatial topology
- Models inter-channel relationships
- Biologically plausible (functional connectivity)

**Limitations:**
- Computational overhead
- Graph structure choice affects performance
- May overfit with limited data

### Temporal Attention

#### Frame-Level Attention
**Objective:** Weight different time points in EEG sequence

**Architecture:**
```python
temporal_features = Encoder(EEG_sequence)  # [batch, time, features]
attention_scores = MLP(temporal_features)  # [batch, time]
attention_weights = softmax(attention_scores)
context = sum(attention_weights * temporal_features)
```

**Applications:**
- **Lawhern et al. (2018)**: EEGNet with temporal attention
  - Identifies critical time windows for P300 detection
  - Attention peaks at 300ms (matching P300 latency)

- **Schirrmeister et al. (2019)**: Deep ConvNets with attention
  - Temporal attention for motor imagery
  - Reveals onset and offset of motor planning

**Benefits:**
- Handles variable-length sequences
- Interpretable (when in time does signal matter)
- Robust to temporal jitter

#### Multi-Scale Temporal Attention
**Objective:** Capture EEG dynamics at multiple time scales

**Architecture:**
- Parallel attention at different temporal resolutions
- Combine short-term (100ms), medium-term (1s), long-term (5s+) context

**Applications:**
- **Li et al. (2020)**: Hierarchical temporal attention for seizure detection
  - Attention at 1s, 10s, and 60s windows
  - Captures both ictal onset and evolution

**Benefits:**
- Matches EEG's multi-scale nature
- Improves performance on long recordings

### Spectral Attention

#### Frequency-Band Attention
**Objective:** Weight different frequency bands (delta, theta, alpha, beta, gamma)

**Methodology:**
1. Decompose EEG into frequency bands (e.g., using wavelet transform)
2. Apply attention across bands
3. Weighted combination

**Applications:**
- **Hou et al. (2020)**: Multi-band attention for drowsiness detection
  - Higher attention to alpha band (8-12 Hz) during drowsiness
  - Attention weights correlate with known physiology

- **Zhang et al. (2021)**: Frequency attention for seizure prediction
  - Dynamically weights gamma and beta bands
  - Anticipates spectral changes before seizure onset

**Benefits:**
- Incorporates domain knowledge (frequency bands)
- Interpretable (which rhythms are relevant)
- Can adapt to task-specific spectral signatures

#### Time-Frequency Attention
**Objective:** Jointly attend over time and frequency

**Architecture:**
```python
# Compute spectrogram: [batch, channels, time, frequency]
TF_features = Spectrogram(EEG)
# 2D attention over (time, frequency)
attention_map = softmax(Conv2D(TF_features))  # [batch, channels, time, freq]
attended = TF_features * attention_map
```

**Applications:**
- **Dose et al. (2018)**: Time-frequency attention for sleep staging
  - Attention highlights sleep spindles (12-14 Hz, brief bursts)
  - Matches expert visual analysis

**Benefits:**
- Captures transient spectral events (spindles, K-complexes)
- More expressive than separate time/frequency attention

## Cross-Attention for EEG-to-Text

### Encoder-Decoder Cross-Attention
**Objective:** Align EEG representations with text tokens during generation

**Architecture (following Transformer decoder):**
```python
# Encoder: EEG → feature sequence
encoder_states = EEG_Encoder(eeg_signals)  # [batch, T_eeg, d_model]

# Decoder: Generate text with cross-attention
for each decoder layer:
    # Self-attention on generated text
    text_hidden = SelfAttention(text_embeddings)
    
    # Cross-attention: text queries attend to EEG keys/values
    Q = text_hidden · W_Q  # Queries from text
    K = encoder_states · W_K  # Keys from EEG
    V = encoder_states · W_V  # Values from EEG
    
    context = Attention(Q, K, V)
    output = FeedForward(text_hidden + context)
```

**Key Insight:** Decoder learns which parts of EEG signal correspond to each word

### Applications to Speech Decoding

#### Alignment Learning
**Challenge:** Map continuous EEG to discrete text without explicit alignment

**Solution:** Cross-attention learns soft alignment
- Early decoder steps attend to early EEG
- Attention shifts monotonically (for speech-like tasks)
- Can handle variable-rate speaking/thinking

**Example:**
- **Sun et al. (2021)**: Transformer for ZuCo
  - Cross-attention weights show word-by-word alignment
  - Attention peaks ~200-400ms after word onset (matches word recognition)

#### Multi-Modal Attention
For datasets with additional modalities (eye-tracking in ZuCo):

```python
# Combine EEG and eye-tracking
eeg_features = EEG_Encoder(eeg)
eye_features = Eye_Encoder(fixations)

# Cross-modal attention
cross_attention = MultiHeadAttention(
    query=eeg_features,
    key=eye_features,
    value=eye_features
)
fused = Concat(eeg_features, cross_attention)
```

**Benefits:**
- Leverages complementary information
- Eye-tracking provides explicit word boundaries
- Improves alignment accuracy

### Monotonic Attention for Streaming

#### Problem with Standard Attention
- Attends to entire input sequence
- Not suitable for real-time/streaming applications
- Future information not available

#### Monotonic Attention (Raffel et al., 2017)
**Constraints:**
- Attention cannot move backward
- Suitable for speech/language (left-to-right)

**Mechanism:**
```python
# At each decoder step, decide whether to move attention forward
for t in decoder_steps:
    p_move = sigmoid(score(encoder_state[attention_pos]))
    if sample(p_move):
        attention_pos += 1
    context = encoder_state[attention_pos]
```

**Applications:**
- **He et al. (2019)**: Monotonic attention for online speech recognition
  - Enables streaming without waiting for full utterance
  - Slight performance trade-off vs. full attention

**Relevance to NEST:**
- Critical for real-time BCI applications
- Transducers naturally incorporate monotonic alignment
- Can achieve low-latency decoding

### Chunked Attention
**Objective:** Balance between full attention (offline) and monotonic (online)

**Mechanism:**
- Process input in chunks (e.g., 1-second windows)
- Full attention within chunk
- Monotonic progression across chunks

**Benefits:**
- Lower latency than full attention
- Better performance than strict monotonic
- Configurable chunk size trades off latency vs. accuracy

## Advanced Attention Variants

### 1. Sparse Attention
**Motivation:** Full attention is O(n²), expensive for long EEG sequences

**Approaches:**
- **Local Attention**: Only attend to nearby time points
- **Strided Attention**: Attend to every k-th position
- **Learned Sparsity**: Learn which positions to attend

**Application:**
- **Child et al. (2019)**: Sparse Transformers
  - Factorized attention patterns
  - Applicable to long EEG recordings (hours)

### 2. Relative Position Attention
**Motivation:** EEG timing is relative, not absolute

**Mechanism:**
```python
# Standard: Absolute position encoding
attention = softmax(Q·K^T + position_bias)

# Relative: Position difference encoding
attention = softmax(Q·K^T + relative_position(i-j))
```

**Benefits:**
- Generalizes better to sequences of different lengths
- More suitable for variable-rate neural activity

**Application:**
- **Dai et al. (2019)**: Transformer-XL with relative positions
  - Adaptable to EEG where timing variability is high

### 3. Multi-Resolution Attention
**Objective:** Attend at different temporal granularities

**Architecture:**
```python
# Coarse level: Downsample EEG (e.g., 10Hz)
coarse_attn = Attention(Downsample(EEG))

# Fine level: Original EEG (e.g., 500Hz)
fine_attn = Attention(EEG, guided_by=coarse_attn)
```

**Benefits:**
- Efficient for long sequences
- Captures both rapid transients and slow trends

### 4. Adaptive Attention
**Motivation:** Not all inputs require same amount of attention

**Mechanism:**
- Learn to allocate computational budget
- Skip or simplify attention for easy samples
- More attention for difficult/ambiguous signals

**Application:**
- **Graves (2016)**: Adaptive Computation Time
  - Dynamically adjusts depth/iterations
  - Could reduce computation for clear EEG patterns

## Interpretability of Attention in BCI

### Visualizing Attention Weights

#### Spatial Attention Visualization
- **Topographic Maps**: Project channel attention onto scalp
- **Identifies Regions of Interest**: Which brain areas contribute
- **Example**: Motor imagery → attention on motor cortex

#### Temporal Attention Visualization
- **Time Series Plots**: Attention weights over time
- **Event-Related Analysis**: Average attention around stimuli
- **Example**: P300 → attention peak at 300ms

#### Attention Heatmaps (Cross-Attention)
- **2D Visualization**: EEG time (y-axis) vs. Generated word (x-axis)
- **Shows Alignment**: Which EEG segment generates which word
- **Debugging Tool**: Identify misalignments

### Limitations of Attention Interpretability
**Caution:** Attention ≠ Explanation

1. **Attention is Not Explanation (Jain & Wallace, 2019)**
   - High attention doesn't prove causality
   - Attention can be diffuse or misleading
   
2. **Multiple Attention Heads**
   - Different heads show different patterns
   - Unclear which head is "correct"

3. **Gradient-Based Methods**
   - Combine attention with gradients for better interpretation
   - Integrated gradients (Sundararajan et al., 2017)

**Best Practice for NEST:**
- Use attention as hypothesis generator
- Validate with ablation studies
- Compare to known EEG phenomena (N400, P300, etc.)

## Attention in NEST Architecture

### Encoder Attention (Conformer)

#### 1. Self-Attention for Temporal Dependencies
```python
# Multi-head self-attention over time
EEG_sequence: [batch, channels, time]
flattened: [batch, time, channels * features]
temporal_attn = MultiHeadSelfAttention(flattened)
```

**Purpose:**
- Capture long-range temporal dependencies
- Model event-related potentials (ERPs)
- Handle variable latencies across subjects

#### 2. Channel Attention for Spatial Selection
```python
# Attention over channels
channel_attn = ChannelAttention(EEG)
weighted_EEG = EEG * channel_attn
```

**Purpose:**
- Adapt to individual electrode layouts
- Reduce noise from artifact-prone channels
- Interpretability (which regions matter)

#### 3. Frequency Attention for Spectral Weighting
```python
# Multi-band decomposition
bands = WaveletTransform(EEG)  # [delta, theta, alpha, beta, gamma]
freq_attn = FrequencyAttention(bands)
weighted_bands = bands * freq_attn
```

**Purpose:**
- Task-specific frequency selection
- Adaptive to individual spectral profiles
- Domain knowledge incorporation

### Decoder Cross-Attention

#### EEG-to-Text Alignment
```python
# Decoder attends to encoder states
for each generated token:
    # Query: What to generate next
    # Key/Value: Which EEG features to use
    context = CrossAttention(
        query=decoder_state,
        key=encoder_states,
        value=encoder_states
    )
    next_token = softmax(FFN(decoder_state + context))
```

**Design Choices:**
1. **Attention Type**: Multi-head for redundancy
2. **Number of Heads**: 8-16 (balance capacity and overfitting)
3. **Monotonic Constraint**: Optional for streaming
4. **Attention Dropout**: Regularization for noisy EEG

### Transducer-Specific Attention

#### Joint Network Attention
In RNN-T, the joint network combines encoder and prediction networks:

```python
# Standard joint network
joint = tanh(W_enc · encoder_state + W_pred · predictor_state)

# Attention-augmented joint network
attention_weights = softmax(score(encoder_state, predictor_state))
context = sum(attention_weights * encoder_states)
joint = tanh(W_enc · context + W_pred · predictor_state)
```

**Benefits:**
- Allows predictor to look back at encoder history
- Improves alignment accuracy
- Small computational overhead

## Training Considerations

### Attention Regularization

#### 1. Attention Entropy Regularization
**Problem:** Attention can be too diffuse (uniform) or too sharp (single-peak)

**Solution:**
```python
# Encourage peaked attention
entropy = -sum(attn_weights * log(attn_weights))
loss = task_loss + λ * entropy  # λ > 0 for peaked attention
```

#### 2. Attention Supervision
**If alignment ground truth available:**
```python
# Supervise attention to align with known word boundaries
supervised_loss = MSE(attention_weights, ground_truth_alignment)
```

**For NEST:**
- ZuCo provides word-level timestamps
- Can supervise initial training, then remove constraint

### Attention Dropout
- Randomly zero out attention connections during training
- Prevents over-reliance on specific EEG channels/times
- Acts as data augmentation

### Initialization
- Initialize attention biases to encourage:
  - Monotonic progression (for streaming)
  - Local attention (for early training)
  - Gradually relax constraints

## Key References

### Foundational Attention Papers
1. **Bahdanau, D., et al. (2015).** "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR.

2. **Vaswani, A., et al. (2017).** "Attention is All You Need." NeurIPS.

3. **Raffel, C., et al. (2017).** "Online and Linear-Time Attention by Enforcing Monotonic Alignments." ICML.

### Attention for EEG
4. **Lawhern, V. J., et al. (2018).** "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces." Journal of Neural Engineering.

5. **Song, T., et al. (2019).** "EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks." IEEE Transactions on Affective Computing.

6. **Yang, H., et al. (2018).** "On the Use of Convolutional Neural Networks and Augmented CSP Features for Multi-class Motor Imagery of EEG Signals Classification." EMBC.

### Attention Interpretability
7. **Jain, S., & Wallace, B. C. (2019).** "Attention is not Explanation." NAACL.

8. **Sundararajan, M., et al. (2017).** "Axiomatic Attribution for Deep Networks." ICML.

### Advanced Attention
9. **Child, R., et al. (2019).** "Generating Long Sequences with Sparse Transformers." arXiv.

10. **Dai, Z., et al. (2019).** "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." ACL.

### Speech Decoding with Attention
11. **Sun, Y., et al. (2021).** "Brain2Word: Decoding Brain Activity for Language Generation." arXiv.

12. **He, Y., et al. (2019).** "Streaming End-to-end Speech Recognition for Mobile Devices." ICASSP.

## Recommendations for NEST

### Encoder Architecture
1. **Multi-scale Temporal Attention**: Capture both rapid ERPs and slow trends
2. **Spatial Graph Attention**: Model inter-channel relationships based on 10-20 system
3. **Frequency Attention**: Weight frequency bands adaptively

### Decoder Architecture
1. **Multi-head Cross-Attention**: 8-16 heads for robustness
2. **Monotonic Attention Option**: For streaming deployment
3. **Attention Supervision**: Initially guide with ZuCo timestamps

### Interpretability Strategy
1. **Visualize Attention Maps**: Validate against known ERP components
2. **Ablation Studies**: Quantify contribution of each attention type
3. **Gradient-Based Attribution**: Complement attention analysis

### Implementation Notes
- Use established libraries (e.g., PyTorch MultiheadAttention)
- Start with standard multi-head attention before customization
- Monitor attention entropy during training

## Conclusion
Attention mechanisms provide powerful tools for processing EEG signals in the NEST framework. Spatial attention handles channel selection, temporal attention captures dynamic neural patterns, and cross-attention aligns EEG with text. The multi-scale, multi-modal nature of attention makes it particularly well-suited for the challenges of EEG-to-text decoding.

---
**Last Updated:** February 2026
