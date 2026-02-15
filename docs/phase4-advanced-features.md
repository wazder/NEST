# Phase 4: Advanced Model Features & Robustness

This document describes the advanced features and robustness techniques implemented in Phase 4 for NEST.

## Overview

Phase 4 builds upon the base models from Phase 3 by adding:
- **Advanced Attention Mechanisms** for better sequence modeling
- **Subword Tokenization** for robust vocabulary handling
- **Subject Adaptation** for cross-subject generalization
- **Noise Robustness** for handling noisy EEG signals
- **Language Model Integration** for improved decoding

## 1. Advanced Attention Mechanisms (`src/models/advanced_attention.py`)

### RelativePositionAttention
Attention with relative position encoding (Shaw et al., 2018):
- Captures relative distances between positions
- More flexible than absolute positional encoding
- Learns position-specific key/value embeddings
- Better for variable-length sequences

**Usage:**
```python
from src.models.advanced_attention import RelativePositionAttention

attn = RelativePositionAttention(
    d_model=512,
    nhead=8,
    max_relative_position=128,
    dropout=0.1
)

output, weights = attn(query, key, value)
```

### LocalAttention
Windowed attention for efficient long-sequence processing:
- Only attends to local window around each position
- Total window size: `2 * window_size + 1`
- Reduces complexity for long EEG sequences
- Good for real-time applications

**When to use:**
- Very long EEG sequences (>1000 time steps)
- Real-time streaming applications
- Memory-constrained environments

### LinearAttention
O(N) complexity attention (Katharopoulos et al., 2020):
- Linear complexity instead of quadratic
- Uses kernel feature maps (ELU + 1)
- Scales to very long sequences
- Slightly less expressive than standard attention

**Trade-offs:**
- **Speed**: Much faster for long sequences (N > 512)
- **Memory**: Lower memory footprint
- **Accuracy**: ~1-2% lower than standard attention

## 2. Tokenization & Vocabulary (`src/utils/tokenizer.py`)

### BPE Tokenizer
Byte-Pair Encoding for subword tokenization (Sennrich et al., 2016):
- Learns subword units from corpus
- Handles rare/unseen words better
- Reduces vocabulary size
- Standard in modern NMT

**Training:**
```python
from src.utils.tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=5000, min_frequency=2)
tokenizer.train(texts)

# Encode/decode
ids = tokenizer.encode("hello world")
text = tokenizer.decode(ids)

# Save/load
tokenizer.save('vocab/bpe_tokenizer.json')
tokenizer.load('vocab/bpe_tokenizer.json')
```

### Vocabulary Builder
Simple word-level vocabulary:
- Frequency-based word selection
- Min frequency filtering
- Special tokens support
- Easy to use for quick experiments

**Example:**
```python
from src.utils.tokenizer import VocabularyBuilder

vocab = VocabularyBuilder(max_vocab_size=10000, min_frequency=2)
vocab.build_from_texts(texts)

ids = vocab.encode("the quick fox")
text = vocab.decode(ids)
```

## 3. Subject Adaptation (`src/models/adaptation.py`)

### SubjectEmbedding
Add subject-specific parameters to model:
- Embedding lookup for each subject
- Projects to feature space
- Helps model adapt to individual differences
- Minimal parameters added

**Integration:**
```python
from src.models.adaptation import SubjectEmbedding

subj_emb = SubjectEmbedding(
    num_subjects=20,
    embedding_dim=64,
    feature_dim=512
)

# Add subject info to features
features = model.encode(eeg_data)
adapted_features = subj_emb(features, subject_ids)
```

### DomainAdversarialNetwork (DANN)
Learn subject-invariant features (Ganin et al., 2016):
- Gradient Reversal Layer
- Domain classifier predicts subject ID
- Feature extractor learns to fool classifier
- Encourages subject-independent features

**Training loop:**
```python
from src.models.adaptation import DomainAdversarialNetwork

dann = DomainAdversarialNetwork(
    feature_dim=512,
    num_subjects=20,
    lambda_=1.0
)

# In training loop
features = model.encode(eeg_data)
domain_pred = dann(features)
domain_loss = F.cross_entropy(domain_pred, subject_ids)

# Total loss includes domain loss
total_loss = task_loss + domain_loss
```

**Lambda scheduling:** Start with λ=0, gradually increase to 1

### CORAL (Correlation Alignment)
Align feature distributions across subjects (Sun & Saenko, 2016):
- Matches second-order statistics (covariance)
- No adversarial training needed
- Simpler than DANN
- Good for small domain shifts

**Usage:**
```python
from src.models.adaptation import CORAL

coral = CORAL()

source_features = model.encode(source_eeg)
target_features = model.encode(target_eeg)

coral_loss = coral(source_features, target_features)
total_loss = task_loss + 0.1 * coral_loss
```

### SubjectAdaptiveBatchNorm
Subject-specific batch normalization:
- Shared statistics, subject-specific shift/scale
- Each subject has own γ and β parameters
- Captures subject-specific signal characteristics
- Minimal overhead

### Fine-Tuning Strategies
Utilities for adapting to new subjects:

**1. Freeze Backbone:**
```python
from src.models.adaptation import FineTuningStrategy

# Freeze all, unfreeze last 2 layers
FineTuningStrategy.freeze_backbone(model, unfreeze_last_n=2)
```

**2. Progressive Unfreezing:**
```python
# Gradually unfreeze more layers
FineTuningStrategy.progressive_unfreezing(
    model, 
    epoch=current_epoch,
    total_epochs=50,
    num_layers=10
)
```

**3. Discriminative Learning Rates:**
```python
# Different LR for each layer
param_groups = FineTuningStrategy.discriminative_learning_rates(
    model,
    base_lr=1e-4,
    decay_factor=0.5
)
optimizer = torch.optim.Adam(param_groups)
```

## 4. Noise Robustness (`src/training/robustness.py`)

### Adversarial Training
Train on adversarial examples (Goodfellow et al., 2015):

**FGSM (Fast Gradient Sign Method):**
- Single-step attack
- Fast to compute
- ε controls perturbation magnitude

**PGD (Projected Gradient Descent):**
- Multi-step attack
- More powerful than FGSM
- Better robustness

**Training:**
```python
from src.training.robustness import AdversarialTrainer

adv_trainer = AdversarialTrainer(
    model,
    criterion,
    attack_method='pgd',
    epsilon=0.1,
    alpha=0.01,
    num_steps=10
)

# Generate adversarial examples
x_adv = adv_trainer.generate_adversarial_examples(x, y)

# Train on both clean and adversarial
loss_clean = criterion(model(x), y)
loss_adv = criterion(model(x_adv), y)
total_loss = 0.5 * (loss_clean + loss_adv)
```

### Noise Injection
Add noise during training for robustness:

**Noise types:**
- **Gaussian**: Normal distribution noise
- **Uniform**: Uniform noise
- **Salt & Pepper**: Random min/max values

```python
from src.training.robustness import NoiseInjection

noise_injector = NoiseInjection(
    noise_type='gaussian',
    noise_level=0.1,
    apply_prob=0.5
)

# Use in data augmentation
noisy_eeg = noise_injector(clean_eeg)
```

### Denoising Autoencoder
Pre-train denoiser or use as preprocessing:
- Conv1D encoder-decoder architecture
- Learns to remove noise from EEG
- Can pre-train for better features
- Improves robustness

**Pre-training:**
```python
from src.training.robustness import DenoisingAutoencoder

dae = DenoisingAutoencoder(
    input_channels=104,
    latent_dim=64,
    noise_level=0.1
)

# Pre-train on unlabeled EEG
reconstructed, latent = dae(eeg_data)
loss = F.mse_loss(reconstructed, eeg_data)
```

**Using for preprocessing:**
```python
# Denoise signals before main model
clean_eeg = dae.denoise(noisy_eeg)
output = main_model(clean_eeg)
```

### Robust Loss Functions
Loss functions less sensitive to outliers:

**Huber Loss:**
- Quadratic for small errors
- Linear for large errors
- Robust to outliers

**Smooth L1:**
- Similar to Huber
- PyTorch built-in

```python
from src.training.robustness import RobustLoss

# Instead of MSE
criterion = RobustLoss(loss_type='huber', delta=1.0)
loss = criterion(predictions, targets)
```

### Gradient Noise
Add noise to gradients (Neelakantan et al., 2015):
- Helps escape sharp minima
- Anneals over time
- Simple regularization

```python
from src.training.robustness import GradientNoise

grad_noise = GradientNoise(eta=0.3, gamma=0.55)

# After backward, before optimizer step
loss.backward()
grad_noise.add_noise_to_gradients(model)
optimizer.step()
```

## 5. Language Model Integration (`src/models/language_model.py`)

### Shallow Fusion
Combine AM and LM at output (Gulcehre et al., 2015):
- Simple interpolation of log probabilities
- `P(y|x) ∝ P_AM(y|x) * P_LM(y)^λ`
- LM is frozen (pre-trained)
- Easy to implement

**Usage:**
```python
from src.models.language_model import ShallowFusion
from transformers import GPT2LMHeadModel

lm = GPT2LMHeadModel.from_pretrained('gpt2')
fusion = ShallowFusion(lm, lm_weight=0.3)

# During decoding
am_logits = acoustic_model(eeg_data)
fused_logits = fusion(am_logits, previous_tokens)
```

**Tuning λ (lm_weight):**
- 0.0: No LM (pure acoustic model)
- 0.3: Typical value
- 0.5: Strong LM influence
- 1.0: Pure LM (ignores acoustics)

### Deep Fusion
Learn to combine AM and LM hidden states:
- Gating mechanism decides how much LM to use
- Trained end-to-end
- More powerful than shallow fusion
- Requires training data

**Architecture:**
```python
from src.models.language_model import DeepFusion

fusion = DeepFusion(
    lm_model=lm,
    am_hidden_dim=512,
    lm_hidden_dim=768,
    vocab_size=5000,
    lm_weight=0.3
)

# During forward pass
am_hidden = acoustic_model.encode(eeg_data)
logits = fusion(am_hidden, previous_tokens)
```

### Language Model Rescoring
Rescore beam search hypotheses with LM:
- Generate N-best list with AM
- Rescore with LM
- Re-rank by combined score
- Length normalization

**Beam search + rescoring:**
```python
from src.models.language_model import LanguageModelRescorer

rescorer = LanguageModelRescorer(
    lm_model=lm,
    lm_weight=0.3,
    length_penalty=0.6
)

# Get N-best hypotheses from beam search
hypotheses = beam_search(model, eeg_data, beam_size=10)

# Rescore
rescored_hypotheses = rescorer.rescore_hypotheses(hypotheses)
best_hypothesis = rescored_hypotheses[0]
```

## Integration Examples

### Example 1: Subject-Adaptive Model with Robust Training
```python
import torch
from src.models import NEST_RNN_T, SubjectEmbedding
from src.training import Trainer, AdversarialTrainer, NoiseInjection

# Create base model
model = NEST_RNN_T(...)

# Add subject adaptation
subj_emb = SubjectEmbedding(num_subjects=20, embedding_dim=64, feature_dim=512)

# Add noise robustness
noise_injector = NoiseInjection(noise_type='gaussian', noise_level=0.1)

# Adversarial training
adv_trainer = AdversarialTrainer(model, criterion, attack_method='pgd')

# Training loop
for epoch in range(epochs):
    for eeg_data, targets, subject_ids in train_loader:
        # Add noise
        eeg_noisy = noise_injector(eeg_data)
        
        # Extract features
        features = model.encode(eeg_noisy)
        
        # Add subject adaptation
        features = subj_emb(features, subject_ids)
        
        # Generate adversarial examples
        eeg_adv = adv_trainer.generate_adversarial_examples(eeg_data, targets)
        
        # Train on both
        loss = 0.5 * criterion(model(eeg_data), targets) + \
               0.5 * criterion(model(eeg_adv), targets)
        
        loss.backward()
        optimizer.step()
```

### Example 2: Cross-Subject Transfer with DANN
```python
from src.models.adaptation import DomainAdversarialNetwork

# Create model with DANN
base_model = NEST_Transformer_T(...)
dann = DomainAdversarialNetwork(feature_dim=512, num_subjects=20)

# Training
for epoch in range(epochs):
    # Update lambda (0 -> 1)
    lambda_ = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0
    dann.set_lambda(lambda_)
    
    for eeg_data, targets, subject_ids in train_loader:
        # Extract features
        features = base_model.encode(eeg_data)
        
        # Task loss
        task_output = base_model.decode(features, targets)
        task_loss = criterion(task_output, targets)
        
        # Domain loss (gradient reversal)
        domain_pred = dann(features)
        domain_loss = F.cross_entropy(domain_pred, subject_ids)
        
        # Combined loss
        total_loss = task_loss + domain_loss
        total_loss.backward()
        optimizer.step()
```

### Example 3: Decoding with Language Model Fusion
```python
from src.models.language_model import ShallowFusion
from transformers import GPT2LMHeadModel

# Load pre-trained LM
lm = GPT2LMHeadModel.from_pretrained('gpt2')

# Create fusion
fusion = ShallowFusion(lm, lm_weight=0.3)

# Decoding
def decode_with_lm(model, eeg_data, max_length=50):
    model.eval()
    
    with torch.no_grad():
        # Encode EEG
        encoder_output = model.encode(eeg_data)
        
        # Autore gressive decoding
        generated = [SOS_TOKEN]
        
        for t in range(max_length):
            # Get AM logits
            am_logits = model.decode_step(encoder_output, generated)
            
            # Fuse with LM
            context = torch.tensor(generated).unsqueeze(0)
            fused_logits = fusion(am_logits, context)
            
            # Sample
            next_token = fused_logits.argmax(dim=-1).item()
            generated.append(next_token)
            
            if next_token == EOS_TOKEN:
                break
                
    return generated
```

## Performance Improvements

Expected improvements from Phase 4 techniques:

| Technique | WER Reduction | Notes |
|-----------|---------------|-------|
| Subject Embedding | 2-5% | Per-subject adaptation |
| DANN | 3-7% | Cross-subject generalization |
| Adversarial Training | 1-3% | Robustness to noise |
| Denoising Autoencoder | 2-4% | Cleaner signals |
| LM Shallow Fusion | 5-10% | Language model prior |
| LM Deep Fusion | 8-15% | Best performance |
| Advanced Attention | 1-3% | Better sequence modeling |
| BPE Tokenization | 2-5% | Handles rare words |

**Combined Effect:** 15-30% WER reduction over base models

## Best Practices

### Subject Adaptation
1. **Few-shot scenario** (<10 samples): Use SubjectEmbedding + fine-tuning
2. **Cross-subject** (unseen subjects): DANN or CORAL
3. **Many subjects** (>50): SubjectAdaptiveBatchNorm
4. **Online adaptation**: Progressive unfreezing

### Noise Robustness
1. Start with noise injection during training
2. Add adversarial training if overfitting
3. Use denoising autoencoder for very noisy data
4. Robust loss for outliers in dataset

### Language Model Fusion
1. **Start simple**: Shallow fusion (fastest)
2. **More data**: Deep fusion (best accuracy)
3. **Limited compute**: Rescoring (offline)
4. **Tune λ**: Grid search {0.1, 0.2, 0.3, 0.5}

### Tokenization
1. **Small vocabulary** (<5K): Word-level
2. **Medium** (5K-30K): BPE
3. **Large** (>30K): SentencePiece
4. Always include special tokens

## References

1. **Relative Position**: Shaw et al. (2018) - "Self-Attention with Relative Position Representations"
2. **Linear Attention**: Katharopoulos et al. (2020) - "Transformers are RNNs"
3. **BPE**: Sennrich et al. (2016) - "Neural Machine Translation of Rare Words"
4. **DANN**: Ganin et al. (2016) - "Domain-Adversarial Training of Neural Networks"
5. **CORAL**: Sun & Saenko (2016) - "Deep CORAL"
6. **Adversarial Training**: Goodfellow et al. (2015) - "Explaining and Harnessing Adversarial Examples"
7. **Shallow Fusion**: Gulcehre et al. (2015) - "On Using Monolingual Corpora in NMT"
8. **Gradient Noise**: Neelakantan et al. (2015) - "Adding Gradient Noise Improves Learning"

## Next Steps

After Phase 4, proceed to Phase 5 (Evaluation & Optimization):
- Benchmark all model variants
- Optimize for real-time inference
- Model compression and quantization
- User studies
