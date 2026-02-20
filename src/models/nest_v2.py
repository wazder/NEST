"""
NEST v2: Neural EEG Sequence Transducer (Frequency-Domain Features)

Takes pre-processed EEG frequency features (840-dim per word) as input,
following Wang et al. 2022 (ACL) approach with improvements.

Input format: (batch, max_words, 840)
  - 105 EEG channels × 8 frequency bands (theta1/2, alpha1/2, beta1/2, gamma1/2)

Architecture variants:
  1. NEST_CTC_v2:  Linear → Transformer → CTC head
  2. NEST_BART_v2: Linear → Transformer → BART cross-attention decoder
"""

import math
import logging
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EEG_DIM = 840  # 105 channels × 8 frequency bands
VOCAB_SIZE_CTC = 29  # blank(0) + a-z(1-26) + space(27) + pad(28)


class EEGProjection(nn.Module):
    """
    Projects 840-dim EEG features to model dimension.
    Includes layer normalization and positional encoding.
    """

    def __init__(self, eeg_dim: int = EEG_DIM, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm_input = nn.LayerNorm(eeg_dim)
        self.proj = nn.Sequential(
            nn.Linear(eeg_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 840)
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.norm_input(x)
        x = self.proj(x)
        x = self.norm_out(x)
        x = x * math.sqrt(self.d_model)
        return self.dropout(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class NEST_CTC_v2(nn.Module):
    """
    NEST v2 with CTC decoder.

    Architecture:
        EEG features (batch, seq, 840)
        → EEGProjection → (batch, seq, d_model)
        → Positional encoding
        → Transformer encoder (num_layers)
        → Linear → log_softmax → CTC loss

    This is the primary model for EEG-to-text with ZuCo frequency features.
    """

    def __init__(
        self,
        eeg_dim: int = EEG_DIM,
        d_model: int = 768,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        vocab_size: int = VOCAB_SIZE_CTC,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # EEG projection
        self.eeg_proj = EEGProjection(eeg_dim, d_model, dropout)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # CTC output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, vocab_size),
        )

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"NEST_CTC_v2: d_model={d_model}, layers={num_encoder_layers}, "
            f"heads={nhead}, params={n_params:,}"
        )

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_padding_mask(
        self, lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Create boolean padding mask: True where padded."""
        batch_size = lengths.shape[0]
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        return mask >= lengths.unsqueeze(1)  # (batch, max_len)

    def forward(
        self,
        eeg: torch.Tensor,
        eeg_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            eeg: (batch, seq_len, 840) EEG feature sequences
            eeg_lengths: (batch,) actual sequence lengths (optional)

        Returns:
            log_probs: (batch, seq_len, vocab_size) log probabilities
        """
        # Project EEG features
        x = self.eeg_proj(eeg)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Create padding mask
        key_padding_mask = None
        if eeg_lengths is not None:
            key_padding_mask = self.make_padding_mask(eeg_lengths, eeg.size(1))

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # CTC output
        logits = self.output_head(x)
        return F.log_softmax(logits, dim=-1)


class NEST_BART_v2(nn.Module):
    """
    NEST v2 with BART decoder.

    Architecture:
        EEG features (batch, seq, 840)
        → EEGProjection → (batch, seq, d_model)
        → Positional encoding
        → Transformer encoder (num_encoder_layers)
        → Projected to BART d_model (1024 for bart-large, 768 for bart-base)
        → BART decoder (cross-attention to encoder output)
        → Text output

    This is the high-quality model targeting SOTA WER.
    Requires `transformers` library.
    """

    def __init__(
        self,
        eeg_dim: int = EEG_DIM,
        d_model: int = 768,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        bart_model_name: str = "facebook/bart-base",
        freeze_bart_encoder: bool = True,
        freeze_bart_decoder_layers: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()

        try:
            from transformers import BartModel, BartConfig
        except ImportError:
            raise ImportError("transformers library required for NEST_BART_v2")

        from transformers import BartModel

        self.d_model = d_model

        # EEG encoder
        self.eeg_proj = EEGProjection(eeg_dim, d_model, dropout)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.eeg_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Load BART
        self.bart = BartModel.from_pretrained(bart_model_name)
        bart_d_model = self.bart.config.d_model

        # Bridge: project EEG encoder output to BART d_model
        self.encoder_bridge = nn.Linear(d_model, bart_d_model) if d_model != bart_d_model else nn.Identity()

        # Freeze BART encoder entirely (we use our EEG encoder)
        if freeze_bart_encoder:
            for param in self.bart.encoder.parameters():
                param.requires_grad = False

        # Freeze first N layers of BART decoder
        if freeze_bart_decoder_layers > 0:
            for i, layer in enumerate(self.bart.decoder.layers):
                if i < freeze_bart_decoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Output head (BART hidden → vocab)
        vocab_size = self.bart.config.vocab_size
        self.lm_head = nn.Linear(bart_d_model, vocab_size, bias=False)

        # Tie with BART's embedding weight
        self.lm_head.weight = self.bart.shared.weight

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"NEST_BART_v2: eeg_encoder ({d_model}d, {num_encoder_layers}L) + "
            f"BART-{bart_model_name.split('/')[-1]}, "
            f"trainable_params={n_params:,}"
        )

    def encode_eeg(
        self, eeg: torch.Tensor, eeg_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode EEG to cross-attention context.

        Returns:
            encoder_out: (batch, seq_len, bart_d_model)
            attention_mask: (batch, seq_len) 1=real, 0=padded
        """
        x = self.eeg_proj(eeg)
        x = self.pos_enc(x)

        key_padding_mask = None
        attention_mask = None
        if eeg_lengths is not None:
            max_len = eeg.size(1)
            mask = torch.arange(max_len, device=eeg_lengths.device).unsqueeze(0)
            key_padding_mask = mask >= eeg_lengths.unsqueeze(1)
            attention_mask = (~key_padding_mask).long()

        x = self.eeg_transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.encoder_bridge(x)
        return x, attention_mask

    def forward(
        self,
        eeg: torch.Tensor,
        eeg_lengths: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            eeg: (batch, eeg_seq_len, 840)
            eeg_lengths: (batch,) actual EEG word counts
            input_ids: (batch, text_seq_len) decoder input tokens
            attention_mask: (batch, text_seq_len) decoder attention mask
            labels: (batch, text_seq_len) target tokens (for loss)

        Returns:
            (loss, logits)  — loss is None if labels not provided
        """
        encoder_out, encoder_attention_mask = self.encode_eeg(eeg, eeg_lengths)

        if input_ids is None:
            # Inference mode: use BOS token
            batch_size = eeg.size(0)
            input_ids = torch.full(
                (batch_size, 1),
                self.bart.config.bos_token_id,
                dtype=torch.long,
                device=eeg.device,
            )

        # BART decoder with EEG encoder as cross-attention context
        decoder_outputs = self.bart.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=encoder_attention_mask,
        )

        hidden_states = decoder_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        eeg: torch.Tensor,
        eeg_lengths: Optional[torch.Tensor] = None,
        max_length: int = 64,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ) -> torch.Tensor:
        """
        Generate text tokens via beam search.

        Returns:
            token_ids: (batch, generated_len)
        """
        from transformers import GenerationConfig

        encoder_out, encoder_attention_mask = self.encode_eeg(eeg, eeg_lengths)

        # Use BART's built-in generation with our encoder output
        gen_config = GenerationConfig(
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            forced_bos_token_id=self.bart.config.bos_token_id,
            forced_eos_token_id=self.bart.config.eos_token_id,
            pad_token_id=self.bart.config.pad_token_id,
        )

        # We need to wrap this since BART's generate expects full model
        # Use manual beam search with decoder
        batch_size = eeg.size(0)
        bos_id = self.bart.config.bos_token_id
        eos_id = self.bart.config.eos_token_id
        pad_id = self.bart.config.pad_token_id

        # Simple greedy decoding (swap for beam search in production)
        input_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=eeg.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=eeg.device)

        for _ in range(max_length):
            decoder_outputs = self.bart.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_out,
                encoder_attention_mask=encoder_attention_mask,
            )
            logits = self.lm_head(decoder_outputs.last_hidden_state[:, -1, :])
            next_tokens = logits.argmax(dim=-1)
            next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            finished = finished | (next_tokens == eos_id)
            if finished.all():
                break

        return input_ids[:, 1:]  # Remove BOS


def build_model(
    model_type: str = "ctc",
    eeg_dim: int = EEG_DIM,
    **kwargs,
) -> nn.Module:
    """Factory function for NEST v2 models."""
    if model_type == "ctc":
        return NEST_CTC_v2(eeg_dim=eeg_dim, **kwargs)
    elif model_type == "bart":
        return NEST_BART_v2(eeg_dim=eeg_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== NEST_CTC_v2 ===")
    model = NEST_CTC_v2(d_model=512, num_encoder_layers=4, nhead=8)
    x = torch.randn(2, 20, 840)
    lengths = torch.tensor([20, 15])
    out = model(x, lengths)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")  # (2, 20, 29)

    print()
    print("=== NEST_BART_v2 ===")
    try:
        bart_model = NEST_BART_v2(d_model=512, num_encoder_layers=4)
        x = torch.randn(2, 20, 840)
        loss, logits = bart_model(x, labels=torch.randint(0, 50265, (2, 10)))
        print(f"Input: {x.shape}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Logits: {logits.shape}")
    except ImportError as e:
        print(f"BART requires transformers: {e}")
