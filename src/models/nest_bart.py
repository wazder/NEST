"""
NEST-BART: EEG-to-Text model combining Conformer encoder with BART decoder.

Two architectures:
  - NEST_ConformerBART: Conformer encoder + BART cross-attention decoder
  - NEST_ConformerCTC_Large: Conformer encoder + CTC head (no transformers dependency)
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.spatial_cnn import EEGNet
from src.models.temporal_encoder import ConformerEncoder

logger = logging.getLogger(__name__)

# Optional BART import
try:
    from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — NEST_ConformerBART unavailable")


class NEST_ConformerCTC_Large(nn.Module):
    """
    Large Conformer encoder + CTC head for EEG-to-text.
    Does not require the transformers library.

    Input:  (batch, n_channels, time)
    Output: log_probs (batch, time_reduced, vocab_size)
    """

    def __init__(
        self,
        n_channels: int = 105,
        vocab_size: int = 29,
        d_model: int = 768,
        num_layers: int = 12,
        nhead: int = 12,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        eegnet_filters: int = 16,
        eegnet_depth: int = 2,
        eegnet_pointwise: int = 64,
    ):
        super().__init__()

        self.eegnet = EEGNet(
            n_channels=n_channels,
            n_temporal_filters=eegnet_filters,
            depth_multiplier=eegnet_depth,
            n_pointwise_filters=eegnet_pointwise,
            dropout=dropout,
        )
        # EEGNet outputs (batch, eegnet_pointwise, T_reduced)
        # ConformerEncoder expects (batch, T, input_dim)
        self.conformer = ConformerEncoder(
            input_dim=eegnet_pointwise,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.ctc_head = nn.Linear(d_model, vocab_size)

        logger.info(
            f"NEST_ConformerCTC_Large: channels={n_channels}, "
            f"d_model={d_model}, layers={num_layers}, vocab={vocab_size}"
        )

    def encode_eeg(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        eeg_data: (batch, channels, time)
        returns:  (batch, T_reduced, d_model)
        """
        x = self.eegnet(eeg_data)           # (batch, 64, T_reduced)
        x = x.permute(0, 2, 1)             # (batch, T_reduced, 64)
        x = self.conformer(x)               # (batch, T_reduced, d_model)
        return x

    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Returns log_probs: (batch, time, vocab_size)
        """
        enc = self.encode_eeg(eeg_data)     # (batch, T, d_model)
        logits = self.ctc_head(enc)         # (batch, T, vocab_size)
        return F.log_softmax(logits, dim=-1)


class _EEGCrossAttnBridge(nn.Module):
    """Projects EEG encoder output to match BART d_model for cross-attention."""

    def __init__(self, eeg_dim: int, bart_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(eeg_dim, bart_dim),
            nn.LayerNorm(bart_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class NEST_ConformerBART(nn.Module):
    """
    Conformer EEG encoder + BART decoder with cross-attention.

    EEG encoder output is passed as encoder_hidden_states to BART decoder.
    First `freeze_bart_layers` encoder layers of BART are frozen.

    Input:  eeg_data (batch, channels, time)
            input_ids (batch, seq_len) — decoder input tokens (teacher forcing)
            labels    (batch, seq_len) — target token ids
    Output: loss (if labels given) or logits (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        n_channels: int = 105,
        freeze_bart_layers: int = 6,
        dropout: float = 0.1,
        eegnet_filters: int = 16,
        eegnet_depth: int = 2,
        eegnet_pointwise: int = 64,
        d_model: int = 768,
        num_conformer_layers: int = 12,
        nhead: int = 12,
        conv_kernel_size: int = 31,
        bart_model_name: str = "facebook/bart-base",
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required for NEST_ConformerBART. "
                "Install with: pip install transformers"
            )
        super().__init__()

        # Spatial feature extractor
        self.eegnet = EEGNet(
            n_channels=n_channels,
            n_temporal_filters=eegnet_filters,
            depth_multiplier=eegnet_depth,
            n_pointwise_filters=eegnet_pointwise,
            dropout=dropout,
        )

        # Temporal encoder
        self.conformer = ConformerEncoder(
            input_dim=eegnet_pointwise,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_conformer_layers,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # Load BART
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        bart_d_model = self.bart.config.d_model  # 768 for bart-base

        # Cross-attention bridge
        self.bridge = _EEGCrossAttnBridge(d_model, bart_d_model)

        # Freeze first N encoder layers of BART
        self._freeze_bart_layers(freeze_bart_layers)

        self.vocab_size = self.bart.config.vocab_size

        logger.info(
            f"NEST_ConformerBART: channels={n_channels}, "
            f"conformer_dim={d_model}, bart={bart_model_name}, "
            f"frozen_layers={freeze_bart_layers}"
        )

    def _freeze_bart_layers(self, n_freeze: int) -> None:
        # Freeze BART encoder entirely (we use our own EEG encoder)
        for param in self.bart.model.encoder.parameters():
            param.requires_grad = False

        # Freeze first n_freeze decoder layers
        decoder_layers = self.bart.model.decoder.layers
        for i, layer in enumerate(decoder_layers):
            if i < n_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def encode_eeg(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        eeg_data: (batch, channels, time)
        returns:  (batch, T_reduced, bart_d_model)
        """
        x = self.eegnet(eeg_data)       # (batch, 64, T_reduced)
        x = x.permute(0, 2, 1)         # (batch, T_reduced, 64)
        x = self.conformer(x)           # (batch, T_reduced, d_model)
        x = self.bridge(x)              # (batch, T_reduced, bart_d_model)
        return x

    def forward(
        self,
        eeg_data: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Training forward with teacher forcing.

        Returns loss if labels provided, else logits.
        """
        encoder_hidden_states = self.encode_eeg(eeg_data)
        # encoder_attention_mask: all ones (no padding in EEG encoder output here)
        B, T_enc, _ = encoder_hidden_states.shape
        encoder_attention_mask = torch.ones(B, T_enc, device=eeg_data.device, dtype=torch.long)

        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=(encoder_hidden_states,),
            # Pass encoder mask so cross-attention works correctly
            # BartForConditionalGeneration accepts encoder_outputs as tuple:
            # (last_hidden_state, ...) and uses it as cross-attn keys/values
            labels=labels,
        )

        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs.logits

    def generate(
        self,
        eeg_data: torch.Tensor,
        max_length: int = 50,
        num_beams: int = 5,
        **kwargs,
    ) -> torch.Tensor:
        """
        Beam search generation from EEG input.
        Returns token ids: (batch, seq_len)
        """
        encoder_hidden_states = self.encode_eeg(eeg_data)
        B, T_enc, _ = encoder_hidden_states.shape
        encoder_attention_mask = torch.ones(B, T_enc, device=eeg_data.device, dtype=torch.long)

        generated = self.bart.generate(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs,
        )
        return generated
