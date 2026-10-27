"""NEST_BART_v2 starts its EEG encoder output at the scale of BART's encoder output."""

from pathlib import Path

import torch

from src.models.nest_v2 import NEST_BART_v2


def save_tiny_bart(path: Path, d_model: int, final_norm_weight: float) -> str:
    from transformers import BartConfig, BartModel

    config = BartConfig(
        vocab_size=64, d_model=d_model, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=64,
    )
    bart = BartModel(config)
    with torch.no_grad():
        bart.encoder.layers[-1].final_layer_norm.weight.fill_(final_norm_weight)
    bart.save_pretrained(path)
    return str(path)


def test_encoder_output_starts_at_bart_encoder_scale(tmp_path: Path) -> None:
    model = NEST_BART_v2(
        d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32,
        bart_model_name=save_tiny_bart(tmp_path / "bart", 16, 0.2),
    ).eval()
    out, _ = model.encode_eeg(torch.randn(2, 9, 840))
    assert torch.allclose(model.eeg_transformer.norm.weight, torch.full((16,), 0.2))
    assert abs(out.std().item() - 0.2) < 0.05


def test_bridge_case_keeps_default_norm(tmp_path: Path) -> None:
    model = NEST_BART_v2(
        d_model=32, nhead=2, num_encoder_layers=1, dim_feedforward=64,
        bart_model_name=save_tiny_bart(tmp_path / "bart", 16, 0.2),
    )
    assert torch.allclose(model.eeg_transformer.norm.weight, torch.ones(32))
