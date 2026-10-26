"""CTC feasibility of NEST_CTC_v2 on realistic ZuCo lengths.

CTC can only align a target of U characters (plus one blank between each
pair of repeated characters) to at least that many output frames. ZuCo has
one EEG step per fixated word, about 9 target characters per step at the
median, so one frame per word step gives an infinite CTC loss, which
zero_infinity turns into 0 with zero gradient.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.data.zuco_pickle_dataset import collate_word_sequence_ctc  # noqa: E402
from src.models.nest_v2 import CTC_OUTPUT_FRAMES, NEST_CTC_v2  # noqa: E402

import train_nest_v2 as tnv2  # noqa: E402
from eval_noise_control import build_model_from_checkpoint  # noqa: E402

pytestmark = pytest.mark.unit

TINY = dict(d_model=32, nhead=2, num_encoder_layers=1, dim_feedforward=64)
CPU = torch.device("cpu")

# (text, fixated words): a fully fixated sentence (T=12, U=83, 6 repeats) and
# one where readers skipped words, as in ZuCo (T=11, U=94)
SENTENCES = [
    ("the committee approved the proposal after a long and difficult discussion yesterday", 12),
    ("she said that the old bridge over the river would be closed for repairs until the end of march", 11),
]


def realistic_batch() -> dict:
    rng = np.random.default_rng(0)
    samples = [
        {"eeg": rng.standard_normal((n_fixated, 840)).astype(np.float32), "eeg_len": n_fixated, "text": text}
        for text, n_fixated in SENTENCES
    ]
    return collate_word_sequence_ctc(samples)


class GradNormSGD(torch.optim.SGD):
    """SGD with lr=0 that records the gradient norm at every step (after
    train_epoch_ctc clips it to 1.0)."""

    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.grad_norms = []

    def step(self, closure=None):
        grads = [p.grad.norm() for group in self.param_groups for p in group["params"] if p.grad is not None]
        self.grad_norms.append(torch.stack(grads).norm().item() if grads else 0.0)
        return super().step(closure)


def train_step(model, batch, optimizer) -> float:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    return tnv2.train_epoch_ctc(
        model, [batch], optimizer, scheduler, None, CPU, grad_accum=1, fp16=False, epoch=0,
    )


def test_realistic_targets_outnumber_word_steps():
    batch = realistic_batch()
    assert batch["eeg_lengths"].tolist() == [12, 11]
    assert batch["target_lengths"].tolist() == [83, 94]


def test_training_step_has_nonzero_loss_and_gradient():
    torch.manual_seed(0)
    model = NEST_CTC_v2(**TINY)
    optimizer = GradNormSGD(model.parameters())
    loss = train_step(model, realistic_batch(), optimizer)
    assert math.isfinite(loss) and loss > 0, f"CTC loss is {loss}: no feasible alignment"
    assert optimizer.grad_norms[0] > 0, "CTC gradient is zero: the model cannot learn"


def test_loss_decreases_when_fitting_one_batch():
    torch.manual_seed(0)
    model = NEST_CTC_v2(**TINY)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = realistic_batch()
    losses = [train_step(model, batch, optimizer) for _ in range(20)]
    assert losses[0] > 0, f"CTC loss is {losses[0]}: no feasible alignment"
    assert losses[-1] < 0.8 * losses[0], losses


def test_eval_loss_is_finite_and_nonzero():
    torch.manual_seed(0)
    metrics = tnv2.eval_ctc(NEST_CTC_v2(**TINY), [realistic_batch()], CPU, fp16=False)
    assert math.isfinite(metrics["loss"]) and metrics["loss"] > 0, metrics["loss"]
    assert len(metrics["hypotheses"]) == 2


def test_default_frames_cover_longest_zuco_target():
    # longest target on the converted ZuCo 1.0 pickles (SR, NR, TSR):
    # 388 = characters + repeated-character pairs
    model = NEST_CTC_v2(**TINY)
    assert model.output_frames == CTC_OUTPUT_FRAMES >= 388
    lengths = torch.tensor([12, 3])
    assert model(torch.randn(2, 12, 840), lengths).shape == (2, CTC_OUTPUT_FRAMES, 29)
    assert model.output_lengths(lengths).tolist() == [CTC_OUTPUT_FRAMES] * 2


def test_frames_keep_word_order_and_skip_padding():
    model = NEST_CTC_v2(output_frames=20, **TINY)
    words = torch.arange(6.0).view(1, 6, 1).expand(2, 6, 3)  # every feature = word index
    frames = model.expand_to_frames(words, torch.tensor([6, 4]))
    assert frames.shape == (2, 20, 3)
    for row, n_words in zip(frames[..., 0].long().tolist(), [6, 4]):
        assert row == sorted(row) and set(row) == set(range(n_words))
        counts = np.bincount(row)
        assert counts.max() - counts.min() <= 1  # frames shared out evenly


def test_one_frame_per_word_without_output_frames():
    model = NEST_CTC_v2(output_frames=None, **TINY)
    lengths = torch.tensor([12, 7])
    assert model(torch.randn(2, 12, 840), lengths).shape == (2, 12, 29)
    assert torch.equal(model.output_lengths(lengths), lengths)


def test_output_frames_add_no_parameters():
    with_frames, without = NEST_CTC_v2(**TINY), NEST_CTC_v2(output_frames=None, **TINY)
    assert {k: v.shape for k, v in with_frames.state_dict().items()} == {
        k: v.shape for k, v in without.state_dict().items()
    }


def test_output_frames_beyond_positional_encoding_raise():
    with pytest.raises(ValueError, match="max_seq_len"):
        NEST_CTC_v2(output_frames=600, max_seq_len=512, **TINY)


def test_greedy_decode_ignores_padded_frames():
    ids = [3, 1, 1, 0, 20, 26, 26]  # c a a <blank> t | z z (padding)
    log_probs = torch.nn.functional.one_hot(torch.tensor([ids]), 29).float().log()
    assert tnv2.ctc_greedy_decode(log_probs) == ["catz"]
    assert tnv2.ctc_greedy_decode(log_probs, torch.tensor([5])) == ["cat"]


@pytest.mark.parametrize("output_frames", [CTC_OUTPUT_FRAMES, None])
def test_checkpoint_rebuild_keeps_output_frames(output_frames):
    config = {"model": "ctc", "d_model": 32, "num_layers": 1, "nhead": 2, "gaze_dim": None}
    if output_frames is not None:  # checkpoints from before output_frames lack the key
        config["output_frames"] = output_frames
    model = NEST_CTC_v2(d_model=32, nhead=2, num_encoder_layers=1, dim_feedforward=128,
                        output_frames=output_frames).eval()
    rebuilt = build_model_from_checkpoint({"model": model.state_dict(), "config": config}, "ctc", CPU).eval()
    assert rebuilt.output_frames == output_frames
    x, lengths = torch.randn(2, 9, 840), torch.tensor([9, 5])
    with torch.no_grad():
        assert torch.allclose(model(x, lengths), rebuilt(x, lengths))
