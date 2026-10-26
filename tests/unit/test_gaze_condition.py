"""Unit tests for the gaze-only and noise input conditions (synthetic data only)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.data.zuco_pickle_dataset import (  # noqa: E402
    GAZE_DIM,
    ZuCoWordSequenceDataset,
    _get_word_gaze,
    collate_word_sequence_ctc,
    get_subject_split,
)
from src.models.nest_v2 import NEST_BART_v2, NEST_CTC_v2  # noqa: E402
from tests.synthetic_zuco import write_synthetic_zuco  # noqa: E402

import train_nest_v2 as tnv2  # noqa: E402

pytestmark = pytest.mark.unit

TINY = dict(d_model=32, nhead=2, num_encoder_layers=1, dim_feedforward=64)


@pytest.fixture
def zuco_dir(tmp_path: Path) -> Path:
    write_synthetic_zuco(tmp_path)
    return tmp_path


def load(zuco_dir: Path, **kwargs) -> ZuCoWordSequenceDataset:
    return ZuCoWordSequenceDataset(str(zuco_dir), tasks=["task1-SR"], **kwargs)


class TestDataset:
    def test_word_gaze_order(self):
        word = {"nFixations": 2, "FFD": 107.0, "GD": 120.0, "TRT": 300.0, "GPT": 150.0}
        assert _get_word_gaze(word).tolist() == [2.0, 107.0, 120.0, 300.0, 150.0]

    def test_word_gaze_rejects_old_pickles(self):
        with pytest.raises(KeyError, match="zuco_mat_to_pickle"):
            _get_word_gaze({"nFixations": 2})

    def test_default_is_eeg_only(self, zuco_dir):
        ds = load(zuco_dir)
        sample = ds[0]
        assert ds.input_modes == ("eeg",)
        assert "gaze" not in sample and "noise" not in sample
        assert sample["eeg"].shape == (sample["eeg_len"], 840)

    def test_conditions_share_length(self, zuco_dir):
        ds = load(zuco_dir, input_modes=("eeg", "gaze", "noise"))
        assert len(ds) == len(load(zuco_dir))  # same samples as the default
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["gaze"].shape == (sample["eeg_len"], GAZE_DIM)
            assert sample["noise"].shape == sample["eeg"].shape

    def test_noise_is_fixed_per_sample_and_matches_stats(self, zuco_dir):
        ds = load(zuco_dir, input_modes=("eeg", "noise"))
        assert np.array_equal(ds[0]["noise"], ds[0]["noise"])
        n = min(ds[0]["eeg_len"], ds[1]["eeg_len"])
        assert not np.array_equal(ds[0]["noise"][:n], ds[1]["noise"][:n])
        eeg_words = np.concatenate([s["eeg"] for s in ds.samples])
        assert np.allclose(ds.noise_mean, eeg_words.mean(axis=0), atol=1e-5)
        assert np.allclose(ds.noise_std, eeg_words.std(axis=0), atol=1e-5)
        noise_words = np.concatenate([ds[i]["noise"] for i in range(len(ds))])
        assert abs(noise_words.std() - eeg_words.std()) < 0.1

    def test_noise_differs_between_subjects_at_same_index(self, zuco_dir):
        train = load(zuco_dir, subjects=["S01"], input_modes=("noise",))
        test = load(zuco_dir, subjects=["S05"], input_modes=("noise",))
        assert train[0]["text"] == test[0]["text"]  # same sentence, different subject

        def standardized(ds):
            return (ds[0]["noise"] - ds.noise_mean) / ds.noise_std

        assert not np.allclose(standardized(train), standardized(test))

    def test_load_datasets_does_not_drop_tasks_without_gaze(self, tmp_path):
        write_synthetic_zuco(tmp_path, task="task1-SR")
        write_synthetic_zuco(tmp_path, task="task2-NR", with_gaze=False)
        tnv2.load_datasets(str(tmp_path), ["task1-SR", "task2-NR"])  # EEG only: fine
        with pytest.raises(KeyError):
            tnv2.load_datasets(str(tmp_path), ["task1-SR", "task2-NR"], input_modes=("eeg", "gaze"))

    def test_unknown_mode_raises(self, zuco_dir):
        with pytest.raises(ValueError):
            load(zuco_dir, input_modes=("eeg", "audio"))

    def test_gaze_mode_needs_gaze_pickle(self, tmp_path):
        write_synthetic_zuco(tmp_path, with_gaze=False)
        load(tmp_path)  # EEG-only use of an old-style pickle still works
        with pytest.raises(KeyError):
            load(tmp_path, input_modes=("gaze",))

    @pytest.mark.parametrize("collate", [collate_word_sequence_ctc, tnv2.collate_bart_v2])
    def test_collate_pads_all_conditions_alike(self, zuco_dir, collate):
        ds = load(zuco_dir, input_modes=("eeg", "gaze", "noise"))
        batch = collate([ds[i] for i in range(4)])
        lengths = batch["eeg_lengths"]
        max_len = int(lengths.max())
        assert batch["eeg"].shape == (4, max_len, 840)
        assert batch["gaze"].shape == (4, max_len, GAZE_DIM)
        assert batch["noise"].shape == (4, max_len, 840)
        for i, n in enumerate(lengths.tolist()):
            assert torch.all(batch["gaze"][i, n:] == 0) and torch.all(batch["noise"][i, n:] == 0)
            assert torch.all(batch["gaze"][i, :n, 1] > 0)  # FFD of real words


def test_subject_split_is_8_2_2_on_zuco_subjects(tmp_path):
    subjects = ["ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH"]
    write_synthetic_zuco(tmp_path, subjects=subjects, n_sentences=1)
    train, val, test = get_subject_split(str(tmp_path), tasks=["task1-SR"])
    assert test == ["ZMG", "ZPH"]
    assert val == ["ZKH", "ZKW"]
    assert train == subjects[:8]


class TestModel:
    def test_default_ctc_unchanged(self):
        model = NEST_CTC_v2()
        assert sum(p.numel() for p in model.parameters()) == 45_310_637
        assert model.gaze_proj is None
        assert not any(k.startswith("gaze_proj") for k in model.state_dict())

    def test_gaze_forward_shape(self):
        model = NEST_CTC_v2(gaze_dim=GAZE_DIM, **TINY)
        out = model(torch.rand(2, 30, GAZE_DIM) * 300, torch.tensor([30, 25]), input_mode="gaze")
        assert out.shape == (2, 30, 29)

    def test_gaze_respects_padding_mask(self):
        model = NEST_CTC_v2(gaze_dim=GAZE_DIM, **TINY).eval()
        gaze = torch.rand(2, 10, GAZE_DIM) * 300
        lengths = torch.tensor([10, 6])
        changed = gaze.clone()
        changed[1, 6:] = 999.0  # only padded positions differ
        with torch.no_grad():
            a = model(gaze, lengths, input_mode="gaze")
            b = model(changed, lengths, input_mode="gaze")
        assert torch.allclose(a[1, :6], b[1, :6], atol=1e-5)

    def test_modes_route_to_their_projection(self):
        model = NEST_CTC_v2(gaze_dim=GAZE_DIM, **TINY)
        lengths = torch.tensor([8, 8])
        for mode, x in [("eeg", torch.randn(2, 8, 840)), ("noise", torch.randn(2, 8, 840)),
                        ("gaze", torch.rand(2, 8, GAZE_DIM) * 300)]:
            model.zero_grad()
            model(x, lengths, input_mode=mode).sum().backward()
            eeg_grad = model.eeg_proj.proj[0].weight.grad
            gaze_grad = model.gaze_proj.proj[0].weight.grad
            if mode == "gaze":
                assert eeg_grad is None or torch.all(eeg_grad == 0)
                assert gaze_grad is not None and gaze_grad.abs().sum() > 0
            else:
                assert eeg_grad is not None and eeg_grad.abs().sum() > 0
                assert gaze_grad is None or torch.all(gaze_grad == 0)

    def test_gaze_without_gaze_proj_raises(self):
        model = NEST_CTC_v2(**TINY)
        with pytest.raises(ValueError, match="gaze_dim"):
            model(torch.rand(1, 4, GAZE_DIM), input_mode="gaze")
        with pytest.raises(ValueError, match="Unknown input_mode"):
            model(torch.randn(1, 4, 840), input_mode="audio")

    def test_bart_gaze_forward_and_generate(self, tmp_path):
        from transformers import BartConfig, BartModel

        config = BartConfig(
            vocab_size=64, d_model=16, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=64,
        )
        BartModel(config).save_pretrained(tmp_path / "tiny-bart")
        model = NEST_BART_v2(
            d_model=32, nhead=2, num_encoder_layers=1, dim_feedforward=64,
            bart_model_name=str(tmp_path / "tiny-bart"), freeze_bart_decoder_layers=0,
            gaze_dim=GAZE_DIM,
        )
        gaze = torch.rand(2, 7, GAZE_DIM) * 300
        lengths = torch.tensor([7, 5])
        labels = torch.randint(3, 64, (2, 6))
        loss, logits = model(gaze, lengths, input_ids=labels, labels=labels, input_mode="gaze")
        assert torch.isfinite(loss) and logits.shape == (2, 6, 64)
        generated = model.generate(gaze, lengths, max_length=5, input_mode="gaze")
        assert generated.shape[0] == 2 and generated.shape[1] <= 5


class RecordingModel(torch.nn.Module):
    """Stands in for NEST_CTC_v2 and records the input mode of each call."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, x, lengths, input_mode="eeg"):
        self.calls.append((input_mode, x.shape[-1]))
        return torch.zeros(x.shape[0], x.shape[1], 29).add(self.weight).log_softmax(-1)


def test_training_alternates_input_modes(zuco_dir):
    ds = load(zuco_dir, input_modes=("eeg", "gaze"))
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_word_sequence_ctc)
    model = RecordingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    tnv2.train_epoch_ctc(
        model, loader, optimizer, scheduler, None, torch.device("cpu"),
        grad_accum=1, fp16=False, epoch=0, input_modes=["eeg", "gaze"],
    )
    modes = [mode for mode, _ in model.calls]
    assert modes[:4] == ["eeg", "gaze", "eeg", "gaze"]
    assert all(dim == (840 if mode == "eeg" else GAZE_DIM) for mode, dim in model.calls)
