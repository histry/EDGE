from types import SimpleNamespace

import numpy as np
import pytest
import torch

from infer_controlled import (
    align_audio_features,
    build_keyframe_constraint,
    load_keyframe,
    normalize_trajectory,
)
from dataset.dance_dataset import DunhuangDataset


class DummyNormalizer:
    def __init__(self):
        self.mean = np.zeros(151, dtype=np.float32)
        self.std = np.ones(151, dtype=np.float32)
        self.mean[4] = 10.0
        self.mean[6] = -2.0
        self.std[4] = 2.0
        self.std[6] = 4.0

    def normalize(self, x):
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def unnormalize(self, x):
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device)
        return x * std + mean


def test_keyframe_must_be_151d(tmp_path):
    bad = tmp_path / "bad_pose.npy"
    np.save(bad, np.zeros(150, dtype=np.float32))

    with pytest.raises(ValueError, match="Expected 151-D keyframe"):
        load_keyframe(str(bad), normalizer=None, keyframe_space="normalized")


def test_physical_keyframe_is_normalized(tmp_path):
    pose = np.zeros(151, dtype=np.float32)
    pose[4] = 12.0
    pose[6] = 6.0

    path = tmp_path / "pose.npy"
    np.save(path, pose)

    out = load_keyframe(
        str(path),
        normalizer=DummyNormalizer(),
        keyframe_space="physical",
    )

    assert out.shape == (151,)
    assert np.isclose(out[4], 1.0)      # (12 - 10) / 2
    assert np.isclose(out[6], 2.0)      # (6 - -2) / 4


def test_trajectory_normalization_uses_root_xz_stats():
    traj = np.asarray(
        [
            [10.0, -2.0],
            [12.0, 6.0],
        ],
        dtype=np.float32,
    )

    out = normalize_trajectory(traj, DummyNormalizer())

    expected = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_audio_feature_alignment_requires_803_dim():
    good = np.zeros((10, 803), dtype=np.float32)
    out = align_audio_features(good, seq_len=150, expected_dim=803)

    assert out.shape == (150, 803)

    bad = np.zeros((10, 802), dtype=np.float32)
    with pytest.raises(ValueError, match="Audio dim mismatch"):
        align_audio_features(bad, seq_len=150, expected_dim=803)


def test_start_end_keyframe_constraint_exactness(tmp_path):
    start = np.zeros(151, dtype=np.float32)
    end = np.ones(151, dtype=np.float32)

    start_path = tmp_path / "start.npy"
    end_path = tmp_path / "end.npy"
    np.save(start_path, start)
    np.save(end_path, end)

    args = SimpleNamespace(
        start_pose=str(start_path),
        end_pose=str(end_path),
        mid_poses="",
        mid_pose_frames="",
        keyframe_space="normalized",
        keyframe_width=1,
        preserve_keyframe_root_xz=True,
    )

    constraint = build_keyframe_constraint(
        args=args,
        seq_len=150,
        normalizer=None,
        traj_norm=None,
    )

    mask = constraint["mask"]
    value = constraint["value"]

    assert mask.shape == (1, 150, 1)
    assert value.shape == (1, 150, 151)

    assert mask[0, 0, 0].item() == 1.0
    assert mask[0, -1, 0].item() == 1.0

    np.testing.assert_allclose(value[0, 0].numpy(), start, atol=1e-6)
    np.testing.assert_allclose(value[0, -1].numpy(), end, atol=1e-6)


def make_dataset_stub(audio_pairing_mode):
    ds = DunhuangDataset.__new__(DunhuangDataset)
    ds.audio_pairing_mode = audio_pairing_mode
    ds.audio_sample_mode = "random"
    ds.paired_audio_missing_policy = "error"
    ds.motion_window_ids = ["window_000"]
    ds.weak_pair_map = {}
    ds.proxy_audios = [np.ones((5, 803), dtype=np.float32)]
    return ds


def test_audio_pairing_mode_none_returns_no_audio():
    ds = make_dataset_stub("none")
    assert ds._sample_audio_feature(0) is None


def test_audio_pairing_mode_paired_requires_pair():
    ds = make_dataset_stub("paired")

    with pytest.raises(RuntimeError, match="requires a paired audio candidate"):
        ds._sample_audio_feature(0)


def test_audio_pairing_mode_proxy_can_use_proxy_audio():
    ds = make_dataset_stub("proxy")
    audio = ds._sample_audio_feature(0)

    assert audio is not None
    assert audio.shape[-1] == 803