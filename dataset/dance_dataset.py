import glob
import os
import pickle
import random
import sys
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from torch.utils.data import Dataset

from dataset.quaternion import ax_to_6v
from dataset.preprocess import Normalizer, vectorize_many
from vis import SMPLSkeleton

# ================= 修复 numpy._core 报错的补丁 =================
import numpy.core
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
sys.modules['numpy._core.umath'] = numpy.core.umath
# ===============================================================

class AISTPPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool,
        backup_path: str = "data/backup",
        feature_type: str = "hybrid",
        normalizer: Any = None,
        data_len: int = -1,
        seq_len: int = 150,  # 显式接收 seq_len 确保截断维度统一
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path
        self.raw_fps = 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type
        self.seq_len = seq_len

        self.normalizer = normalizer
        self.data_len = data_len

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        if not train and normalizer is not None:
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
            )
            
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset and applying strict shape clipping...")
            data = self.load_aistpp()
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print(
            f"Loaded {self.name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )

        pose_input = self.process_dataset(data["pos"], data["q"])
        self.data = {
            "pose": pose_input,
            "filenames": data["filenames"],
            "wavs": data["wavs"],
        }
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_)).float()
        pose = self.data["pose"][idx]

        # 【核心安全检查】：强制补齐或截断音频特征，确保长度完全等于 seq_len
        if feature.shape[0] > self.seq_len:
            feature = feature[:self.seq_len]
        elif feature.shape[0] < self.seq_len:
            pad_len = self.seq_len - feature.shape[0]
            feature = torch.cat([feature, torch.zeros(pad_len, feature.shape[1])], dim=0)

        # 封装为字典以完美兼容 EDGE 模型的特征提取逻辑
        cond = {"audio": feature}
        return pose, cond, filename_, self.data["wavs"][idx]

    # def load_aistpp(self):
    #     split_data_path = os.path.join(
    #         self.data_path, "train" if self.train else "test"
    #     )

    #     motion_path = os.path.join(split_data_path, "motions_sliced")
    #     sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
    #     wav_path = os.path.join(split_data_path, "wavs_sliced")
        
    #     motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
    #     features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
    #     wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

    #     all_pos = []
    #     all_q = []
    #     all_names = []
    #     all_wavs = []
    #     assert len(motions) == len(features), f"Mismatch: {len(motions)} motions vs {len(features)} audio features."
        
    #     required_len = self.seq_len * self.data_stride

    #     for motion, feature, wav in zip(motions, features, wavs):
    #         m_name = os.path.splitext(os.path.basename(motion))[0]
    #         f_name = os.path.splitext(os.path.basename(feature))[0]
    #         w_name = os.path.splitext(os.path.basename(wav))[0]
    #         assert m_name == f_name == w_name, str((motion, feature, wav))
            
    #         data = pickle.load(open(motion, "rb"))
    #         pos = data["pos"]
    #         q = data["q"]
            
    #         # 【修复一维对象数组异常】：丢弃长度不达标的残次切片，截断过长的切片
    #         if pos.shape[0] < required_len:
    #             continue

    #         all_pos.append(pos[:required_len])
    #         all_q.append(q[:required_len])
    #         all_names.append(feature)
    #         all_wavs.append(wav)

    #     # 此时数据形状绝对整齐，np.array 不会坍塌为 1D object
    #     all_pos = np.array(all_pos)  # N x required_len x 3
    #     all_q = np.array(all_q)      # N x required_len x (joint * 3)
        
    #     all_pos = all_pos[:, :: self.data_stride, :]
    #     all_q = all_q[:, :: self.data_stride, :]
        
    #     data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs}
    #     return data
    def load_aistpp(self):
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        motion_path = os.path.join(split_data_path, "motions_sliced")
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, "wavs_sliced")
        
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        all_pos = []
        all_q = []
        all_names = []
        all_wavs = []
        
        # ==================== 👇 修复逻辑：安全取交集匹配 👇 ====================
        motion_dict = {os.path.splitext(os.path.basename(m))[0]: m for m in motions}
        feature_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in features}
        wav_dict = {os.path.splitext(os.path.basename(w))[0]: w for w in wavs}

        common_keys = sorted(list(set(motion_dict.keys()) & set(feature_dict.keys()) & set(wav_dict.keys())))
        
        print(f"🧩 正在取交集: 匹配到 {len(common_keys)} 个完整的音视频切片对 (动作库:{len(motions)} / 音频库:{len(features)})")

        required_len = self.seq_len * self.data_stride

        for key in common_keys:
            motion = motion_dict[key]
            feature = feature_dict[key]
            wav = wav_dict[key]
            
            data = pickle.load(open(motion, "rb"))
            pos = data["pos"]
            q = data["q"]
            
            # 丢弃长度不达标的残次切片，截断过长的切片
            if pos.shape[0] < required_len:
                continue

            all_pos.append(pos[:required_len])
            all_q.append(q[:required_len])
            all_names.append(feature)
            all_wavs.append(wav)
        # =========================================================================

        # 此时数据形状绝对整齐，np.array 不会坍塌为 1D object
        all_pos = np.array(all_pos)  # N x required_len x 3
        all_q = np.array(all_q)      # N x required_len x (joint * 3)
        
        all_pos = all_pos[:, :: self.data_stride, :]
        all_q = all_q[:, :: self.data_stride, :]
        
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs}
        return data

    def process_dataset(self, root_pos, local_q):
        smpl = SMPLSkeleton()
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))

        # root_q = local_q[:, :, :1, :]
        # root_q_quat = axis_angle_to_quaternion(root_q)
        # rotation = torch.Tensor([0.7071068, 0.7071068, 0, 0])
        # root_q_quat = quaternion_multiply(rotation, root_q_quat)
        # root_q = quaternion_to_axis_angle(root_q_quat)
        # local_q[:, :, :1, :] = root_q

        # pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        # root_pos = pos_rotation.transform_points(root_pos)

        positions = smpl.forward(local_q, root_pos)
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)

        local_q = ax_to_6v(local_q)

        l = [contacts, root_pos, local_q]
        global_pose_vec_input = vectorize_many(l).float().detach()

        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
            
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)
        assert not torch.isnan(global_pose_vec_input).any()

        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        return global_pose_vec_input


class OrderedMusicDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool = False,
        feature_type: str = "hybrid",
        data_name: str = "aist",
    ):
        self.data_path = data_path
        self.data_fps = 30
        self.feature_type = feature_type
        self.test_list = set(
            [
                "mLH4", "mKR2", "mBR0", "mLO2",
                "mJB5", "mWA0", "mJS3", "mMH3",
                "mHO5", "mPO1",
            ]
        )
        self.train = train
        self.data_name = data_name
        if self.data_name != "aist":
            self.train = True
        self.data = self.load_music()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None

    def get_batch(self, batch_size, idx=None):
        key = random.choice(self.keys) if idx is None else self.keys[idx]
        seq = self.data[key]
        if len(seq) <= batch_size:
            seq_slice = seq
        else:
            max_start = len(seq) - batch_size
            start = random.randint(0, max_start)
            seq_slice = seq[start : start + batch_size]

        filenames = [os.path.join(self.music_path, x + ".npy") for x in seq_slice]
        features = np.array([np.load(x) for x in filenames])
        return torch.Tensor(features), seq_slice

    def load_music(self):
        split_data_path = os.path.join(self.data_path)
        if self.feature_type == "baseline":
            feat_dir = f"{self.data_name}_baseline_feats"
        elif self.feature_type == "hybrid":
            feat_dir = f"{self.data_name}_hybrid_feats"
        else:
            feat_dir = f"{self.data_name}_juke_feats/juke_66"
            
        music_path = os.path.join(split_data_path, feat_dir)
        self.music_path = music_path
        
        all_names = []
        key_func = lambda x: int(x.split("_")[-1].split("e")[-1])

        def stringintcmp(a, b):
            aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
            ka, kb = key_func(a), key_func(b)
            if aa < bb: return -1
            if aa > bb: return 1
            if ka < kb: return -1
            if ka > kb: return 1
            return 0

        for features in glob.glob(os.path.join(music_path, "*.npy")):
            fname = os.path.splitext(os.path.basename(features))[0]
            all_names.append(fname)
            
        all_names = sorted(all_names, key=cmp_to_key(stringintcmp))
        data_dict = {}
        for name in all_names:
            k = "".join(name.split("_")[:-1])
            if (self.train and k in self.test_list) or (
                (not self.train) and k not in self.test_list
            ):
                continue
            data_dict[k] = data_dict.get(k, []) + [name]
            
        self.keys = sorted(list(data_dict.keys()))
        return data_dict


class DummyNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()
    def normalize(self, x):
        return (x - self.mean) / self.std
    def unnormalize(self, x):
        if isinstance(x, torch.Tensor):
            return x * self.std.to(x.device) + self.mean.to(x.device)
        return x * self.std.numpy() + self.mean.numpy()


class DunhuangDataset(Dataset):
    def __init__(
        self,
        data_path,
        seq_len=150,
        audio_dim=803,
        overlap=0.5,
        normalizer=None,
        return_traj=False,
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.audio_dim = audio_dim
        self.return_traj = return_traj

        search_path = os.path.join(data_path, "processed", "*.pkl")
        self.pkl_files = glob.glob(search_path)

        self.motions = []
        self.trajs = []

        if not self.pkl_files:
            print("Warning: No PKL files found.")

        for f in self.pkl_files:
            data = pickle.load(open(f, "rb"))
            pos = data["pos"]
            q = data["q"]
            motion = np.concatenate([pos, q], axis=-1)

            traj_xy = pos[:, [0, 2]]
            step = max(1, int(seq_len * (1 - overlap)))
            num_frames = motion.shape[0]
            if num_frames < seq_len:
                continue
            for start in range(0, num_frames - seq_len + 1, step):
                self.motions.append(motion[start : start + seq_len])
                self.trajs.append(traj_xy[start : start + seq_len])

        if len(self.motions) > 0:
            self.motions = np.array(self.motions, dtype=np.float32)
            self.trajs = np.array(self.trajs, dtype=np.float32)

            if normalizer is None:
                mean = np.mean(self.motions, axis=(0, 1), keepdims=True)
                std = np.std(self.motions, axis=(0, 1), keepdims=True) + 1e-5
                self.normalizer = DummyNormalizer(mean, std)
            else:
                self.normalizer = normalizer

            mean_np = self.normalizer.mean.numpy()
            std_np = self.normalizer.std.numpy()
            self.motions = (self.motions - mean_np) / std_np
        else:
            self.normalizer = DummyNormalizer(
                np.zeros((1, 1, 381), dtype=np.float32),
                np.ones((1, 1, 381), dtype=np.float32),
            )
            self.trajs = np.zeros((0, seq_len, 2), dtype=np.float32)

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion = torch.from_numpy(self.motions[idx])
        dummy_audio = torch.zeros((self.seq_len, self.audio_dim), dtype=torch.float32)
        if self.return_traj:
            traj = torch.from_numpy(self.trajs[idx])
            cond = {
                "audio": dummy_audio,
                "trajectory": traj,
            }
        else:
            cond = dummy_audio
        return motion, cond, f"dunhuang_motion_{idx}", "dummy_audio"