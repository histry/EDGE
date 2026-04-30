import glob
import os
import pickle
import random
import sys
import csv
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Dict, Optional

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
        seq_len: int = 150,
        include_contacts: bool = True,
        force_reload: bool = False,
        return_traj: bool = False,
    ):
        self.return_traj = return_traj
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

        if feature.shape[0] > self.seq_len:
            feature = feature[:self.seq_len]
        elif feature.shape[0] < self.seq_len:
            # 🚀 修复：废弃 Cyclic Padding，改用 Reflect (反射/镜像) Padding
            # 反射填充能保证边界连续性，消除频率和强度的突变，防止动作抽搐
            # 🚀 修复：安全的 Reflect Padding (完美支持 pad_len 远大于原序列长度的边界情况)
            pad_len = self.seq_len - feature.shape[0]
            feature_np = feature.numpy()
            
            # ✨ 核心改进：自适应混合填充 (Adaptive Hybrid Padding)
            # 解决极短音频反射导致的信号学高频震荡 (High-frequency Oscillation)
            MIN_REFLECT_LEN = 15  # 阈值：15帧(0.5秒)。小于此长度认为不具备音乐节拍周期性
            
            if feature_np.shape[0] == 1:
                feature_np = np.repeat(feature_np, self.seq_len, axis=0)
            elif feature_np.shape[0] < MIN_REFLECT_LEN:
                # 极短序列使用 'edge' (末端值复制/零阶保持)，避免来回反弹产生的虚假高频噪音
                feature_np = np.pad(feature_np, ((0, pad_len), (0, 0)), mode='edge')
            else:
                # 具有有效音乐短语的长序列，保留 'reflect' 以维持节拍的自然起伏
                while pad_len > 0:
                    current_pad = min(pad_len, feature_np.shape[0] - 1)
                    feature_np = np.pad(feature_np, ((0, current_pad), (0, 0)), mode='reflect')
                    pad_len -= current_pad
                    
            feature = torch.from_numpy(feature_np).float()

        # AIST++ slices are true audio-motion pairs, so mark them as paired.
        # This keeps GaussianDiffusion.disable_unpaired_audio_condition compatible
        # with both AIST++ and Dunhuang proxy/no-audio modes.
        if feature.shape[-1] > 768:
            onset = feature[:, 768:769].clamp_min(0.0)
            onset = onset / onset.amax().clamp_min(1e-6)
        else:
            onset = torch.zeros((self.seq_len, 1), dtype=torch.float32)

        cond = {
            "audio": feature,
            "audio_paired": torch.tensor(1.0, dtype=torch.float32),
            "onset": onset,
        }

        if self.return_traj:
            # pose is already normalized; root X/Z are dimensions 4 and 6.
            # This keeps trajectory condition in the same normalized space as motion.
            cond["trajectory"] = pose[:, [4, 6]].float()

        return pose, cond, filename_, self.data["wavs"][idx]

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
            
            if pos.shape[0] < required_len:
                continue

            all_pos.append(pos[:required_len])
            all_q.append(q[:required_len])
            all_names.append(feature)
            all_wavs.append(wav)

        all_pos = np.array(all_pos) 
        all_q = np.array(all_q)      
        
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
        train: Optional[bool] = None,
        seq_len=150,
        audio_dim=803,
        overlap=0.5,
        normalizer=None,
        return_traj=False,
        weak_pairs_path="data/proxy_weak_pairs/weak_pairs.csv",
        use_weak_pairs=True,
        split_ratio=0.9,
        split_seed=42,
        audio_sample_mode="random",
        audio_pairing_mode="proxy",
        paired_audio_missing_policy="error",
        traj_aug_prob=0.0,
        traj_aug_scale_range=(1.0, 1.0),
        traj_aug_rot_deg=0.0,
    ):
        self.data_path = data_path
        self.train = train
        self.seq_len = seq_len
        self.audio_dim = audio_dim
        self.return_traj = return_traj
        self.audio_pairing_mode = str(audio_pairing_mode).lower()
        self.paired_audio_missing_policy = paired_audio_missing_policy

        valid_pairing_modes = {"none", "proxy", "paired"}
        if self.audio_pairing_mode not in valid_pairing_modes:
            raise ValueError(
                f"audio_pairing_mode must be one of {sorted(valid_pairing_modes)}, "
                f"got {audio_pairing_mode}"
            )

        if self.audio_pairing_mode == "none":
            use_weak_pairs = False
            audio_sample_mode = "zero"

        if self.audio_pairing_mode == "paired":
            # paired 模式下默认用 best，避免训练时随机抽错候选。
            audio_sample_mode = "best"

        self.audio_sample_mode = audio_sample_mode
        # Trajectory augmentation is enabled only for training.
        # Validation/test data should remain deterministic.
        self.traj_aug_prob = float(traj_aug_prob) if train else 0.0
        self.traj_aug_scale_range = traj_aug_scale_range
        self.traj_aug_rot_deg = float(traj_aug_rot_deg)

        self.proxy_audios = []
        self.motion_window_ids = []
        self.weak_pair_map = {}
        self.weak_pair_audio_cache = {}
        if use_weak_pairs:
            self.weak_pair_map = self._load_weak_pairs(weak_pairs_path)

        rag_db_path = "data/dunhuang_rag_db"
        if os.path.isdir(rag_db_path):
            for rag_file in sorted(glob.glob(os.path.join(rag_db_path, "*.npy"))):
                try:
                    record = np.load(rag_file, allow_pickle=True).item()
                    audio_feat = record.get("audio_feat", None)
                    audio_feat = self._validate_audio_feature(
                        audio_feat,
                        source=rag_file,
                    )
                    if audio_feat is not None:
                        self.proxy_audios.append(audio_feat)
                except Exception as e:
                    print(f"⚠️ 跳过损坏的 RAG 文件 {rag_file}: {e}")

        # 增加路径容错
        candidate_dirs = [data_path, os.path.join(data_path, "processed")]
        self.pkl_files = []
        for candidate in candidate_dirs:
            if os.path.isdir(candidate):
                files = sorted(glob.glob(os.path.join(candidate, "*.pkl")))
                if files:
                    self.pkl_files = files
                    self.data_path = candidate
                    break

        if not self.pkl_files: # 只有当容错路径没找到时，才使用原路径搜索
            search_path = os.path.join(self.data_path, "*.pkl")
            self.pkl_files = sorted(glob.glob(search_path))

        self.all_pkl_files = list(self.pkl_files)
        self.split_source_files(train=train, split_ratio=split_ratio, split_seed=split_seed)

        motions_list = []
        trajs_list = []

        if not self.pkl_files:
            print("Warning: No PKL files found.")
            self.motions = np.zeros((0, seq_len, 151), dtype=np.float32)
            self.trajs = np.zeros((0, seq_len, 2), dtype=np.float32)
            self.normalizer = normalizer if (normalizer is not None and hasattr(normalizer, 'mean')) else None
            return

        smpl = SMPLSkeleton()

        for f in self.pkl_files:
            data = pickle.load(open(f, "rb"))
            pos = data["pos"]
            q = data["q"]

            pos_t = torch.Tensor(pos).unsqueeze(0)
            q_t = torch.Tensor(q).unsqueeze(0)

            bs, sq, c = q_t.shape
            q_t_reshaped = q_t.reshape((bs, sq, -1, 3))

            positions = smpl.forward(q_t_reshaped, pos_t)
            feet = positions[:, :, (7, 8, 10, 11)]
            feetv = torch.zeros(feet.shape[:3])
            feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
            contacts = (feetv < 0.01).to(q_t_reshaped)

            q_6v = ax_to_6v(q_t_reshaped)
            l = [contacts, pos_t, q_6v]
            motion_t = vectorize_many(l)
            motion = motion_t.squeeze(0).float().detach().numpy()

            traj_xz = pos[:, [0, 2]]
            step = max(1, int(seq_len * (1 - overlap)))
            num_frames = motion.shape[0]
            if num_frames < seq_len:
                continue

            for start in range(0, num_frames - seq_len + 1, step):
                slice_motion = motion[start : start + seq_len].copy()
                slice_traj = traj_xz[start : start + seq_len].copy()

                local_start_x = slice_motion[0, 4]
                local_start_z = slice_motion[0, 6]

                slice_motion[:, 4] -= local_start_x
                slice_motion[:, 6] -= local_start_z
                slice_traj[:, 0] -= local_start_x
                slice_traj[:, 1] -= local_start_z

                # Apply geometric trajectory augmentation in physical space.
                # Important: motion root X/Z and trajectory condition must be transformed together.
                slice_motion, slice_traj = self._augment_motion_traj_physical(
                    slice_motion,
                    slice_traj,
                )

                motions_list.append(slice_motion)
                trajs_list.append(slice_traj)
                self.motion_window_ids.append(f"{Path(f).stem}_{start:06d}_{start + seq_len:06d}")

        if len(motions_list) > 0:
            self.motions = np.array(motions_list, dtype=np.float32)
            self.trajs = np.array(trajs_list, dtype=np.float32)

            if normalizer is None or not hasattr(normalizer, 'mean'):
                print("⚠️ 未提供有效 Normalizer，将基于当前敦煌数据集重新计算统计量。")
                self.normalizer = Normalizer(torch.from_numpy(self.motions))
            else:
                self.normalizer = normalizer

            self.motions = self.normalizer.normalize(torch.from_numpy(self.motions)).numpy()
            if self.normalizer is not None and hasattr(self.normalizer, 'std'):
                # ✨ 修复 2：获取根节点 X(4) 和 Z(6) 维度的完整统计量
                mean_xz = np.array([self.normalizer.mean[4], self.normalizer.mean[6]], dtype=np.float32)
                std_xz = np.array([self.normalizer.std[4], self.normalizer.std[6]], dtype=np.float32)
                
                # 源头预处理：严格执行 Z-Score 规范化，完全减去均值并除以方差
                # 确保目标轨迹流形与扩散模型的预测流形实现 100% 重合
                self.trajs = (self.trajs - mean_xz) / std_xz
        else:
            self.motions = np.zeros((0, seq_len, 151), dtype=np.float32)
            self.trajs = np.zeros((0, seq_len, 2), dtype=np.float32)
            self.normalizer = normalizer if (normalizer is not None and hasattr(normalizer, 'mean')) else None
    def _augment_motion_traj_physical(self, motion, traj):
        """
        Apply the same geometric transform to motion root X/Z and trajectory X/Z.

        motion: [T, 151], physical-space motion before normalization
        traj:   [T, 2],   physical-space trajectory before normalization

        Important:
        We must transform motion root X/Z and trajectory condition together.
        Otherwise, the model would receive one trajectory but be trained to
        reconstruct a different root path, which is wrong supervision.
        """
        if self.traj_aug_prob <= 0:
            return motion, traj

        if random.random() > self.traj_aug_prob:
            return motion, traj

        motion = motion.copy()
        traj = traj.copy()

        scale_min, scale_max = self.traj_aug_scale_range
        scale = random.uniform(float(scale_min), float(scale_max))

        angle = random.uniform(
            -self.traj_aug_rot_deg,
            self.traj_aug_rot_deg,
        ) * np.pi / 180.0

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot = np.array(
            [
                [cos_a, -sin_a],
                [sin_a, cos_a],
            ],
            dtype=np.float32,
        )

        # Root X/Z in the 151-D representation are dim 4 and dim 6.
        root_xz = motion[:, [4, 6]].astype(np.float32)
        root_xz = root_xz @ rot.T
        root_xz = root_xz * scale

        traj_xz = traj.astype(np.float32)
        traj_xz = traj_xz @ rot.T
        traj_xz = traj_xz * scale

        motion[:, 4] = root_xz[:, 0]
        motion[:, 6] = root_xz[:, 1]

        traj[:, 0] = traj_xz[:, 0]
        traj[:, 1] = traj_xz[:, 1]

        return motion, traj
    def split_source_files(self, train, split_ratio=0.9, split_seed=42):
        if train is None or len(self.pkl_files) <= 1:
            self.split_name = "all" if train is None else ("train" if train else "val")
            if train is not None and len(self.pkl_files) <= 1:
                print("⚠️ 敦煌源文件少于 2 个，训练/验证将复用同一源文件。")
            return

        split_ratio = float(np.clip(split_ratio, 0.0, 1.0))
        rng = random.Random(split_seed)
        files = list(self.pkl_files)
        rng.shuffle(files)

        split_idx = int(round(len(files) * split_ratio))
        split_idx = max(1, min(len(files) - 1, split_idx))
        train_files = sorted(files[:split_idx])
        val_files = sorted(files[split_idx:])

        self.pkl_files = train_files if train else val_files
        self.split_name = "train" if train else "val"
        print(
            f"📦 DunhuangDataset file-level split [{self.split_name}]: "
            f"{len(self.pkl_files)}/{len(files)} source files, "
            f"split_ratio={split_ratio:.2f}, seed={split_seed}"
        )

    def __len__(self):
        return len(self.motions)

    def _resolve_audio_feature_path(self, audio_path):
        if not audio_path:
            return None

        candidates = []
        path = Path(audio_path)
        if path.suffix == ".npy":
            candidates.append(path)
        else:
            candidates.append(path.with_suffix(".npy"))
        candidates.append(Path("proxy_music") / f"{path.stem}.npy")

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return None

    def _validate_audio_feature(self, audio_feat, source=""):
        """
        Validate one audio feature array before using it as a model condition.

        Why:
        - Dunhuang has no guaranteed paired music-motion dataset.
        - Proxy/RAG/weak-pair feature files can accidentally mix 768-D wav2vec,
          803-D hybrid, empty arrays, or corrupted NaN/Inf values.
        - Failing early in the dataset layer is safer than silently feeding bad
          conditions into cond_projection.
        """
        if audio_feat is None:
            return None

        try:
            audio_feat = np.asarray(audio_feat, dtype=np.float32)
        except Exception as exc:
            print(f"⚠️ 跳过音频特征 {source}: 无法转成 float32 ({exc})")
            return None

        if audio_feat.ndim != 2:
            print(f"⚠️ 跳过音频特征 {source}: 期望 [T, C]，实际 {audio_feat.shape}")
            return None

        if audio_feat.shape[0] <= 0:
            print(f"⚠️ 跳过音频特征 {source}: 时间长度为空 {audio_feat.shape}")
            return None

        if audio_feat.shape[1] != self.audio_dim:
            print(
                f"⚠️ 跳过音频特征 {source}: "
                f"期望 audio_dim={self.audio_dim}，实际 {audio_feat.shape[1]}"
            )
            return None

        if not np.isfinite(audio_feat).all():
            print(f"⚠️ 跳过音频特征 {source}: 包含 NaN/Inf")
            return None

        return audio_feat

    def _get_weak_pair_audio(self, audio_path):
        feature_path = self._resolve_audio_feature_path(audio_path)
        if feature_path is None:
            return None
        if feature_path not in self.weak_pair_audio_cache:
            try:
                loaded = np.load(feature_path)
                self.weak_pair_audio_cache[feature_path] = self._validate_audio_feature(
                    loaded,
                    source=feature_path,
                )
            except Exception as exc:
                print(f"⚠️ 跳过 weak pair 音频特征 {feature_path}: {exc}")
                self.weak_pair_audio_cache[feature_path] = None
        return self.weak_pair_audio_cache[feature_path]

    def _load_weak_pairs(self, weak_pairs_path):
        if not weak_pairs_path or not os.path.isfile(weak_pairs_path):
            return {}

        pair_map = {}
        with open(weak_pairs_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                window_id = row.get("window_id", "")
                audio_path = row.get("audio_path", "")
                audio_feat = self._get_weak_pair_audio(audio_path)
                if not window_id or audio_feat is None:
                    continue
                try:
                    score = max(float(row.get("score", 1.0)), 1e-6)
                except ValueError:
                    score = 1.0
                pair_map.setdefault(window_id, []).append(
                    {"audio_feat": audio_feat, "score": score, "audio_path": audio_path}
                )

        if pair_map:
            print(f"🎵 已加载 weak proxy music 候选: {len(pair_map)} 个动作窗口。")
        return pair_map

    def _choose_weak_candidate(self, candidates):
        if not candidates:
            return None

        if self.audio_sample_mode == "best":
            return max(candidates, key=lambda candidate: candidate["score"])

        weights = [max(float(candidate.get("score", 1.0)), 1e-6) for candidate in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]


    def _sample_audio_feature(self, idx):
        """
        Return:
            audio_feat or None

        audio_pairing_mode:
        - none:
            Always returns None, and __getitem__ will convert it to zero audio.
        - proxy:
            Uses weak-pair candidates if available; otherwise falls back to proxy/RAG music.
            This is weak rhythm guidance only, not paired supervision.
        - paired:
            Requires a weak_pair_map entry for the exact motion window id.
            If missing, either raises error or returns None according to paired_audio_missing_policy.
        """
        if self.audio_pairing_mode == "none" or self.audio_sample_mode == "zero":
            return None

        window_id = self.motion_window_ids[idx] if idx < len(self.motion_window_ids) else ""
        weak_candidates = self.weak_pair_map.get(window_id, [])

        if self.audio_pairing_mode == "paired":
            candidate = self._choose_weak_candidate(weak_candidates)
            if candidate is not None:
                return candidate["audio_feat"]

            message = (
                f"audio_pairing_mode='paired' requires a paired audio candidate "
                f"for motion window '{window_id}', but none was found. "
                f"Check --weak_pairs_path or use --audio_pairing_mode proxy/none."
            )

            if self.paired_audio_missing_policy == "zero":
                print(f"⚠️ {message} Falling back to zero audio.")
                return None

            raise RuntimeError(message)

        # proxy mode: weak candidates are preferred, but fallback is allowed.
        candidate = self._choose_weak_candidate(weak_candidates)
        if candidate is not None:
            return candidate["audio_feat"]

        if hasattr(self, "proxy_audios") and len(self.proxy_audios) > 0:
            if self.audio_sample_mode == "best":
                return self.proxy_audios[0]
            return random.choice(self.proxy_audios)

        return None

    def __getitem__(self, idx):
        motion = torch.from_numpy(self.motions[idx])

        audio_feat = self._sample_audio_feature(idx)
        audio_feat = self._validate_audio_feature(audio_feat, source=f"idx={idx}")

        if audio_feat is not None:
            if audio_feat.shape[0] > self.seq_len:
                audio_feat = audio_feat[:self.seq_len]
            elif audio_feat.shape[0] < self.seq_len:
                pad_len = self.seq_len - audio_feat.shape[0]
                # Use safe reflect padding. Very short sequences fall back to
                # repeat/edge behavior to avoid artificial high-frequency bounces.
                if audio_feat.shape[0] == 1:
                    audio_feat = np.repeat(audio_feat, self.seq_len, axis=0)
                elif audio_feat.shape[0] < 15:
                    audio_feat = np.pad(audio_feat, ((0, pad_len), (0, 0)), mode="edge")
                else:
                    curr_pad = pad_len
                    while curr_pad > 0:
                        step_pad = min(curr_pad, audio_feat.shape[0] - 1)
                        audio_feat = np.pad(
                            audio_feat,
                            ((0, step_pad), (0, 0)),
                            mode="reflect",
                        )
                        curr_pad -= step_pad

            audio_tensor = torch.from_numpy(audio_feat).float()
        else:
            audio_tensor = torch.zeros((self.seq_len, self.audio_dim), dtype=torch.float32)

        audio_is_paired = (
            self.audio_pairing_mode == "paired"
            and audio_feat is not None
        )

        # onset 单独暴露出来，方便后面做节拍弱引导；不是强监督标签。
        if audio_tensor.shape[-1] > 768:
            onset = audio_tensor[:, 768:769].clamp_min(0.0)
            onset = onset / onset.amax().clamp_min(1e-6)
        else:
            onset = torch.zeros((self.seq_len, 1), dtype=torch.float32)

        cond = {
            "audio": audio_tensor,
            "audio_paired": torch.tensor(float(audio_is_paired), dtype=torch.float32),
            "onset": onset,
        }

        if self.return_traj:
            traj = torch.from_numpy(self.trajs[idx])
            cond["trajectory"] = traj

        return motion, cond, f"dunhuang_motion_{idx}", "paired_audio" if audio_is_paired else "unpaired_or_proxy_audio"
