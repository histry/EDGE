import os
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

if not hasattr(scipy.signal, "hann") and hasattr(scipy.signal, "windows"):
    scipy.signal.hann = scipy.signal.windows.hann

# 动作序列目标帧率
FPS = 30
# Librosa 采样率技巧：保证 1 hop 严格对应 1 frame
HOP_LENGTH = 512
SR_LIBROSA = FPS * HOP_LENGTH 
# Wav2Vec2 标准采样率
SR_WAV2VEC = 16000 
_EXTRACTOR = None

class LightweightAudioExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading Wav2Vec2 on {self.device}...")
        # 使用基础版 Wav2Vec2，参数量小，非常适合单卡 4090 的显存管理
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        self.model.eval()

    def extract_librosa_features(self, fpath):
        """
        提取显式节奏特征（适用于民乐等鼓点清晰的音频）
        """
        data, _ = librosa.load(fpath, sr=SR_LIBROSA)
        
        # 1. 节拍强度与 MFCC
        envelope = librosa.onset.onset_strength(y=data, sr=SR_LIBROSA)
        mfcc = librosa.feature.mfcc(y=data, sr=SR_LIBROSA, n_mfcc=20).T
        chroma = librosa.feature.chroma_cens(y=data, sr=SR_LIBROSA, hop_length=HOP_LENGTH, n_chroma=12).T
        
        # 2. 峰值检测 (Onset)
        peak_idxs = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=SR_LIBROSA, hop_length=HOP_LENGTH)
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0
        
        # 3. 节拍追踪 (Beat)
        start_bpm = librosa.beat.tempo(y=data, sr=SR_LIBROSA)[0]
        _, beat_idxs = librosa.beat.beat_track(onset_envelope=envelope, sr=SR_LIBROSA, hop_length=HOP_LENGTH, start_bpm=start_bpm, tightness=100)
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0

        # 拼接形状: (seq_len, 1 + 20 + 12 + 1 + 1) = (seq_len, 35)
        rhythm_features = np.concatenate([
            envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
        ], axis=-1)
        
        return rhythm_features

    def extract_wav2vec2_features(self, fpath, target_seq_len):
        """
        提取高维语义特征并插值对齐到目标 30 FPS 序列长度
        """
        # 🌟 修复：从 CPU 唤醒模型回 GPU，适配 Gradio 的显存释放机制
        self.model.to(self.device)
        
        data, _ = librosa.load(fpath, sr=SR_WAV2VEC)
        inputs = self.processor(data, sampling_rate=SR_WAV2VEC, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            # 提取隐藏层状态，形状为 (1, w_seq_len, 768)
            outputs = self.model(inputs.input_values.to(self.device))
            hidden_states = outputs.last_hidden_state 
            
        # 转换形状以进行一维插值: (batch, channels, length) -> (1, 768, w_seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        
        # 使用线性插值严格对齐到 Librosa 的序列长度（即动作帧数）
        aligned_states = F.interpolate(hidden_states, size=target_seq_len, mode='linear', align_corners=False)
        
        # 恢复形状: (target_seq_len, 768)
        aligned_states = aligned_states.squeeze(0).transpose(0, 1).cpu().numpy()
        return aligned_states

    def process(self, fpath, dest_dir, skip_completed=True):
        os.makedirs(dest_dir, exist_ok=True)
        audio_name = Path(fpath).stem
        save_path = os.path.join(dest_dir, f"{audio_name}.npy")

        if os.path.exists(save_path) and skip_completed:
            return save_path

        # 1. 提取节奏基线特征
        librosa_feats = self.extract_librosa_features(fpath)
        target_seq_len = librosa_feats.shape[0]

        # 2. 提取 Wav2Vec2 语义特征并对齐
        wav2vec_feats = self.extract_wav2vec2_features(fpath, target_seq_len)

        # 3. 特征融合 (seq_len, 768 + 35 = 803)
        fused_features = np.concatenate([wav2vec_feats, librosa_feats], axis=-1)

        # 裁剪到 5秒 * 30FPS 的确切长度（如果数据集有此强制要求）
        # fused_features = fused_features[:5 * FPS] 

        np.save(save_path, fused_features)
        return save_path


def get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = LightweightAudioExtractor()
    return _EXTRACTOR


def extract(fpath, skip_completed=True, dest_dir=None):
    if dest_dir is None:
        dest_dir = os.path.dirname(fpath)
    extractor = get_extractor()
    save_path = extractor.process(fpath, dest_dir, skip_completed=skip_completed)
    return np.load(save_path), save_path

def extract_folder(src, dest):
    extractor = get_extractor()
    fpaths = sorted(list(Path(src).glob("*.mp3")) + list(Path(src).glob("*.wav")))
    
    for fpath in tqdm(fpaths, desc="Extracting Hybrid Audio Features"):
        extractor.process(fpath, dest)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source path to audio files")
    parser.add_argument("--dest", required=True, help="dest path to audio features")
    args = parser.parse_args()

    extract_folder(args.src, args.dest)
