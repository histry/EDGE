# EDGE Environment Setup

This repo has been verified with:

- Python 3.9
- PyTorch 2.8.0 + CUDA 12.8 wheels
- PyTorch3D 0.7.9 from the upstream commit used by the current working env
- RTX 4090 class GPU for practical training and inference

## One-command setup

From the repo root:

```bash
bash setup_edge_env.sh --env-name edge
```

Then activate it:

```bash
conda activate edge
```

The setup script installs dependencies in this order:

1. Conda base packages: Python, pip, ffmpeg, libsndfile.
2. PyTorch CUDA 12.8 wheels: `torch==2.8.0`, `torchvision==0.23.0`, `torchaudio==2.8.0`.
3. Core Python packages from `requirements-edge.txt`.
4. PyTorch3D from the commit matching the verified environment.
5. An import health check for training, inference, evaluation, rendering, and Gradio demo modules.

## Dependency groups

- Training: `torch`, `accelerate`, `wandb`, `einops`, `p-tqdm`, `tqdm`.
- Motion geometry: `pytorch3d`, `numpy`, `scipy`.
- Audio features and metrics: `librosa`, `soundfile`, `scikit-learn`, `transformers`.
- Visualization and demo: `matplotlib`, `ffmpeg`, `opencv-python`, `gradio`.
- Evaluation utilities: `fastdtw`, `tensorboard`.

## Optional legacy Jukebox path

The current Dunhuang pipeline uses Wav2Vec2 + Librosa features. The old Jukebox feature extractor is optional and heavier. Install it only if you need `data/audio_extraction/jukebox_features.py`:

```bash
bash setup_edge_env.sh --env-name edge --with-jukebox
```

## Notes

- The first Wav2Vec2 feature extraction may download `facebook/wav2vec2-base` from Hugging Face.
- `gradio` is included because `gradio_app.py` imports it directly.
- `fastdtw` is included because `eval_comprehensive.py` imports it directly.
- If PyTorch3D source build is already available in the environment, the script skips rebuilding it.
