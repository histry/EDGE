# EDGE RAG Segment Training Patch

This patch adds a training-level RAG-Diffusion stage.

It addresses the failure mode observed in inference-only RAG:

- auto mid keyframes can be correct but motion between them is still a local pulse;
- postprocess or denoising prior improves metrics but does not teach the model continuous transitions;
- therefore the model must learn during training to use retrieved continuous motion segments.

## Files

```text
build_retrieved_segment_pairs.py
train_rag_segment.py
rag_segment_utils.py
rag_segment_losses.py
dataset/retrieved_segment_dataset.py
```

## 1. Build retrieved segment training pairs

```bash
cd /home/disk/lsm/storage/EDGE

python build_retrieved_segment_pairs.py \
  --rag_db data/dunhuang_rag_db/mmr_rag_index.npz \
  --out data/rag_segment_pairs/dunhuang_train_pairs.jsonl \
  --max_pairs 20000 \
  --top_k 64 \
  --source_gap 150 \
  --disallow_same_source \
  --energy_target 0.55 \
  --energy_band 0.25 \
  --contact_weight 0.5
```

## 2. Train retrieved segment imitation stage

```bash
python train_rag_segment.py \
  --checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --pairs_jsonl data/rag_segment_pairs/dunhuang_train_pairs.jsonl \
  --project runs/train \
  --exp_name exp19_rag_segment_from_exp17_train30 \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 150 \
  --batch_size 16 \
  --epochs 20 \
  --save_interval 5 \
  --learning_rate 1e-5 \
  --weight_decay 0.02 \
  --train_stage stage2 \
  --prior_feature_mode upper \
  --segment_min_len 24 \
  --segment_max_len 60 \
  --base_loss_weight 1.0 \
  --segment_imitation_weight 1.0 \
  --segment_velocity_weight 0.5 \
  --transition_smooth_weight 0.2
```

## 3. Use trained checkpoint for generation

Use the best saved checkpoint, for example:

```bash
runs/train/exp19_rag_segment_from_exp17_train30/weights/train-5.pt
```

Then generate with your existing `generate_controlled.py` using the stable inference setting:

```text
--retrieved_clip_prior_denoise
--retrieved_prior_body_part upper
--retrieved_prior_strength 0.16
--retrieved_prior_window 24
```

## Why this patch exists

Inference-only RAG gives the model retrieved clips only at generation time.  The
model was never trained to use those clips, so it tends to respond locally near
keyframes instead of producing continuous transitions.

This patch trains the model with:

1. retrieved segment pairs;
2. random middle segment masks;
3. retrieved prior mask/value as decoder input;
4. segment imitation loss;
5. velocity consistency loss;
6. transition boundary smoothness loss.

The prior feature mode defaults to `upper` because previous ablations showed
that touching pelvis/root orientation or leg chains can conflict with trajectory
anchoring and amplify foot sliding.
