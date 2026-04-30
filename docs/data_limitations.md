# Data Limitations and Music Supervision Statement

## Background

The current Dunhuang choreography prototype does not have a real paired
Dunhuang music-motion dataset. The available Dunhuang data mainly provides
motion sequences, while the music used in controlled generation is either
external test music, weak proxy music, or rhythm candidates selected by
heuristics.

## What the current system can claim

The system can currently claim:

1. It adapts the EDGE diffusion-based motion generation framework to a
   Dunhuang-style 151-D SMPL motion representation.
2. It supports start/end and optional middle keyframe constraints.
3. It supports 2D root X/Z trajectory conditioning.
4. It uses music features as a weak rhythmic condition.
5. It can apply inference-time onset/beat guidance to encourage motion
   energy changes near musical accents.

## What the current system should not claim

The system should not claim:

1. The model has learned strict Dunhuang music-motion alignment.
2. The model can perfectly synchronize dance motion to arbitrary Dunhuang music.
3. Proxy music is equivalent to true paired supervision.
4. Beat guidance is the same as supervised music-dance learning.

## Why MMR/audio-motion alignment loss is disabled by default

When `audio_pairing_mode` is `none` or `proxy`, the music is not a ground-truth
paired label. Enabling a strong cross-modal alignment loss in this case may
teach the model incorrect music-motion associations.

Therefore, `mmr_loss_weight` should remain 0 unless `audio_pairing_mode=paired`
and each motion window has a verified paired audio feature.

## Current recommended wording

"Because a real paired Dunhuang music-motion dataset is currently unavailable,
the music branch is used as weak rhythmic guidance rather than strong paired
supervision. The current system demonstrates controllable choreography through
keyframes and 2D trajectory constraints, while music synchronization is evaluated
as a weak beat-response tendency rather than a fully supervised alignment result."

## Future work

1. Build or annotate a real Dunhuang music-motion paired dataset.
2. Add verified beat/downbeat labels and phrase-level music structure labels.
3. Train with paired audio-motion supervision.
4. Re-enable MMR or contrastive audio-motion loss only after paired data exists.
5. Report separate metrics for raw model output and postprocessed system output.