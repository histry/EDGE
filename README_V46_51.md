# V46.51 Fresh-WAV Schedule Transaction for EDGE

## 1. Target baseline

This package targets the current repository state:

```text
histry/EDGE
commit 2d6e7d9feb62da960be832d34a29b02bee3be079
V46.50 Event-Level Heading State
```

It requires the validated V46.49.4 retargeter and V46.50 event-heading files.

## 2. Scientific correction

V46.50 requires a user-supplied `SLOTS_JSON`. Although a reused MSSD can be
duration-valid, that design does not prove that it was rebuilt from the current
WAV. V46.51 replaces schedule reuse with an immutable generation transaction:

```text
current WAV
    ↓ SHA-256 + decoded duration
unique run-local music feature cache
    ↓
fresh V21 router + V26 planner + V23 duration scheduling
    ↓
fresh raw V26 report and motion
    ↓
strict final MSSD with current-WAV provenance
    ↓
Audio–Schedule Contract
    ↓
V46.50 event-heading / boundary closed-loop generation
```

A schedule is accepted only when:

```text
audio SHA-256 matches
schedule run_id matches the current generation
usage = generate_schedule
is_final_schedule = true
slot_source = v21_router_v26_planner
no slot gap or overlap
slot frame extents equal target_frames
sum(target_frames) matches current audio frames within tolerance
raw V26 report names the same WAV
raw V26 generated motion matches current audio frame count
```

## 3. Training lifecycle

The action database and learned models do not need to be retrained for every
WAV. They are rebuilt/retrained when the `change/` motion corpus or method
changes.

Every generation does rebuild only the music-side scheduling transaction:

```text
one-time / dataset change:
    V46.49.4 retarget
    → source split before event slicing
    → V46.50 heading Event-RAG DB
    → AESD
    → V44/V45/V46 training

every WAV:
    fresh V21/V26/V23 schedule
    → duration/provenance hard audit
    → V46.50 heading-aware retrieval and closed-loop repair
```

## 4. Source-disjoint research protocol

The formal full script performs train/val/test assignment at source sequence
level before event slicing:

```text
retargeted source NPY
    ↓ source_uid split
train / val / test cache roots
    ↓ adaptive event segmentation inside each split
train / val / test Event-RAG databases
```

Only the train database is used for model training and formal generation.
Val/test are evaluation-only. The optional `qualitative_all_change` mode is an
upper-bound demonstration and must not enter the main quantitative table.

## 5. Files

```text
tools/v46_51_audio_schedule_contract.py
tools/v46_51_build_fresh_mssd.py
tools/v46_51_heading_closed_loop.py
tools/v46_51_resolve_scheduler_assets.py
tools/v46_51_split_retarget_cache.py
tools/v46_51_split_event_db.py
configs/v46_51_fresh_wav_schedule.env
configs/v46_50_event_heading.env
scripts/run_v46_50_full_rebuild_retrain.sh
scripts/run_v46_51_generate_fresh_wav.sh
tests/test_v46_51_fresh_wav_schedule.py
install_v46_51.sh
```

`configs/v46_50_event_heading.env` and
`scripts/run_v46_50_full_rebuild_retrain.sh` are direct replacements. The
remaining files are new.

## 6. Installation

```bash
unzip -o v46_51_fresh_wav_schedule_solution.zip -d /tmp

cd /tmp/v46_51_fresh_wav_schedule_solution

bash install_v46_51.sh /home/disk/lsm/storage/EDGE
```

The installer backs up replaced files under:

```text
EDGE/output/v46_51_install_backup_<timestamp>/
```

## 7. Scheduler asset preflight

```bash
cd /home/disk/lsm/storage/EDGE

source configs/v46_51_fresh_wav_schedule.env

"$V46_51_PYTHON" tools/v46_51_resolve_scheduler_assets.py \
  --out_json output/v46_51_scheduler_assets.json \
  --out_env output/v46_51_scheduler_assets.env
```

If an asset cannot be resolved, set its explicit path:

```bash
export V46_51_INDEX_JSON="..."
export V46_51_DURATION_INDEX_NPZ="..."
export V46_51_ROUTER_CKPT="..."
export V46_51_PLANNER_CKPT="..."
export V46_51_V23_CKPT="..."
export V46_51_HIERARCHY_INDEX_NPZ="..."   # optional
export V46_51_START_POSE="..."            # optional
```

The resolver uses fixed validated candidates and never selects a checkpoint
only because it is the newest file.

## 8. Full motion rebuild and retraining

```bash
cd /home/disk/lsm/storage/EDGE

source configs/v46_51_fresh_wav_schedule.env

export AUDIO="test_music_bank/dunhuangwu2.wav"
export V46_51_DB_MODE=paper_train_only

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
export RUN_TAG
export OUT_ROOT="output/v46_51_fresh_wav_${RUN_TAG}"

nohup bash scripts/run_v46_50_full_rebuild_retrain.sh \
  > "output/v46_51_${RUN_TAG}.log" 2>&1 &

echo "PID=$!"
tail -f "output/v46_51_${RUN_TAG}.log"
```

No `SLOTS_JSON` is accepted or needed.

## 9. Generate another WAV without retraining

After a completed full run:

```bash
cd /home/disk/lsm/storage/EDGE

source configs/v46_51_fresh_wav_schedule.env

export AUDIO="test_music_bank/another_song.wav"
export DB_AESD="output/<full-run>/event_db_split/train/events_aesd.npz"
export V44_CKPT="output/<full-run>/v44_train_only_contrastive.pt"
export V45_CKPT="output/<full-run>/v45_train_only_refiner.pt"
export V46_CKPT="output/<full-run>/v46_train_only_diffusion.pt"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/run_v46_51_generate_fresh_wav.sh \
  > "output/v46_51_generate_${RUN_TAG}.log" 2>&1 &

tail -f "output/v46_51_generate_${RUN_TAG}.log"
```

This always creates a new schedule from `another_song.wav`; it cannot reuse the
Dunhuangwu2 MSSD.

## 10. Deep music features

Default:

```bash
export V46_51_DEEP_MUSIC_FEATURES=1
export V46_51_REQUIRE_DEEP_MUSIC=0
```

This uses CLAP/deep features when the existing local setup works and records the
success state. After the CLAP checkpoint and success-rate audit pass:

```bash
export V46_51_REQUIRE_DEEP_MUSIC=1
```

Formal experiments should report whether deep features were required or only
optional.

## 11. Database modes

Formal paper:

```bash
export V46_51_DB_MODE=paper_train_only
```

Qualitative upper bound:

```bash
export V46_51_DB_MODE=qualitative_all_change
```

The latter may improve candidate coverage but is not a source-disjoint
quantitative experiment.

## 12. Acceptance gates

Fresh-WAV schedule:

```text
current audio SHA-256 match
current transaction run_id match
0 timeline gaps
0 timeline overlaps
frame error <= 2
duration error <= 0.10 s
raw V26 motion length consistent
```

Motion database:

```text
split before event slicing
zero source_uid overlap
V46.49.4 cache contract
heading entry p95 <= 5 degrees
no invalid saved heading event
```

Final motion:

```text
final frames = fresh schedule frames
planned root-heading error p95 <= 2 degrees
no non-turn yaw-budget violation
gravity audit passes
boundary closed-loop audit passes
scientific render smoothing window = 1
```

## 13. Validation status of this package

Completed in the package environment:

```text
Python syntax compilation: pass
Bash syntax validation: pass
fresh transaction unit tests: pass
changed-WAV rejection test: pass
wrong-run-id rejection test: pass
timeline-gap rejection test: pass
source split determinism/disjointness test: pass
```

The package has not executed the full private `change/` corpus, GPU training, or
your local trained V21/V23/V26 checkpoints. The first local run remains the
authoritative integration test.
