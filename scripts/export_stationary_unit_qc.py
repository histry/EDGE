import numpy as np
from pathlib import Path
import json

DB = "data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz"
OUT = Path("output/stationary_unit_qc")
OUT.mkdir(parents=True, exist_ok=True)

BLACKLIST = set()
black_path = Path("data/dunhuang_choreo_unit_rag/mobility_blacklist.txt")
if black_path.exists():
    for line in black_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            BLACKLIST.add(int(line))

z = np.load(DB, allow_pickle=True)

if "unit_motions_physical" not in z:
    raise RuntimeError("DB has no unit_motions_physical. Cannot do visual QC safely.")

if "unit_motions" not in z:
    raise RuntimeError("DB has no unit_motions. Cannot export condition-space units.")

motions_raw = z["unit_motions"]
motions_phys = z["unit_motions_physical"]
N = motions_raw.shape[0]

labels = z["mobility_label"] if "mobility_label" in z else np.array([""] * N)

def get_arr(name, default=0.0):
    if name in z and z[name].shape[0] == N:
        return z[name].astype(np.float32)
    return np.full((N,), default, dtype=np.float32)

root_path = get_arr("mobility_metric_root_path")
upper = get_arr("mobility_metric_upper_activity")
torso = get_arr("mobility_metric_torso_activity")
lower = get_arr("mobility_metric_lower_activity")
jerk = get_arr("mobility_metric_jerk")
turn = get_arr("mobility_metric_turn")
expr = get_arr("mobility_score_expressive")
stationary = get_arr("mobility_score_stationary")

# 更保守：原地敦煌舞优先要 upper 有动作，而不是只靠 torso/turn 分数。
score = (
    3.0 * upper
    + 1.2 * torso
    + 0.8 * expr
    + 0.5 * stationary
    - 3.0 * root_path
    - 2.0 * jerk
    - 0.8 * lower
)

candidates = []

for i in range(N):
    if i in BLACKLIST:
        continue

    lab = str(labels[i])

    # 原地 expressive 第一轮先排除 turn_in_place，避免把奇怪转体片段当舞蹈短句。
    if lab not in ["stationary_expressive", "stationary"]:
        continue

    # root 小、上肢不能太低、jerk 不能太高。
    if root_path[i] > 0.20:
        continue
    if upper[i] < 0.006:
        continue
    if jerk[i] > 0.35:
        continue

    if not np.isfinite(score[i]):
        continue

    candidates.append((float(score[i]), i, lab))

candidates = sorted(candidates, reverse=True)

report = []
TOPK = 80

for rank, (s, idx, lab) in enumerate(candidates[:TOPK], 1):
    unit_raw = motions_raw[idx].astype(np.float32)
    unit_phys = motions_phys[idx].astype(np.float32)

    center = unit_raw.shape[0] // 2

    raw_out = OUT / f"rank{rank:03d}_idx{idx}_{lab}_raw_unit.npy"
    phys_out = OUT / f"rank{rank:03d}_idx{idx}_{lab}_physical.npy"
    cond_mid_out = OUT / f"rank{rank:03d}_idx{idx}_{lab}_cond_mid.npy"
    phys_mid_out = OUT / f"rank{rank:03d}_idx{idx}_{lab}_physical_mid_static30.npy"

    np.save(raw_out, unit_raw)
    np.save(phys_out, unit_phys)

    # 后续如果要给 generate_controlled_v9.py 作为 mid pose，用 condition-space midpoint。
    np.save(cond_mid_out, unit_raw[center])

    # 仅用于视觉检查 mid pose。
    phys_mid = np.repeat(unit_phys[center:center+1], 30, axis=0)
    np.save(phys_mid_out, phys_mid)

    report.append({
        "rank": rank,
        "idx": int(idx),
        "label": lab,
        "score": s,
        "root_path": float(root_path[idx]),
        "upper": float(upper[idx]),
        "torso": float(torso[idx]),
        "lower": float(lower[idx]),
        "jerk": float(jerk[idx]),
        "turn": float(turn[idx]),
        "raw_unit_path": str(raw_out),
        "physical_unit_path": str(phys_out),
        "cond_mid_path": str(cond_mid_out),
        "physical_mid_static30_path": str(phys_mid_out),
    })

(OUT / "qc_candidates.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

print("exported", len(report), "candidates to", OUT)
for r in report[:30]:
    print(r)
