import numpy as np
from pathlib import Path
import json

DB = "data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz"
WL = "data/dunhuang_choreo_unit_rag/stationary_expr_whitelist.txt"
OUT = Path("output/stationary_expr_whitelist")
OUT.mkdir(parents=True, exist_ok=True)

z = np.load(DB, allow_pickle=True)

if "unit_motions_physical" not in z:
    raise RuntimeError("Missing unit_motions_physical")
if "unit_motions" not in z:
    raise RuntimeError("Missing unit_motions")

indices = []
for line in Path(WL).read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        indices.append(int(line))

report = []

for rank, idx in enumerate(indices, 1):
    raw = z["unit_motions"][idx].astype(np.float32)
    phys = z["unit_motions_physical"][idx].astype(np.float32)
    center = raw.shape[0] // 2

    prefix = OUT / f"good{rank:03d}_idx{idx}"

    raw_path = str(prefix) + "_raw_unit.npy"
    phys_path = str(prefix) + "_physical_unit.npy"
    cond_mid_path = str(prefix) + "_cond_mid.npy"
    phys_mid_path = str(prefix) + "_physical_mid_static30.npy"

    np.save(raw_path, raw)
    np.save(phys_path, phys)
    np.save(cond_mid_path, raw[center])
    np.save(phys_mid_path, np.repeat(phys[center:center+1], 30, axis=0))

    item = {
        "rank": rank,
        "idx": idx,
        "raw_unit_path": raw_path,
        "physical_unit_path": phys_path,
        "cond_mid_path": cond_mid_path,
        "physical_mid_static30_path": phys_mid_path,
    }

    for k in [
        "source",
        "source_frame",
        "motion_text",
        "motion_energy",
        "root_speed",
        "upper_activity",
        "lower_activity",
        "spatial_range",
        "turning",
        "mobility_label",
        "mobility_metric_root_path",
        "mobility_metric_upper_activity",
        "mobility_metric_torso_activity",
        "mobility_metric_lower_activity",
        "mobility_metric_jerk",
    ]:
        if k in z:
            v = z[k][idx]
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            item[k] = str(v) if isinstance(v, np.str_) else v

    report.append(item)

Path(OUT / "whitelist_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
