#!/usr/bin/env python3
from pathlib import Path

p = Path("functional_dual_context_selector.py")
txt = p.read_text(encoding="utf-8")

backup = Path("functional_dual_context_selector.py.bak_before_v15_support_quality")
if not backup.exists():
    backup.write_text(txt, encoding="utf-8")

insert = r'''

_SUPPORT_QUALITY_CACHE = {}

def _support_quality_components(db):
    """Load V15 support-prior quality sidecar and return arrays aligned with rag DB."""
    n = len(field(db, "mobile_score", 0.0))

    if not env_bool("EDGE_ENABLE_SUPPORT_QUALITY_RAG", False):
        z = np.zeros((n,), dtype=np.float32)
        return {
            "quality": z,
            "gate": np.ones((n,), dtype=np.float32),
            "hf": z,
            "tail_freeze": z,
            "root_drag": z,
            "large_jump": z,
            "high_jerk": z,
        }

    sidecar = os.environ.get("EDGE_SUPPORT_QUALITY_GATE_NPZ", "").strip()
    if not sidecar:
        sidecar = os.environ.get("EDGE_SUPPORT_QUALITY_SIDECAR", "").strip()

    key = sidecar or "__db_fields__"
    if key in _SUPPORT_QUALITY_CACHE:
        return _SUPPORT_QUALITY_CACHE[key]

    def z():
        return np.zeros((n,), dtype=np.float32)

    def arr_from_sidecar(npz, name, default=0.0):
        if name in npz.files:
            a = np.nan_to_num(np.asarray(npz[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if len(a) == n:
                return a
        return np.full((n,), float(default), dtype=np.float32)

    comp = None
    if sidecar and Path(sidecar).is_file():
        npz = np.load(sidecar, allow_pickle=True)
        comp = {
            "quality": arr_from_sidecar(npz, "support_prior_quality", 0.0),
            "gate": arr_from_sidecar(npz, "good_gate", 1.0),
            "hf": arr_from_sidecar(npz, "hf_event_score_norm", 0.0),
            "tail_freeze": arr_from_sidecar(npz, "tail_freeze", 0.0),
            "root_drag": arr_from_sidecar(npz, "root_drag", 0.0),
            "large_jump": arr_from_sidecar(npz, "large_jump", 0.0),
            "high_jerk": arr_from_sidecar(npz, "high_jerk", 0.0),
        }
    else:
        # Forward-compatible path: allow fields to be embedded directly in future DB.
        comp = {
            "quality": field(db, "support_prior_quality", 0.0),
            "gate": field(db, "good_gate", 1.0),
            "hf": field(db, "hf_event_score_norm", 0.0),
            "tail_freeze": field(db, "tail_freeze", 0.0),
            "root_drag": field(db, "root_drag", 0.0),
            "large_jump": field(db, "large_jump", 0.0),
            "high_jerk": field(db, "high_jerk", 0.0),
        }

    for k, v in list(comp.items()):
        v = np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if len(v) != n:
            print(f"⚠️ V15 support-quality field {k} length mismatch: {len(v)} vs DB {n}; disabling this field.")
            v = z()
            if k == "gate":
                v = np.ones((n,), dtype=np.float32)
        comp[k] = v.astype(np.float32)

    _SUPPORT_QUALITY_CACHE[key] = comp
    return comp


def support_quality_bonus(db, role):
    """V15 Support-Prior Quality Gate bonus/penalty for RAG rerank."""
    if not env_bool("EDGE_ENABLE_SUPPORT_QUALITY_RAG", False):
        return np.zeros_like(field(db, "mobile_score", 0.0), dtype=np.float32)

    c = _support_quality_components(db)

    if role == "support":
        wq = env_float("EDGE_SUPPORT_QUALITY_WEIGHT", 0.35)
        whf = env_float("EDGE_SUPPORT_QUALITY_HF_WEIGHT", 0.10)
    else:
        wq = env_float("EDGE_SUPPORT_QUALITY_EXP_WEIGHT", 0.20)
        whf = env_float("EDGE_SUPPORT_QUALITY_EXP_HF_WEIGHT", 0.06)

    penalty = (
        env_float("EDGE_SUPPORT_QUALITY_TAIL_FREEZE_PENALTY", 0.30) * c["tail_freeze"]
        + env_float("EDGE_SUPPORT_QUALITY_ROOT_DRAG_PENALTY", 0.25) * c["root_drag"]
        + env_float("EDGE_SUPPORT_QUALITY_LARGE_JUMP_PENALTY", 0.20) * c["large_jump"]
        + env_float("EDGE_SUPPORT_QUALITY_HIGH_JERK_PENALTY", 0.15) * c["high_jerk"]
    )

    bonus = wq * c["quality"] + whf * c["hf"] - penalty

    if env_bool("EDGE_SUPPORT_QUALITY_REQUIRE_GATE", True):
        # Hard reject bad prior candidates.
        bonus = np.where(c["gate"] > 0.5, bonus, -1.0e4)

    return np.nan_to_num(bonus.astype(np.float32), nan=0.0, posinf=0.0, neginf=-1.0e4)
'''

marker = "\ndef score_contexts(db, event_strength, role):\n"

if "def support_quality_bonus(db, role):" not in txt:
    if marker not in txt:
        raise SystemExit("ERROR: cannot find score_contexts marker")
    txt = txt.replace(marker, insert + marker)

old_support = '''        score = score + video_bonus(db, "support", event_strength)
        return np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
'''

new_support = '''        score = score + video_bonus(db, "support", event_strength)
        score = score + support_quality_bonus(db, "support")
        return np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
'''

old_expr = '''        score = score + video_bonus(db, "expressive", event_strength)
        return np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
'''

new_expr = '''        score = score + video_bonus(db, "expressive", event_strength)
        score = score + support_quality_bonus(db, "expressive")
        return np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
'''

if 'score = score + support_quality_bonus(db, "support")' not in txt:
    if old_support not in txt:
        raise SystemExit("ERROR: support score block not found")
    txt = txt.replace(old_support, new_support)

if 'score = score + support_quality_bonus(db, "expressive")' not in txt:
    if old_expr not in txt:
        raise SystemExit("ERROR: expressive score block not found")
    txt = txt.replace(old_expr, new_expr)

p.write_text(txt, encoding="utf-8")
print("✅ patched functional_dual_context_selector.py with V15 Support-Prior Quality Gate")
