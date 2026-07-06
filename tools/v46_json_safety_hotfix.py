from pathlib import Path
import shutil
import time
import re

p = Path("tools/v46_motionrag_diff.py")
s = p.read_text(encoding="utf-8")

bak = p.with_suffix(p.suffix + f".json_safe_hotfix_{time.strftime('%Y%m%d_%H%M%S')}.bak")
shutil.copy2(p, bak)
print("[BAK]", bak)

def find_top_level_func_span(text: str, name: str):
    m = re.search(rf"^def\s+{re.escape(name)}\s*\(", text, flags=re.M)
    if not m:
        return None
    start = m.start()
    m2 = re.search(r"^(def\s+|class\s+)", text[m.end():], flags=re.M)
    end = m.end() + m2.start() if m2 else len(text)
    return start, end

# Remove previous helper if it exists, to keep this hotfix idempotent.
span = find_top_level_func_span(s, "_v46_json_safe")
if span is not None:
    s = s[:span[0]] + s[span[1]:]

save_span = find_top_level_func_span(s, "save_json")
if save_span is None:
    # Print nearby top-level functions for debugging.
    funcs = re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", s, flags=re.M)
    raise SystemExit("[ERROR] Cannot find save_json function. Top-level funcs head: " + ", ".join(funcs[:30]))

replacement = r'''def _v46_json_safe(x):
    """Make report/meta objects JSON serializable.

    V46.31 hotfix:
    Chang-E semantic ontology may contain Python set values, e.g. aliases.
    events_meta.json must remain writable, so convert sets/numpy/Path safely.
    """
    import dataclasses as _dataclasses
    import numpy as _np
    from pathlib import Path as _Path

    if _dataclasses.is_dataclass(x):
        return _v46_json_safe(_dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _v46_json_safe(v) for k, v in x.items()}
    if isinstance(x, set):
        return sorted([_v46_json_safe(v) for v in x], key=lambda z: str(z))
    if isinstance(x, (list, tuple)):
        return [_v46_json_safe(v) for v in x]
    if isinstance(x, _Path):
        return str(x)
    if isinstance(x, _np.ndarray):
        return _v46_json_safe(x.tolist())
    if isinstance(x, _np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_v46_json_safe(obj), f, ensure_ascii=False, indent=2)
'''

s = s[:save_span[0]] + replacement + s[save_span[1]:]
p.write_text(s, encoding="utf-8")

print("[OK] Replaced save_json with JSON-safe version in tools/v46_motionrag_diff.py")
