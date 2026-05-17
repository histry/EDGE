#!/usr/bin/env python3
from pathlib import Path

p = Path("train.py")
s = p.read_text(encoding="utf-8")

needle = '("unit_reconstruction_patch", "install_v3_unit_reconstruction_patch"),'
insert = '''("unit_reconstruction_patch", "install_v3_unit_reconstruction_patch"),
            ("v3c_visible_fk_patch", "install_v3c_visible_fk_patch"),'''

if "v3c_visible_fk_patch" not in s:
    if needle not in s:
        raise SystemExit("Could not find unit_reconstruction_patch entry in train.py. Patch manually.")
    s = s.replace(needle, insert, 1)

needle2 = '_call_install("unit_reconstruction_patch", "install_v3_unit_reconstruction_patch", verbose=True)'
insert2 = '''_call_install("unit_reconstruction_patch", "install_v3_unit_reconstruction_patch", verbose=True)
    _call_install("v3c_visible_fk_patch", "install_v3c_visible_fk_patch", verbose=True)'''

if '_call_install("v3c_visible_fk_patch", "install_v3c_visible_fk_patch", verbose=True)' not in s:
    if needle2 in s:
        s = s.replace(needle2, insert2, 1)

p.write_text(s, encoding="utf-8")
print("✅ train.py patched for V3C visible-FK patch")
