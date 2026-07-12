# V46.49.2 Non-root Position Contract

The current Chang-E data exposes XYZ position channels on every articulated
joint, but non-root positions are approximately static calibration values.
The formal path must keep root translation while using hierarchy OFFSET plus
joint rotation for children.

```bash
export V46_49_NONROOT_POSITION_MODE=ignore
python tools/apply_v46_49_2_nonroot_position_patch.py
python -m py_compile tools/chang_e_edge_retarget.py
```
