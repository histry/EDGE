# V23-v2.5.1 num_bins hotfix

Fixes:

```python
AttributeError: 'V23MonotonicDurationNet' object has no attribute 'num_bins'
```

The model stores the number of duration bins in `self.num_duration_bins`, but
`predict_duration()` incorrectly referenced `self.num_bins`. The hotfix changes
that clamp to use `self.num_duration_bins`.

Install:

```bash
bash install_hotfix.sh /home/disk/lsm/storage/EDGE
```

Then rerun:

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python tools/smoke_v23_monotonic_duration.py \
  --window_len 120 \
  --duration_edges 12,24,37,50,63,76,89
```
