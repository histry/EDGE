# EDGE checkpoint compatibility OOM hotfix

Replace:

```bash
cp model/checkpoint_compat.py /home/disk/lsm/storage/EDGE/model/checkpoint_compat.py
```

What it changes:
- The original `adapt_checkpoint_state_dict()` cloned `model.state_dict()` tensors on their current device.
- If the model was already on CUDA, this duplicated initialized reference tensors on GPU during checkpoint adaptation.
- This hotfix clones fallback initialized reference tensors on CPU by default.

Default:
```bash
EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1
```

You can disable it with:
```bash
EDGE_CHECKPOINT_COMPAT_CPU_MERGE=0
```

Important:
This lowers the GPU peak during checkpoint loading, but it cannot run when the GPU is already full. Kill stale GPU processes first.
