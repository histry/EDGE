# Turn-aware Event Refiner v2 hotfix

Fixes `RuntimeError: The size of tensor a (151) must match the size of tensor b (150)` in `refiner_loss`.

Cause: `model.forward()` returns `pred` as `[1,T,151]`, while `target/base/anchor/event` from the training script can remain `[T,D]`. The previous loss only unsqueezed all tensors when `pred.ndim == 2`, so `target` stayed 2-D and frame-energy tensors became shape-incompatible.

Fix: normalize each tensor independently to `[B,T,D]`, match batch size, and crop defensively to the shortest sequence length.
