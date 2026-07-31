---
name: triton-cuda-error-fix
description: Frequent triton-cuda errors and fixes to avoid similar problems in code generation
category: fix
version: "1.0.0"
metadata:
  source: error_fix
  case_type: fix
  backend: cuda
  dsl: triton_cuda
---

### Mathematical Functions Call Error

- **Systems**: `AttributeError: module 'triton.language' has no attribute 'tanh'`
- **Rehabilitation**:
  - Triton CUDA backend mathematical function to be called through `tl.extra.cuda.libdevice` module
  - Avoid direct use of `tl.math.xxx` or `tl.xxx`

```python
# Error: Directly using tl.tanh
result = 0.5 * x * (1.0 + tl.tanh(inner))

# Correct: Call through libdevice
result = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
```