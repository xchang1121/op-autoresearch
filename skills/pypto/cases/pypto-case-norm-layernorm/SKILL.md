---
name: pypto-case-norm-layernorm
description: "Example C: 2D Norm + Loop — Layer Norm, display forward down to 2D, kernel internal sum return + unified"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "norm,reduction,loop"
---

# Mode C-1: 2D Norm — Layer Norm

Forward `reshape(batch, -1)` down to 2D, kernel along the bat-dimensional loop.

```python
BASIC_BATCH = 4

def create_layernorm_kernel(batch, hidden, eps):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        x: pypto.Tensor((batch, hidden), pypto.DT_FP32),
    ) -> pypto.Tensor((batch, hidden), pypto.DT_FP32):
        output = pypto.tensor([batch, hidden], pypto.DT_FP32)
        num_iters = ceil_div(batch, BASIC_BATCH)
        pypto.set_vec_tile_shapes(1, 16384)
        inv_h = 1.0 / hidden
        for bi in pypto.loop(0, num_iters, 1, name="LOOP_LN", idx_name="bi"):
            offset = bi * BASIC_BATCH
            x_chunk = pypto.view(x, [BASIC_BATCH, hidden], [offset, 0])
            mean = pypto.sum(x_chunk, dim=1, keepdim=True) * inv_h
            var = pypto.sum(x_chunk * x_chunk, dim=1, keepdim=True) * inv_h - mean * mean
            normed = (x_chunk - mean) / pypto.sqrt(var + eps)
            pypto.assemble(normed, [offset, 0], output)
        return output
    return kernel
```

forward:`reshape(B, -1)` → kernel → `reshape(x.shape)`
GroupNorum with mode: `reshape(B*G, -1)` → 2D Kernel in Forward.

## Elements of a model
- `pypto.sum(dim=int)` — dim can only flyer int
- = `sum * (1/size)` — no meaning API
- Difference = `E[x²] - E[x]²` — double sum achieved
- `set_vec_tile_shapes(1, 16384)` — 2D, 1D small batch, 2D large file
