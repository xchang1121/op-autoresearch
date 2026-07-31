---
name: pypto-case-norm-batchnorm
description: "Example C: 3D Norm — BatchNorm, displaying 3D down, single-axis sum multi-dimensional returns, expand_crone broadcast"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "norm,reduction,loop,expand_clone"
---

# Mode C-2: 3D Norm — BatchNom

Forward `reshape(B, C, -1)` down to 3D, kernel along Channel V loop.

```python
BASIC_CHANNEL = 8
MAIN_CHANNEL_LOOP = 8   # channels / BASIC_CHANNEL

def create_batchnorm_kernel(batch, channels, spatial, eps):
    assert channels == MAIN_CHANNEL_LOOP * BASIC_CHANNEL
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        x: pypto.Tensor((batch, channels, spatial), pypto.DT_FP32),
    ) -> pypto.Tensor((batch, channels, spatial), pypto.DT_FP32):
        output = pypto.tensor([batch, channels, spatial], pypto.DT_FP32)
        inv_total = 1.0 / (batch * spatial)
        pypto.set_vec_tile_shapes(1, 1, 16384)
        for ci in pypto.loop(0, MAIN_CHANNEL_LOOP, 1, name="LOOP_CH", idx_name="ci"):
            ch_off = ci * BASIC_CHANNEL
            x_chunk = pypto.view(x, [batch, BASIC_CHANNEL, spatial], [0, ch_off, 0])
            # Multi-axis contract: two consecutive single-axis sum
            s = pypto.sum(x_chunk, dim=2, keepdim=True)
            s = pypto.sum(s, dim=0, keepdim=True)      # (1, C, 1)
            sq = pypto.sum(x_chunk * x_chunk, dim=2, keepdim=True)
            sq = pypto.sum(sq, dim=0, keepdim=True)
            mean = s * inv_total
            var = sq * inv_total - mean * mean
            denom = pypto.sqrt(var + eps)
            # expand_cline broadcastback batt-dimensional
            mean_b = pypto.expand_clone(mean, [batch, BASIC_CHANNEL, 1])
            denom_b = pypto.expand_clone(denom, [batch, BASIC_CHANNEL, 1])
            normed = (x_chunk - mean_b) / denom_b
            pypto.assemble(normed, [0, ch_off, 0], output)
        return output
    return kernel
```

forward:`reshape(B, C, -1)` → kernel → `reshape(x.shape)`
RMSNornm Same Mode: 3D `(B, features, spatial)` for `sqrt(mean(x²) + eps)` only.

## Elements of a model
- `pypto.sum(dim=2)`, and `pypto.sum(dim=0)` -- multiaxis will have to be phased.
- `pypto.expand_clone(mean, [B, C, 1])` — Single-axis broadcast, restore dimensions after contract for operation
- `set_vec_tile_shapes(1, 1, 16384)` — 3D, first two dimensions small, last dimension file
