---
name: triton-ascend-grid-config
description: "Grid/Block Configuration Policy, including Numeric Number Selection, Parallel Moderation, Double Slit and Large Shape operator Processing Scheme. This applies to kernel start-up parameters that need to be defined, polynuclear parallel efficiency optimized, or kernel code generation scenarios that process mega-data"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Grid Configuration Policy

## Grid Limit
- Grid must be tuple, up to 3D: `(x,)`, `(x, y)`, `(x, y, z)`
- No more than 65535 by dimensions
- BLONK_SIZE must be less than 65536

## Recommended scenario: stagger cycle (fixed Grid as core)

operator (Element-wise, Reduce, Normalization, etc.) applied independently by line/block.

```python
@triton.jit
def kernel(
    input_ptr, output_ptr, M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    # Staggering: Pid=0 Processing 0, CORE_NUM, 2*CORE_NUM, line...
    for row_idx in range(pid, M, CORE_NUM):
        row_ptr = input_ptr + row_idx * stride_m
        out_ptr = output_ptr + row_idx * stride_m
        for col_start in range(0, N, BLOCK_N):
            offs = col_start + tl.arange(0, BLOCK_N)
            mask = offs < N
            data = tl.load(row_ptr + offs * stride_n, mask=mask)
            result = compute(data)
            tl.store(out_ptr + offs * stride_n, result, mask=mask)
```

## Dynamic access core number

**Must be obtained in `__init__`**, which is prohibited from calling in forward (trigger device sync).

```python
import torch
import triton

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            device = torch.npu.current_device()
            properties = triton.runtime.driver.active.utils.get_device_properties(device)
            self.VEC_CORE_NUM = properties.get("num_vectorcore", 40)
            self.CUBE_CORE_NUM = properties.get("num_aicore", 20)
        except:
            self.VEC_CORE_NUM = 40
            self.CUBE_CORE_NUM = 20

    def forward(self, x):
        M, N = x.shape
        out = torch.empty_like(x)
        grid = (self.VEC_CORE_NUM,)
        kernel[grid](x, out, M, N, x.stride(0), x.stride(1),
                     BLOCK_N=256, CORE_NUM=self.VEC_CORE_NUM)
        return out
```

### Core Number Selection
- **vector operator**(election-wise, softmax, unified): Use `VEC_CORE_NUM`
- **Matrix operator**(matmul, attent): Use `CUBE_CORE_NUM`

## Multiple splitting strategies

If BLONK_SIZE over-limits or single-slice ultra-hardware caches, the loops can be embedded for multi-layered splits:

```python
for m_start in range(pid_m * BLOCK_M, min((pid_m + 1) * BLOCK_M, M), SUB_BLOCK_M):
    for n_start in range(pid_n * BLOCK_N, min((pid_n + 1) * BLOCK_N, N), SUB_BLOCK_N):
        # Block to process SUB_BLONK size
```
