---
name: triton-ascend-case-reduction-prod-small
description: "Small-scale reduce 1st axis (prod) optimization: use of custom mul function in conjunction with tl.reduce ' s multiplication (triton has no prod interface), with a significantly smaller number of optimal grids than the number of AI Cores (grid = 16 best 2.15us), and a high degree of parallelity which reduces performance due to movement costs and applies to the first reduce scenario for the smaller size of sape (100,000 element)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Small-scale Prod return optimized

## Task characteristics
- **Data size**(16, 2048), first axis of reduce, medium of non-reduce axis

## Optimization: Custom reduce function

```python
# Simple.
accumulator = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
for m in range(M):
  Error:accumulator = tl.where(mask, accumulator * data, accumulator)

# Correct: Optimization
@triton.jit
def mul(a, b):
    return a * b

col_prod = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 1.0, dtype=tl.float32)
for m_start in range(0, M, BLOCK_SIZE_M):
    col_prod *= block_vals
col_prod = tl.reduce(col_prod, axis=0, combine_fn=mul)  # tritonNothing.prodInterface
```

## Autotune Configuration

```python
# (AI core=40)
# 1. grid=64>40 -> 4.21 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32})

# Grid = 40 with tails - > 3.28 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 52})

# 3. grid=32<40 -> 2.61 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64})

# 4. Grid = 16<40 -> 2.15 us best
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128})

# Grid = 2 < 40, UB full - > 2.63 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1024})

# 6. grid=1 -> 3.25 us
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048})
```

### Summary
operatorshapeSmaller time (10^5~10^6Element) The optimal grid number may need to be significantly less thanAI CoreQuantities, too many parallels, would, in turn, reduce performance as a result of movement costs.
