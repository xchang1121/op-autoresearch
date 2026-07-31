---
name: triton-ascend-case-reduction-sum-large
description: "Large-scale return (sum) non-reduce axis is highly optimized: calculate the number of reorganization reductions and allocate large fraction dimensions to reduce axis with priority for full UB (BLONK_SIZE_N=1024 best 685.65us), for a very large nonduce axis (60,000+) and medium (thousands) reduce axis"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Large Sum Convention Optimization

## Task characteristics
- **Data size**(65536, 2048), nonreduce axis very large, reduce axis medium

## Optimization: reduce axial mass + calculate reorganization

```python
# Simple.
total_sum = 0.0
for n_offset in range(0, N, BLOCK_SIZE):
    row_sum += tl.sum(block_vals)

# Correct: Optimization
acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    acc += block_vals
row_sum = tl.sum(acc, axis=1)
```

## Autotune Configuration

```python
# 1. Reduce axes are smaller, UB is full - > 700.42 us
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256})

# 2. Reduce axles increased to 512 - > 695.08 us
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512})

# - > 685.65 us best.
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1024})

# 4. Reduce axle split to 2048 - > 686.89 us
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048})

# 5. Reduce axes are larger and UBs are not full - >743.83 us
triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 2048})
```

### Summary
In order to give priority to full UB, the reduce axis is allocated a larger dimension to reduce the number of cycles.
