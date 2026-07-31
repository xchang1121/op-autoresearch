---
name: triton-ascend-case-reduction-amin-medium
description: "Large-scale 2D return (amin) reduce axis is highly optimized: distribution of large fraction sizes of reduce axis (BLONK_SIZE_N=16384 best) with priority for full UB, reduction of the number of cycles, with a trade-off of single-temperature loads, applicable to the scene of nonreduce axis medium, reduce axis large (500,000 class elements)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Large-scale 2D Amin return optimized

## Task characteristics
- **Data size**(2048, 262144), nonreduce axis medium, reduce axis large

## Optimization: reduce axis large splits

```python
# Error: simple: multiple returns within the cycle
row_min = float('inf')
for n_start in range(0, N, BLOCK_SIZE_N):
    curr_min = tl.min(data_block, 1)
    row_min = tl.minimum(curr_min, row_min)

# Correct: Optimization: Maintenance of matrix structure
curr_min = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    curr_min = tl.minimum(data_block, curr_min)
row_min = tl.min(curr_min, 1)
```

## Autotune Configuration

```python
# 1. Reduce axles are larger, full of UB - >2864.90 us
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048})

# 2-4. Gradual increase of reduce axle and corresponding reduction of M-string - > Gradual increase of performance
triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096})   # 2840.48 us
triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 8192})   # 2801.20 us
triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16384})  # 2779.78 us Best
```

### Summary
When priority is given to full UB, the reduce axis is allocated a larger cut-off dimension and the number of cycles is reduced, subject to a trade-off with a single inverted load.
