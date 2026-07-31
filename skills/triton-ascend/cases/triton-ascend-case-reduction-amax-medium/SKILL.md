---
name: triton-ascend-case-reduction-amax-medium
description: "Medium-sized contract (max) optimization: calculates a reduction in the number of returns (circumulation, out-of-circumulation), grid = 40 equals the highest performance of the core number of times (25.73us), and applies to the 2D return scenario for the nonreduce axis medium, larger reduce axis (tens of millions of elements)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Medium Size Amax Recession Optimization

## Task characteristics
- **Data size**: (2048,8192), nonreduce axis medium, reduce axis larger

## Optimization 1: Calculate reorganization

```python
# Error: Simple way: multiple returns within the cycle
row_max = -float('inf')
for n_offset in range(0, N, BLOCK_SIZE):
    curr_max = tl.max(data_block, 1)
    row_max = tl.maximum(curr_max, row_max)

# Correct: Optimized approach: maintenance of matrix structure, circular out-of-contract
curr_max = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), -float('inf'), dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    curr_max = tl.maximum(data_block, curr_max)
row_max = tl.max(curr_max, 1)
```

## Optimizing 2: Grid Configuration

```python
# (AI core=40)
# Grid =32 < 40, UB full - > 29.05 us
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256})

# 2 grid>40, UB full - > 29.09 us
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512})

# Grid = 40, UB within - > 25.73 us best
triton.Config({'BLOCK_SIZE_M': 52, 'BLOCK_SIZE_N': 256})
```

### Summary
1. Computation of reorganization: consolidation of multiple returns into one and reduction of the number of returns
2. Grid Configuration: Grid equals the best performance for a core number of hours, and needs to ensure that UB is within range
