---
name: triton-ascend-case-reduction-mean-medium
description: "Medium scale reduce first axis (mean) optimization: calculates the number of times the reorganization reduces the number of returns, the grid is slightly smaller than the number of AI Cores and the best performance when avoiding tailings (grid = 32 best 9.98us), and applies to the 2D return scene of reduce first axis, with both axes medium (millions of elements)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Medium Size Mean Reduce First Axis

## Task characteristics
- **Data size**: (1024,4096), first axis of reduce, medium of nonreduce axis

## Optimization: calculation of reorganization

```python
# Simple.
total_sum = 0.0
for n_offset in range(0, N, BLOCK_SIZE):
  Error:row_sum += tl.sum(block_vals)

# Correct: Optimization
col_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for m_start in range(0, M, BLOCK_SIZE_M):
    col_sum += block_vals
col_sum = tl.sum(col_sum, axis=0)
```

## Autotune Configuration

```python
# (AI core=40)
# 1. Grid = 16 < 40, UB full - > 13.32 us
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256})

# Grid = 40 with tails - > 35.12 us
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 103})

# Grid = 32 < 40, UB full - > 9.98 us best
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128})

# 4. Grid = 64 > 40, UB full - > 13.33 us
triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64})

# 5. Grid = 128> 40, UB full - > 22.22 us
triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 32})
```

### Summary
Grid size is slightly smaller than the number of AI Cores and works best when avoiding tail blocks. The tail block results in a significant decrease in performance.
