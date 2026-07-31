---
name: triton-ascend-case-reduction-amin-large
description: "Extreme scale 1D (amin) optimization: double-segregation avoids over-UB+ calculated reduction in the number of returns, grids close to the number of AI Cores (grid = 32), UBs with maximum performance when full and no tail block (9,61us), full-scale attribution scenarios for very large scale 1D data (4 million elements)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Extreme Large-scale 1D Amin Recession Optimization

## Task characteristics
- **Data size**: (4194304,), very large 1D data

## Optimizing 1: Quadrant

```python
pid = tl.program_id(0)
for start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    offsets = pid * BLOCK_SIZE + start + tl.arange(0, SUB_BLOCK_SIZE)
```

## Optimization 2: Calculate reorganization

```python
# Error: Simple
row_min = float('inf')
for n_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    curr_min = tl.min(block_data)
    row_min = tl.minimum(curr_min, row_min)

# Correct: Optimization
curr_min = tl.full((SUB_BLOCK_SIZE,), float('inf'), dtype=tl.float32)
for start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    curr_min = tl.minimum(curr_min, block_data)
min_val = tl.min(curr_min)
```

## Autotune Configuration

```python
# (AI core=40)
# 1. Grid = 16 < 40, UB full - > 15.12 us
triton.Config({'BLOCK_SIZE': 262144, 'SUB_BLOCK_SIZE': 16384})

# Grid = 32 < 40, UB full - > 9.61 us best
triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 16384})

# Grid = 32, UB Unused - > 10.29 us
triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 8192})

# 4. Grid = 40, UB full with tails - > 10.17 us
triton.Config({'BLOCK_SIZE': 104858, 'SUB_BLOCK_SIZE': 16384})

# Grid = 64>40, UB full - > 11.64 us
triton.Config({'BLOCK_SIZE': 65536, 'SUB_BLOCK_SIZE': 32768})
```

### Summary
Grids are close to the number of AI Cores, the best performance when UBs are full and no tailings. A secondary cut avoids exceeding the hardware cache.
