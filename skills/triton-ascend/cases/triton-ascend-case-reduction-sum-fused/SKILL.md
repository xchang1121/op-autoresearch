---
name: triton-ascend-case-reduction-sum-fused
description: "Reduction + Elementwise integration operator optimization: first on an element-by-element basis, second-step + calculation of reorganization, grid = 40 and best performance when SUB cut without tail (47.58us), integration optimization logic based on reduce, applicable to integration scenarios that require element-by-component calculation before reduce"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Reduction + Elementwise Integration operator Optimization

## Task characteristics
- **Data size**: (1,000, 8192), (8192), integration of operator
- **Characteristics**: vector element-by-element operation, followed by column orientation and return

## Optimizing 1: a two-dimensional split of lines

```python
pid = tl.program_id(0)
for m_start in range(0, BLOCK_SIZE_M, SUB_BLOCK_SIZE_M):
    m_offsets = pid * BLOCK_SIZE_M + m_start + tl.arange(0, SUB_BLOCK_SIZE_M)
```

## Optimization 2: Calculate reorganization

```python
# Error: Simple
total_sum = 0.0
for n_offset in range(0, N, BLOCK_SIZE):
    total_sum += tl.sum(tl.where(mask, t3, 0.0))

# Correct: Optimization
acc = tl.zeros([SUB_BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    acc += tl.where(mask, t3, 0.0)
total_sum = tl.sum(acc, axis=1)
```

## Autotune Configuration

```python
# (AI core=40)
# 1. grid=20<40 -> 91.69 us
triton.Config({'BLOCK_SIZE_M': 50, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256})

# grid = 40, SUB stegregated with tails - > 53.30 us
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 2048})

# 3. Grid = 40, SUB cut without tails - > 47.58 us best
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256})

# 4. Grid>40, integer number - > 79.00 us
triton.Config({'BLOCK_SIZE_M': 20, 'SUB_BLOCK_SIZE_M': 20, 'BLOCK_SIZE_N': 256})
```

### Summary
The logic of integration of operator optimization is dominated by reduce. Grid equals the number of nuclei, and SUB cut without tails is the best.
