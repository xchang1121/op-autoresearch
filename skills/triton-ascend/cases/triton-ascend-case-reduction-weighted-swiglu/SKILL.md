---
name: triton-ascend-case-reduction-weighted-swiglu
description: "3D integration of operator (Weighted SwiGLU Backward) Optimization: Reshape downsizes the first two dimensions to simplify the parallel strategy, avoids hyperUB, assigns larger fraction sizes to reduce axes with priority for full UB, and the greater likelihood of grid having larger numbers can be applied to 3Dtensor-by-Element +reduce integration"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Weighted SwigLU Backward Integration operator Optimization

## Task characteristics
- **Data Dimensions**:(16, 1024, 2048) × 3,3DIntegrationoperator
- **Characteristics**: Element-by-Element operation, last axis of reduce

## Optimize 1: Reshape dimension

```python
# Merge the first two dimensions (B & M) into a single dimension BM
x_reshaped = x.reshape(BM, N)
weight_reshaped = weight.reshape(BM, N)
grad_reshaped = grad.reshape(BM, N)

weighted_x_reshaped = weighted_x.reshape(BM, N)
grad_weight_reshaped = grad_weight.reshape(BM, N)
grad_x_reshaped = grad_x.reshape(BM)
```

**Strength**: streamline parallel strategies, optimize memory access models and improve the efficiency of kernel implementation.

## Optimizing 2: Line Double-Stract

```python
for bm_start in range(0, BLOCK_SIZE_BM, SUB_BLOCK_SIZE_BM):
    bm_offsets = pid * BLOCK_SIZE_BM + bm_start + tl.arange(0, SUB_BLOCK_SIZE_BM)
    bm_mask = bm_offsets < BM
```

## Autotune Configuration

```python
# (AI core=40)
# 1. Grid = 512 > 40, reduce axle smaller, UB full - > 1105.84 us
triton.Config({'BLOCK_SIZE_BM': 32, 'SUB_BLOCK_SIZE_BM': 32, 'BLOCK_SIZE_N': 128})

# Grid = 1024>40, reduce axle split to 256 - > 1110.47 us
triton.Config({'BLOCK_SIZE_BM': 16, 'SUB_BLOCK_SIZE_BM': 16, 'BLOCK_SIZE_N': 256})

# Grid = 2048>40, reduce axle split to 512 - > 1091.26 us best
triton.Config({'BLOCK_SIZE_BM': 8, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512})

# Grid =32 < 40, reduce axle is larger and UB is full - > 1098.53 us
triton.Config({'BLOCK_SIZE_BM': 512, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512})

# Grid = 40 with tails, reduce axles are larger, UB is full - > 1094.60 us
triton.Config({'BLOCK_SIZE_BM': 416, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512})
```

### Summary
1. Reshape's down dots can simplify parallel strategies and optimize memory access
2. Allocation of larger cut sizes for reduce axis with priority for full UB.
3. When Grid is larger, the likelihood is better (configuration 3)
