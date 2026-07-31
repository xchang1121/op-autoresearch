---
name: triton-ascend-case-elemwise-broadcast-2d
description: "2D broadcasting is optimized: complete processing of small dimensions unseparated (recycling plus load reuse), inter-nuclear parallel (40 nucleus) through stationary NUM_BLONKS, and intra-nuclear SUB_M control particle size balance UB utilization, applicable to 2D scenarios with large but not small broadcast axes"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# 2D Broadcast Disvision Optimization Case

## Task characteristics (two configurations)

### Configure 1: (131072, 16) / (1,16)
- First Axis of Broadcast
- broadcast size (131072), non-broadcast size (16)

### Configure 2: (2048, 131072) / (2048, 1)
- Second axis of Broadcast
- Medium (2048) of broadcast axis, large (131072)

## Optimization 1: Complete processing of small dimensions

```python
# Smaller N-dimensional (N=16), complete uncut
offs_n = tl.arange(0, N)  # N=16

# All lines of divisor shared, loaded out of cycle once
divisor = tl.load(divisor_ptr + offs_n)  # shape: (N,)

# Internal cycle: SUB_M rows per process
for sub_start in range(row_start, row_end, SUB_M):
    offs_m = sub_start + tl.arange(0, SUB_M)
    dividend = tl.load(dividend_ptr + dividend_offs, mask=mask_2d, other=0.0)
    output = dividend / divisor  # divisorRadio: (N,) -> (SUB_M, N)
```

### Optimizing content
- Optimization of UB utilization due to small N-dimensional (N = 16), selection of complete treatment without cut-off
- All lines of divisor shared, loaded out of cycle once, automatically broadcast reused in cycle
- Whether the dimensions are divided, not the broadcast direction.

## Optimize 2: Grid split configuration

```python
# NUM_BLONKS Control Numeric Number, SUB_M Control Internal Processing Lines
triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 512}), # 8.55usThe best, the number.=40With all the physics.
triton.Config({'NUM_BLOCKS': 64, 'SUB_M': 512}), # 9.83usQuantified>40Control costs are high.
triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 256}), # 9.78us,ubUnused
triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 1024}), # Super.ub
grid = lambda meta: (meta['NUM_BLOCKS'],)
```

### Optimizing content
- Control the core number ≤40 by grid cutting in M-dimensional.
- SUB_M = 512 hours to balance UB utilization and memory pressure

## Optimization 3: Universal 2D schedule method

For general 2D Shape, generic method of movement:
1. **Nuclear Parallels (NUM_BLONKS)**: split along MD and distributed to different calculator cores
2. **SUB_M)**: control of the number of lines per process to balance UB utilization
3. **Column vector (BLONK_N)**: Loading along N-dimensional segments to achieve continuous access and vector

### Summary
1. For smaller dimensions, non-ditation should be fully addressed in order to maximize UB utilization
2. Inter-nuclear parallelation by stationary NUM_BLONKS and control of data particles by nucleometric splits
3. Align with autotune parameters
