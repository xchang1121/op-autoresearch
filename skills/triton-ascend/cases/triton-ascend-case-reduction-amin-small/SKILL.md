---
name: triton-ascend-case-reduction-amin-small
description: "Moderate 1D integration (amin) optimization: Moderate parallelity (grid = best 2.21us at 8), existence of optimal balance point (too small, resulting in overloading of individual loads, overloading of movement control costs) and full return scenario for medium size 1D data (60 000 elements)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Medium Size 1D Amin Recession Optimization

## Task characteristics
- **Data size**(65536,), medium size 1D

## Optimization: moderate parallelity

```python
# (AI core=40)
# 1. grid=4<40 -> 2.47 us
triton.Config({'BLOCK_SIZE': 16384})

# 2. Grid = 8<40 -> 2.21 us best
triton.Config({'BLOCK_SIZE': 8192})

# 3. grid=32<40 -> 2.92 us
triton.Config({'BLOCK_SIZE': 2048})

# 4. grid=40 -> 3.44 us
triton.Config({'BLOCK_SIZE': 1639})

# 5. grid=128>40 -> 6.70 us
triton.Config({'BLOCK_SIZE': 512})
```

### Summary
Medium-sized data with the optimal number of appropriate hours of grids. There are optimal parallel balance points: too small grids lead to overloading of a single block, and too large introduces excessive dispatch costs.
