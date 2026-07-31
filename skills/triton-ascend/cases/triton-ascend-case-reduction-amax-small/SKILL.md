---
name: triton-ascend-case-reduction-amax-small
description: "Mini-small-scale contract conversion (max) optimization: single-nuclear processing (grid = 1) is better than multi-nuclear parallel (2.16us vs. 3.51us), avoids the movement costs associated with parallelization and applies to the return scene of very small data size (<1,000 elements)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Mini Small Amax Recession Optimization

## Task characteristics
- **Data size**(16, 16), very small
- **Strategy**: single-nucleus treatment is better than multi-nucleus

## Optimization: single/small nuclear processing

```python
# Single nuclear processing - > Performance: 2.16 us best
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16})

# Multi-nuclear parallel - > Performance: 3.51 us
triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16})
```

### Summary
For small-scale computing missions, mononucleus (nucleus) processing is more efficient and the cost of parallelization outweighs the benefits.
