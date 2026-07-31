---
name: triton-ascend-case-reduction-mean-large
description: "Large-scale reduce final axis (mean) line double-string optimization: a kernel calculates the number of reduction lines in multiple rows, a kernel double-string avoids ultra-UB, grid = 40 and the SUB cut does the best performance when it does not contain tail blocks (16.00us), and the end block calculation significantly reduces performance and applies to the nonreduce axis medium, larger 2D engagement scenario"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Large-scale Mean Convention Optimization (line two-steps)

## Task characteristics
- **Data size: (1,000,8192), nonreduce axis medium, reduce axis larger

## Optimization: rows split diagonally

```python
pid = tl.program_id(0)
for m_start in range(0, BLOCK_SIZE_M, SUB_BLOCK_SIZE_M):
    m_offsets = pid * BLOCK_SIZE_M + m_start + tl.arange(0, SUB_BLOCK_SIZE_M)
```

**Purpose**
- Each kernel calculates multiple lines (BLONK_SIZE_M), reducing the number of bus blocks
- Double split of rows in Kernel (SUB_BLONK_SIZE_M) to avoid exceeding the hardware cache

## Autotune Configuration

```python
# (AI core=40)
# 1. grid<40 -> 28.64 us
triton.Config({'BLOCK_SIZE_M': 50, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 512})

# grid = 40, SUB stegregated with tails - > 16.54 us
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096})

# Grid = 40, SUB cut without tails - > 16.00 us best
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 512})

# 4. Grid > 40, integer - > 25.86 us
triton.Config({'BLOCK_SIZE_M': 20, 'SUB_BLOCK_SIZE_M': 20, 'BLOCK_SIZE_N': 512})
```

### Summary
1. Grid equals nuclei. SUB cut without tails is the best.
2. The end block calculation reduces performance.
3. When Grid exceeds the number of cores and the number of non-nuclei is multiple, the computational missions are uneven and have poor performance
