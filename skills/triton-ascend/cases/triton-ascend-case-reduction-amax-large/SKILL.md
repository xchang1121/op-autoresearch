---
name: triton-ascend-case-reduction-amax-large
description: "Non-reduce axis small and reduce axis large-scale alignment optimization: mapping reduce axis to multiple cores (rather than conventional non-reduce axis), using atomic operation cross-line components to contract, avoiding ultra-UB by a secondary split, and applying to the return scenario of extreme sape ratio (M<N <N like 16×262144)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Large-scale Amax contract optimization (reduce axis mapping multiple cores)

## Task characteristics
- **Data size**(16,262144), nonreduce axis small, reduce axis large
- **Strategy**: mapping reduce axis to polynucleus, using atomic operations

## Optimizing 1: Cut Policy Adjustments

```python
# Error: Simple way: nonreduce axis map multiple core
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)

# Correct: Optimization: reduce axis map multiple cores
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)

# Retract columns within Kernel
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
```

## Optimizing 2: Atomic Operations

### Option I: Atom Operations in Cycle
```python
for m_start in range(0, M, BLOCK_SIZE_M):
    row_min = tl.min(curr_min, 1)
    tl.atomic_min(output_ptrs, row_min, mask=mmask)
```

### Option II: Outer-cycle atomic operations
```python
all_row_min = tl.full((M,), float('inf'), dtype=tl.float32)
for m_start in range(0, M, BLOCK_SIZE_M):
    row_min = tl.min(curr_min, 1)
    all_row_min = tl.insert_slice(all_row_min, row_min, ...)
tl.atomic_min(output_ptrs, all_row_min)
```

## Optimization 3: Configure

```python
@triton.autotune(
    configs=[
        # Grid = 32 < 40, UB full
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 1024}),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 512}),
    ],
    key=[...],
    restore_value=['out_ptr0'],  # autotune I have to. restore_value
)
```

### Summary
When the nonreduce axis is small and the reduce axis is large, reduce axis is mapped to multiple cores and combined with atomic operations to avoid exceeding the UB by a secondary split.
