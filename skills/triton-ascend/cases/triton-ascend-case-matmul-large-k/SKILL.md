---
name: triton-ascend-case-matmul-large-k
description: "matrix multiplication matrix multiplication A [M, K]@B[K, N]=C[M, N] Large K-dimensional matrix multiplication(K>M, N) Optimization: for a scene of M/N smaller but large K (e.g. M=N=256, K=131072), Split-K cut K-dimensional parallelization, Workspace+Reduce replacement global synchronization to achieve significant performance improvement"
category: deprecated
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3, Atlas A5"
---

# Large K-dimensional matrix multiplication optimization cases

## Task characteristics
- **Operating type**: matrix multiplication A[M, K]@B[K, N] = C[M, N]
- **Typical data size**: A [256, 131072] @ B [131072, 256] = C [256, 256]
- **Characteristics**: K is much greater than M and N (K/M = 512 times), output blocks are much less than the core, and conventional matmul core utilization is low

### Core issues

```
M=256, N=256, K=131072, BLOCK_M=64, BLOCK_N=64:
  Number of output blocks = ceil(256/64) × ceil(256/64) = 4 × 4 = 16
  Available Quantities = 32
  → 16 Blocks < 32 Nuclear, Half the nuclear space.!
  → Every nuclear. K-loop = 131072/256 = 512 Numbers, The single nucleus is huge.
```


## Optimize 1: Split-K + Atomic Add

### Rationale

When the number of output blocks < core number, the K dimension is divided into the `SPLIT_K` section, allowing multiple nuclears to calculate the different K sector of the same output block in parallel, adding the divided partial results to C with `tl.atomic_add`. In addition, adjusting the core number if the `SPLIT_K` parameter is placed in the Grid.

```python
# grid = (NUM_MN_BLOCKS, SPLIT_K)
# For example: AI_Cude=32, M=N=256, BLONK=128: NUM_MN_BLONKS=2*2 =4
# grid = (4,16) → 64, 32 nuclear processed 2 pieces of data per nuclear
@triton.jit
def matmul_splitk_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)       # Output Block ID
    split_id = tl.program_id(1)  # K Subparagraph ID

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_idx in range(k_block_start, k_block_end):
        a = tl.load(A_ptr + ...)
        b = tl.load(B_ptr + ...)
        acc += tl.dot(a, b)

    # Atoms plus: multiple sprit parties add directly to C
    tl.atomic_add(C_ptr + ..., acc, mask=...)
```

### Core elements
- The grid number should be configured close to or above the core number to ensure full coverage
- The more the `SPLIT_K` goes, the more the tomic_add competes.

## Optimizing 2: Workspace + Reduce

### Rationale

Global sync (e.g., `tl.debug_barrier`) allows all cores to wait at the same point, amounting to complete serialization of CUBE calculations and VEC returns, with extremely poor performance. Unlike AscendC, which has AIC/AV hardware in parallel, it writes CUBE results directly to the workspace, and then calls for Reduces to return. Also, workspace should be as full as possible and not oversized.

```python

@triton.jit
def matmul_splitk_to_ws_kernel(A_ptr, B_ptr, WS_ptr, M, N, K, ...,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    # K-Dept Counts...
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_idx in range(k_block_start, k_block_end):
        acc += tl.dot(tl.load(A_ptr + ...), tl.load(B_ptr + ...))

    # Go straight to workspace, don't make any returns.
    tl.store(WS_ptr + split_id * stride_ws_s + ..., acc, mask=...)

# hostend
...
# Return
C = torch.sum(workspace, dim=0)
```

### Core elements
- CUBE in Triton and VEC cannot really go through AIC/AIV hardware circuits like AscendC
- `tl.debug_barrier` Universal Synchronizes all nuclear barriers, equal to serialization, with the worst performance
- Referring to the kernel external use `torch.sum`, avoiding a nuclear CUBE-VEC serial problem, which is measured more than 1 times faster than the global sync scheme

## Summary

For the matrix multiplication scene of K far greater than M/N (e.g. M=N=256, K=131072), three optimized combinations are available:

2. **Split-K + Atomic Add**: Split K-dimensional to the grid outer dimension, multi-nucleus to handle different K segments of the same output block in parallel, with `tl.atomic_add` cumulation.
3. **Workspace + Reduce**: Split-K paragraphs write to workspace and use `torch.sum` external returns to avoid nuclear-wide synchronization problems. More than 1 times faster than the `debug_barrier` scheme
