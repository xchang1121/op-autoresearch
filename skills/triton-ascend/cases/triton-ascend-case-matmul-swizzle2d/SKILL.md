---
name: triton-ascend-case-matmul-swizzle2d
description: "Big matrix multiplicationSwizzle2D Optimization: Fixed Core Start (grid = 20 instead of all blocks) + Swizzle2D block reordering (GROUP_SIZE = 4) enhances the cache locality, selects group orientations according to M/N ratios, and applies to Ascend NPU scenarios for large-scale matrix multiplication (tens of millions of elements)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# matrix multiplication Swizzle2D optimisation case

## Task characteristics
- **Operating type**: matrix multiplication A[M, K]@B[K, N] = C[M, N]
- **Data size**: A [2048, 7168] @ B [7168, 16384] = C [2048, 16384]
- **Characteristics**: computational intensive, with a significant impact of the core distribution strategy on the C. C. V. and load balance

## Optimization 1: Fixed core number activated (most important!)

### Error: Error: Start all blocks
```python
grid = (NUM_BLOCKS_M * NUM_BLOCKS_N,)  # Start1024A procedure
```

### Correct: Correct: Fixed core number activated
```python
num_cores = 20  # Ascend 910B4Yes.20individualAI Core

@triton.jit
def matmul_kernel(..., num_cores: tl.constexpr):
    pid = tl.program_id(axis=0)  # 0~19
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # Multiple blocks per core cycle
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # Handle Block...
        pass

matmul_kernel[(num_cores,)](...)  # grid=(20,)
```

**Core point**: Ascend NPU must be activated using a fixed core number, with each core cycle processing multiple blocks.

## Optimize 2: Swizzle2D block reordering

```python
@triton.jit
def matmul_kernel_swizzle2d(..., GROUP_SIZE: tl.constexpr, DIRECTION: tl.constexpr):
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        if DIRECTION == 0:  # M≥N: Line Priority Grouping
            task_m_idx, task_n_idx = tl.swizzle2d(
                block_m, block_n, NUM_BLOCKS_M, NUM_BLOCKS_N, GROUP_SIZE
            )
        else:  # M<N: Column priority grouping (manually achieved)
            size_gj = GROUP_SIZE * NUM_BLOCKS_M
            group_id = block_idx // size_gj
            off_n = group_id * GROUP_SIZE
            cur_size_g = tl.minimum(NUM_BLOCKS_N - off_n, GROUP_SIZE)
            local_ij = block_idx % size_gj
            task_m_idx = local_ij // cur_size_g
            task_n_idx = off_n + local_ij % cur_size_g
```

### Optimizing content
- Swizzle2D rearrange blocks by grouping through GRUP_SIZE and share data among groups
- GRUP_SIZE recommended value 4 can be searched for atotonne [1,2,3,4,5,5]

## Optimization 3: Matrix shape self-adaptation

```python
DIRECTION = 1 if m < n else 0  # M<NColumn Priority, M≥NLine Priority
```

- **M≥N**: line priority grouping less mat_a load
- **M<N**: column priority groups reduced by mat_b load

## Optimize 4: Select the size of the segment

```python
# float16/bfloat16
BLOCK_M, BLOCK_K, BLOCK_N = 128, 256, 256

# float32
BLOCK_M, BLOCK_K, BLOCK_N = 128, 128, 128
```

### Summary
1. **Fixed core start**: `grid=(20,)`, multiple blocks per core cycle
2. **Swizzle2D Reorder**: Cache locality raised by block grouping
3. **From the adaptation cluster orientation**: selection of rows or rows according to M/N ratio
4. **Appropriate fraction size**: selection based on data type and cache capacity
