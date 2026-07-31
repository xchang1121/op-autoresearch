---
name: triton-cuda-basics
description: "Triton CUDA programming base, including core concepts (program_id, block, Grid), kernel structure, decorator usage and standard code model. A scenario for generating any CUDA nuclear code that uses Triton CUDA and requires basic syntax structure"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "all"
---

# Triton CUDA programming base

## 1. Core concepts

### Kernel
- **Definition**: using the Python function of `@triton.jit` decoration, compiled and executed in parallel on GPU
- **Characteristics**: A subset of data processed for each kernel example, distinguished by application ID

### Grid (Grid) and Block (Block)
- **Grid**: Parallel dimensions configuration at kernel start-up, e. g. `(num_blocks_x, num_blocks_y)`
- **Block**: Size of data blocks processed for each application instance, e.g. `BLOCK_SIZE = 1024`
- **Relationship**: `grid_size = ceil(total_elements / block_size)`

### Memory Level
- **global memory (Global Memory)**: Main Memory (HBM), accessible, latency high, bandwidth large
- **shared memory (Shared Memory)**: SM in-house sharing, latency low, limited capacity (usually 48-164 KB/SM)
- **Registers**: private, quickest access for each thread

### CUDA GPU Structure Elements
- **SM (Streaming Multiprocessor)**: GPU Basic Calculator Unit
- **Warp**: 32 threads for parallel implementation
- **Tensor Core**: Dedicated Matrix Calculating Module (Ampere/ Hopper Architecture)

## 2. Standard kernel structure (five-step model)

All Triton kernels follow the same five-step structure:

```python
@triton.jit
def standard_kernel(
    output_ptr, input_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Acquiring program ID and calculating offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 2. Creation of a border mask
    mask = offsets < n_elements

    # 3. Loading data
    data = tl.load(input_ptr + offsets, mask=mask)

    # 4. Implementation calculations
    result = compute_function(data)

    # 5. Storage results
    tl.store(output_ptr + offsets, result, mask=mask)
```

## 3. kernel startup mode

### Function Form
```python
def launch_kernel(input_tensor, output_tensor):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(input_tensor.numel(), BLOCK_SIZE),)

    kernel[grid](
        output_tensor, input_tensor, input_tensor.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

### ModelNew class format (recommended)
```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        output_tensor = torch.empty_like(input_tensor)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(input_tensor.numel(), BLOCK_SIZE),)

        kernel[grid](
            output_tensor, input_tensor, input_tensor.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output_tensor
```

## 4. Border processing

### Use mask to process borders
```python
# Basic border checks
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### Conditional calculation
```python
# Use tl.where to make a condition selection
result = tl.where(condition, true_value, false_value)

# A combination of complex conditions
valid_mask = (offsets < n_elements) & (offsets >= 0)
data = tl.load(ptr + offsets, mask=valid_mask, other=0.0)
```

## 5. Autotune Usage (static Shape only)

Autotune finds the optimal configuration and cache of the current hardware and data size by auto-benchmark multigroup configuration parameters, without manual referencing.

### Apply scene

- **Recommended**: input shape fixed or limited range of changes (static shape), e. g. MatMul for fixed bat size, Attention for fixed sequence length, etc.
- **Ban on use**: Enter Shape Frequent Changes (Dynamic Shape). Autotune best config based on `key` parameters cache, dynamic Shape triggers a full benchmark with a severe drag on chronic energy

### Mandatory rules

1. **Must write `restore_value`**: list all**output pointer parameters**for kernel. uututune benchmark will repeat kernel, `restore_value` will save and output a copy of tensor before each config and restore values after each traverse to prevent contamination of results between different configs.**Failure to write `restore_value` will result in certification failure.**
2. **grid must use lmbda**: `grid = lambda meta: (...)` to ensure that grid can be calculated according to current config dynamics.
3. **Call without calling configs parameters**: autotransmittune.
4. **configs must be constexpr**: declared `PARAM: tl.constexpr` in Kernel.
5. **key Parameter**: reautonne when you specify which input dimensions change.
6. **num_warps**: Controls the number of warps per block (common: 2, 4, 8).
7. **num_stages**: Control software pipeline class (common: 2, 3, 4, 5).

### Standardized

```python
# Correct: There is restore_value
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
    restore_value=['c_ptr'],  # ⚠ Must: list all output pointer parameter names
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pass

grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
matmul_kernel[grid](a, b, c, M, N, K, ...)
```

```python
# Error: Lack of resource_value → CodeChecker intercepts, authentication fails
@triton.autotune(
    configs=[...],
    key=[...],
)
@triton.jit
def kernel(input_ptr, output_ptr, ...):
    pass
```

## best practice

1. **Mask**: addressing border situations and preventing cross-border visits
2. **Rational choice BLONK_SIZE**: balancing parallelity and resource occupancy (recommends 2 quartiles)
3. **Use constexpr**: compile time constants, improve performance
4. **Note: data type**: Visible type conversion to avoid accuracy losses
5. **Use autotune**: Automatically find optimal configurations (including num_warps and num_stages)
6. **Allow_tf32 enabled with Tensor Core**: MatMul class operator
