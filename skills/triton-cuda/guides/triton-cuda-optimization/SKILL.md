---
name: triton-cuda-optimization
description: "Triton CUDA Performance Optimizing Universal Policy, API Limit Description and Debugging Skills Summary. This applies to the generation and optimization of kernel codes that need to be upgraded to the GPU internal nuclei, that need to be checked in case of a compilation/run error, or that need to know the CUDA platform limitations"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
structure:
  child_skills:
    - triton-cuda-memory
    - triton-cuda-grid-config
    - triton-cuda-debugging
---

# Triton CUDA Performance Optimization Guide

## 1. Performance Optimization Policy

### 1.1 Block size selection

- **Principle**: Balancing parallelity and resource occupation
- **Recommendation**: Use 2 quails (256, 512, 1024)
- **GPU Consider**: Need enough warp to hide latency

### 1.2 Warp and Stage Modifier

Two important parameters specific to CUDA backend:

- **num_warps**: number of warps per block (each warp = 32 threads)
  - Small BLONK_SIZE: less used warp (2-4)
  - Large BLONK_SIZE: More used warp (4-8)
  - MatMul: Usually 4-8 warps

- **num_stages**: Software pipeline Class Number
  - More stage to better hide memory latency
  - But it'll take more shared memory.
  - Usually choose between 2 and 5

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=4),
    ],
    key=['n_elements'],
    restore_value=['output_ptr'],  # Must: list all output pointer parameter names
)
```

### 1.3 Memory access optimization

- **Merge access**: threads within the same warp should access the continuous memory address
- **2D data**: Prefer `tl.make_block_ptr` to `boundary_check`
- **step design**: careful design of stride parameters, error setting will seriously affect performance
- **Data layout**: continuity and locality of memory access maintained

### 1.4 operator splitting policy

- **Complex operator**: Split into simple kernels to avoid individual kernels being too complicated
- **Integration strategy**: Moderate integration to reduce global memory reading and writing (e.g. used integration)
- **Balance**: CUDA backend integration is usually more effective than NPU, but still requires attention

### 1.5 Occupancy Optimization

GPU utilization is a key indicator of performance:

- **Repositor used**: Reduced usage of register per thread, increased number of blocks distributed
- **shared memory**: Rational use of shared memory, not exceeding hardware limitations
- **Block Size**: Select the block size that will remove the maximum number of SM threads

## 2. Numerical stability

### 2.1 Spill-proofing

**Softmax numerical stabilization**:
```python
# Minus maximum value to prevent exp spill
max_val = tl.max(scores, axis=0)
scores = scores - max_val
p = tl.exp(scores)  # CUDA backendDirect Use tl.exp
```

### 2.2 At the beginning of the defence value

```python
# Ensure non-negative before variance is calculated
variance = tl.maximum(variance, 0.0)
std = tl.sqrt(variance + eps)
```

### 2.3 accuracy Upgrade

- **excrete with float32**: even if input is float16/bflota16
- **Final re-conversion**: returns target accuracy after calculation is completed
- **TF32**: Ampere+GPU available on TF32 Accelerating MatMul

```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
# Aggregated calculation...
result = tl.cast(accumulator, output_dtype)
```

## 3. API Usage Limit

### 3.1 Use of the use of grammar

**Ban on use**: `return`, `break`, `continue`, `lambda`

The Triton kernel is a one-time complete logic that does not support early return or jumpovers.

```python
# Error: use return
@triton.jit
def kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return  # Compiler error!

# Correct: use mask
@triton.jit
def kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    data = tl.load(ptr + offsets, mask=mask, other=0.0)
    # All codes are executed at the same level.
```

### 3.2 tl.constexpr Correct use

- **For kernel parameters only**: `BLOCK_SIZE: tl.constexpr`
- **Not available on host side**: tl.constexpr not available in startup function

### 3.3 Output tensor Creation Code

- Correct: Use `torch.empty` or `torch.empty_like`
- Error: Avoid `torch.zeros` or `torch.ones` (avoid unnecessary initialization costs)

### 3.4 Attention to the preparation of the Conv volume operator

The volume operator generation in the Torch Modeule will contain a random weight weight right ahead, which needs to be generated in the host side code to ensure that the results are consistent:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triton_kernel():
    pass

def triton_host():
    args = ...
    weight = nn.Conv2d(**args).weight.to(device)
```

The specific parameter and module called in `nn` are to be aligned with the torch, setting Device to `"cuda"`. The same random torrent will be fixed before calling Triton, just create examples of the class correctly and export weights.

## 4. Performance Checklist

### Memory Access
- [ ] Do memory access merge (codesced)?
- [ ] Did you use 2D block_ptr to optimize multi-dimensional data access?
- [ ] Do continuity of memory visits are ensured?

### Parallel Configuration
- [ ] BLONK_SIZE is a 2-year-old?
- [ ] Are the num_warps reasonable (2-8)?
- [ ] Are num_stages reasonable (2-5)?

### Design operator
- [ ] Do you need to split the complex operator?
- [ ] Is it reasonable to use operator integration?
- [ ] Did you use autotune?

### Numerical stability
- [ ] Does the Reduce operation have spill protection?
- [ ] Do you want to use float32 for intermediate accumulation?
- [ ] Have border situations, such as zero, negative-numbered openings, been addressed?

## Summary of best practice

1. **Autotune**: best search using autotune BLONK_SIZE, num_warps, num_stages
2. **RAM Merge**: Ensure access to a continuous address by the same warp inner-space
3. **Tensor Core**: MatMul Class operator enabled all_tf32
4. **pipeline**: Hide Memory latency by Num_stages
5. **Value stable**: use float32 cumulative, minus maximum spill protection
6. **Occupancy**: Balance repository and shared memory use
