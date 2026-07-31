---
name: triton-cuda-memory
description: "CUDA GPU memory access optimization strategy, including shared memory usage, combined access, Bank Conflict avoidance and data layout optimization techniques. Application to memory bandwidth restricted, need to optimize global memory access efficiency, or to process large-scale data for CUDA kernel performance optimization"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# Memory access optimization

Memory access is a key bottleneck for GPU performance. This document provides the Triton CUDA memory access optimization policy.

---

## 1. GPU Memory Level

### Memory bandwidth and latency

| Memory Type | bandwidth (A100) | latency | Capacity |
|---------|-------------|------|------|
| Organisation | ~19 TB/s | 1 cycle | 256 KB/SM |
| shared memory | ~19 TB/s | ~20 cycles | 164 KB/SM |
| L2 Cache | ~5 TB/s | ~100 cycles | 40 MB |
| global memory (HBM) | ~2 TB/s | ~400 cycles | 40/80 GB |

### Optimization principle

- **Reduced global memory access**: using shared memory and register
- **Consolidated Access**: same warp insider access continuous address
- **Increased L2 Cache Rate**: Through technology such as Grouping

---

## 2. Merge Access (Coalesced Access)

### What's a combined visit?

When 32 threads of the same warp access consecutive memory addresses, GPU can consolidate these requests into one or a few memory services, significantly increasing the utilization of bandwidth.

```python
# Correct: combined access (continuous address)
@triton.jit
def coalesced_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Continuous Offset
    mask = offsets < n
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)

# Error: Non-merger access (jumping address)
@triton.jit
def strided_kernel(input_ptr, output_ptr, n, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Each thread leaps the length elements, leading to non-merger access
    strided_offsets = offsets * stride
    mask = strided_offsets < n
    data = tl.load(input_ptr + strided_offsets, mask=mask)
```

---

## 3. Block Selection Policy

### Preference principle
- **Balancing parallelity and resource consumption**, avoiding oversized or too small
- **BLONK_SIZE**Common value: 128, 256, 512, 1024
- Too small: insufficient parallels to make full use of warp
- Oversized: register/shared memory spill, occupancy drop

### Recommended Settings
- **Element-wise operator**: BLONK_SIZE = 1024 or 512
- **Reduce operator**: BLONK_SIZE = triton.next_power_of_2(n_cols)
- **MatMull operator**: BLONK_M = 128, BLONK_N = 128, BLONK_K = 32-64

---

## 4. 2D Data Memory Access Optimization

### Prefer `tl.make_block_ptr`

For 2D data (e.g. matrix),**priority is given to `tl.make_block_ptr` in conjunction with `boundary_check`**, which automatically optimizes memory consolidation.

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create 2D Block Pointer
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),  # Row-major
    )

    # Automatically handle borders using baseary_check
    a = tl.load(A_block_ptr, boundary_check=(0, 1))
```

### Stride Design Elements
- **Specificly designed stride parameters**, error setting will seriously affect performance
- **Successive visits**: ensuring continuity and locality of memory visits
- **Low main order (Row-major)**: PyTorch default, stride(0) > stride(1)

---

## 5. Optimization of one-dimensional access to a continuous memory

### Recommended option: one-dimensional access after transition

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Conversion from non-continuous tensor to continuous (one-time costs)
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        output_tensor = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        elementwise_kernel[grid](
            input_tensor, output_tensor,
            n_elements,
            BLOCK_SIZE=1024
        )
        return output_tensor
```

### Performance Comparison

| Programme | Advantages | Disadvantages |
|------|------|------|
| **`.contiguous()` + 1-D access** | Merge Access, Cache Friendly | One-time memory copy cost |
| **stride access** | No copy required. | Non-merger access, accumulated costs |

**Recommendation**: The non-continuous tensor first calls the `.contiguous()` conversion, then uses one-dimensional access, and the overall performance is better.

---

## 6. L2 Cache Optimization

### Grouped Ordering

For MatMul et al. 2D operator, increase the L2 Cache Cache Rate by grouping:

```python
# Standard pass: L2 Cache utilization is low
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

# Grouped Ordering: L2 High utilization of caches
GROUP_SIZE_M = 8
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

### swizzle2d

```python
task_m, task_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)
```

---

## 7. Software pipeline (Software Pipelining)

### Num_stages Arguments

Controls the pre-take level by `num_stages`, hides the memory latency:

- **num_stages=2**: Minimum shared memory
- **num_stages=3-4**: usually best
- **num_stages=5+**: May exceed the shared memory limit

```python
@triton.autotune(
    configs=[
        triton.Config({...}, num_stages=2, num_warps=4),
        triton.Config({...}, num_stages=3, num_warps=4),
        triton.Config({...}, num_stages=4, num_warps=8),
    ],
    key=[...],
    restore_value=['output_ptr'],  # Must: list all output pointer parameter names
)
```

---

## 8. best practice

### Element-wise operator
1. Transient: `input.contiguous()`
2. 1-D access: `ptr + offsets`
3. BLOCK_SIZE = 1024

### 2D operator (MatMul, Attention)
1. Use `tl.make_block_ptr`
2. Match `boundary_check`
3. Grouped Ordering Optimizing L2 Cache
4. Rational settings

### A trap to avoid.
- Non-continuous tensor direct access by stide
- BLONK_SIZE setting too big to cause ocupancy to decline
- Forget border checks lead to cross-border visits.
- Ignore L2 Cache Optimization

---

## 9. Debug Recommendations

### Performance screening
1. Check if tensor is continuous: `tensor.is_contiguous()`
2. Check if memory access merges
3. Analyse memory bandwidth utilization using Nsight Computer
4. Check if ccupancy is reasonable

### Common Errors
- **Memory access cross-border**: check mark and baseary_check
- **Poor performance**: check combined access and L2 cache optimization
- **Result error**: Checks for correctness of stride calculation
