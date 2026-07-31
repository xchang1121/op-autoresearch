---
name: triton-cuda-matmul
description: "matrix multiplicationoperator (matmul/bmm/linear) optimisation strategy, including partition Tiling, shared memory Cache, Tensor Core Use and Large Matrix Processing Techniques. Application to CUDA kernel generation scenarios for the implementation of matrix calculations such as GEMM, batch matrix multipliers, full connectivity layers"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "matmul"
  algorithms: "matmul, bmm, linear"
---

# MatMul operator Optimization

> Applicable to matrix multiplication and related operations

## CUDA GPU MatMul Optimizing Core

### Tensor Core

- **Ampere (A100)**: support FP16, BF16, TF32, INT8 Tensor Core
- **Hopper (H100)**: extra support FP8, wgmma command
- **Key**: `tl.dot(a, b, allow_tf32=True)` enabled TF32 Tensor Core

### Partition Configuration Proposal

Common configuration (2 times):

| Configure | BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages | Apply scene |
|------|---------|---------|---------|-----------|------------|---------|
| Small Matrix | 64 | 64 | 32 | 4 | 4 | M, N < 1024 |
| Medium Matrix | 128 | 128 | 32 | 4 | 3 | M, N < 4096 |
| Large Matrix | 128 | 256 | 64 | 8 | 3 | M, N >= 4096 |
| High king | 64 | 128 | 64 | 4 | 4 | K, it's big. |

## Standard MatMul Kernel (using block_ptr)

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 2D index calculation
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create block points
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )

    # Use float32 loader
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K-dimensional cycle
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)

        # Move block points
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    # Storage results (required for visible conversion type, matching output dtype)
    c = accumulator.to(c_ptr.dtype.element_ty)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))
```

## Optimize using Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
    restore_value=['c_ptr'],  # Must: list all output pointer parameter names
)
@triton.jit
def matmul_kernel_autotune(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # L2 Cache Optimization: Grouped ordering
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Follow-up is the same as the standard Kernel
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))
```

## Host Side Start

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

        matmul_kernel_autotune[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
        return c
```

## L2 Cache Optimization: Grouped Ordering

### Grouped Ordering?

The standard row or row priority flow leads to low utilization of the L2 cache. By grouping adjacent blocks, data can be reused:

```python
# Standard Pass: adjacent pid access to block A in different rows
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

# Grouped ordering: adjacent pid access to block A of the same group
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

### Swizzle2D

Another way to optimize the cache:
```python
task_m, task_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)
```

## Elements of optimization

### 1. Part Configuration

- Search for optimal configuration using autotune
- Consider Tensor Core requirements (number of blocks of 16)
- A larger piece of → better data reuse, but higher memory pressure

### 2. accuracy control

- Thrusters use float32: `tl.zeros(..., dtype=tl.float32)`
- Even if input is fp16/ bf16, add with fload32
- Automatically return target accuracy on final storage

### 3. Memory Access

- Prefer `tl.make_block_ptr` and `boundary_check`
- Move block pointer with `tl.advance`
- Optimizing the L2 cache using Grouped Ordering

### 4. pipeline

- `num_stages` Control Software pipeline Levels
- More stage → better hide memory latency
- But it'll take more shared memory.

## Performance Checklist

- [ ] Did you use autotune to search for optimal configuration?
- [ ] Does the compulsor use float32?
- [ ] Grouped Ordering or swizzle2d optimized L2 caches?
- [ ] Is the K-dimensional cycle correctly achieved?
- [ ] Does it make sense for num_warps and num_stages?
- [ ] block size is 16 multiples (Tensor Core requirement)?

## Common Errors

1. **Additional fp16**: accuracy serious loss
2. **Forget K-dimensional loop**: Result error
3. **block not the same size Tensor Core**: poor performance
4. **L2 Cache not optimized**: Large Matrix Decline
5. **num_warps mismatch**: block size and warp number mismatch leads to waste of resources
