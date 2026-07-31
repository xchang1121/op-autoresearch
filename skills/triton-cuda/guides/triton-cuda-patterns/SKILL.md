---
name: triton-cuda-patterns
description: "Triton CUDA Standard Realization Template and Code Template for the three core programming models (vector/Element by Elements, Resignation, matrix multiplication). This applies to CUDA internal nuclear code generation scenarios that need to quickly determine which type of programming mode operator belongs to or that need to understand the structure of the basic code of each model."
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "elementwise, reduce, matmul"
structure:
  child_skills:
    - triton-cuda-elementwise
    - triton-cuda-reduce
    - triton-cuda-matmul
---

# Triton CUDA programming mode

## 3.1 vector operating mode

For element-level operations: Adding, Multiplication, Activation Functions, etc.

### Standard code structure

```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)
```

### Apply operator
- Algorithmic Operations: add, Mul, sub, div
- Activate function: relu, sigmoid, taunh (required with `tl.extra.cuda.libdevice.tanh`), gelu
- Mathematical functions: ext, log, sqrt, pow

### Key points
- Use 1-D index and offset
- Border processing with `mask`
- Simple direct data stream: Load → calculate → storage

## 3.2 Models of return

Applies to sum, max, min.

### Standard code structure

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Loading data
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Internal Convention
    block_sum = tl.sum(data, axis=0)

    # Atomic Operations Return global memory
    tl.atomic_add(output_ptr, block_sum)
```

### Apply operator
- Basic affiliation: sum, mean, max, min
- Normalize: softmax, logsoftmax, playnorm, watchnorm
- Statistics: varance, std

### Key points
- Block internal union: use `tl.sum`, `tl.max`, etc.
- Atomic Operations: Write global memory back using `tl.atomic_add` etc.
- Numerical stability: minus maximum value to prevent spill (see Triton-cuda-reduce)

## 3.3 matrix multiplication mode

For multi-dimensional block calculations such as matrix multiplication.

### Standard code structure

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Acquire program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Initialise accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K-dimensional cycle
    for k in range(0, K, BLOCK_SIZE_K):
        # Create a block pointer
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, k),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(k, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
        )

        # Loading data blocks
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        # Matrix multiplied by cumulative
        accumulator += tl.dot(a, b)

    # Storage results (required for visible conversion type, matching output dtype)
    c = accumulator.to(c_ptr.dtype.element_ty)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))
```

### Host Side Start

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
        )
        return c
```

### Apply operator
- Matrix operation: matmul, bmm (batch matmul), linear
- Volume: conv2d, conv3d
- Other multi-dimensional calculations

### Key points
- **2D Grid**: using `grid=(grid_m, grid_n)` 2D parallel
- **Branch calculation**: break-down of large arrays into small blocks to reduce memory occupancy
- **K dimension cycle**: multi-part product added
- **block_ptr**: Simplify 2D data access using `tl.make_block_ptr`
- **Tensor Core**: Auto-use Tensor Core with `tl.dot`

## Mode Selection Guide

| operator Type | Recommended Mode | Key features |
|---------|---------|---------|
| Element-wise | vector operating mode | Element-by-Element calculation |
| Reduction | Reunification Mode | Multiple values need to be aggregated |
| MatMul/Conv | matrix multiplication mode | Multi-dimensional block calculations, 2D Grid |
| Attention | Convention +matrix multiplication | Group mode, see Triton-cuda-attention |

## best practice

1. **Select the appropriate mode**: Select the base mode based on operator characteristics
2. **Optimizing block size**: balancing parallelity and resource occupancy
3. **Note boundary**: use mask to handle irregular shape
4. **Numerical stability**: special attention for reduce class operator
5. **Memory access**: Optimization of data layout to increase Cache Rate
6. **Using Hardware Features**: Optimizing pipeline using num_warps/num_stages
