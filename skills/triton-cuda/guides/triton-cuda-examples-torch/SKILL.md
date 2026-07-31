---
name: triton-cuda-examples-torch
description: "Full integration example of the Triton CUDA kernel under PyTorch framework, including standard operators such as vector_add, matmul, player_norm, softmax. This applies to the CUDA kernel creation scenario that requires reference to PyTorch operator packaging methods, torch.autograd. Function mode"
category: example
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  framework: torch
  examples: "vector_add, matmul, layer_norm, softmax, double_kernel"
---

# PyTorch + Triton CUDA Example Code

This Skill contains a full runable example code showing how Triton CUDA is used in PyTorch to write high performance kernel.

## Example List

### 1. Victor Add (vector plus)
**operator type**: Element-wise
**Key points**:
- The simplest example of Triton Kernel
- 1-D Index and Mask
- Standard five-step model

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton vectorAdd kernel, every program processed. BLOCK_SIZE An element"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output
```

### 2. Softmax
**operator type**:Reduce
**Key points**:
- Numeric stabilization (minus)
- Line-by-line, grid mode loop
- `tl.range` with `tl.num_programs`

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # Numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        y = torch.empty_like(x)
        num_programs = min(32, n_rows)
        softmax_kernel[(num_programs, 1, 1)](
            y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE
        )
        return y
```

### 3. Layer Norm
**operator type**: Reduce + Element-wise
**Key points**:
- Multiple scans (average → variance → unified)
- midpoint 32
- Save statistics for backpropagation

```python
import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    X, Y, W, B, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # Round one: Calculate the average
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # Second time: calculate the difference
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Third: Normalization
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, normalized_shape, weight, bias, eps=1e-5):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_SIZE = 1024
        layer_norm_kernel[(M,)](
            x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return y
```

### 4. MatMul (matrix multiplication)
**operator type**: MatMul
**Key points**:
- Use `tl.dot` for matrix multiplication
- 2D index calculation
- Block_ptr Simplify Access

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

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

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        )
        return c
```

### 5. Double Kernel (double core call)
**operator type**: Multi Kernel group
**Key points**:
- Call multiple Kernels in one forward
- Intermediate results passed by tensor
- Configure grid and block independently for each kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def first_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    result = tl.maximum(data, 0.0)  # ReLU
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def second_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    result = data * data  # Square
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # First kernel: ReLU
        intermediate = torch.empty_like(x)
        first_kernel[grid](x, intermediate, n_elements, BLOCK_SIZE)

        # Second Kernel: Square
        output = torch.empty_like(x)
        second_kernel[grid](intermediate, output, n_elements, BLOCK_SIZE)

        return output
```

## Universal Mode

All examples follow the same structure:

### Kernel definition
```python
@triton.jit
def kernel_name(
    output_ptr, input_ptr,   # Input/Output Pointer
    M, N, K,                  # shapeParameters
    BLOCK_SIZE: tl.constexpr, # Compiler constant
):
    pid = tl.program_id(0)
    offsets = ...
    mask = ...
    data = tl.load(...)
    result = compute(data)
    tl.store(...)
```

### ModelNew Class
```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        M, N = inputs[0].shape
        output = torch.empty_like(inputs[0])
        grid = (triton.cdiv(M, BLOCK_SIZE),)
        kernel_name[grid](output, inputs[0], M, N, BLOCK_SIZE=1024)
        return output
```

## Key note

### 1. tensor device and data type.
```python
# Make sure that the output tensor is the same as the input device
output = torch.empty_like(input_tensor)  # Recommendations
# or
output = torch.empty(shape, dtype=input_tensor.dtype, device=input_tensor.device)
```

### 2. Grid Configuration
```python
# Simplicity: Direct calculation
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

# 2D Situation
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

# Aitutune case: use llambda
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)

# Number of restricted procedures (line-by-line)
grid = (min(n_rows, 32),)
```

### 3. ModelNew Format Requirements
- **Must**inherit `torch.nn.Module`
- **Must**achieve the `forward` method
- Output tensor with `torch.empty_like` or `torch.empty`

### 4. Parameter Transfer
```python
# Correct: all parameters passed as position parameters, keyword used
kernel[grid](output, input, M, N, BLOCK_SIZE=1024)

# Error: Non-constexpr arguments use keywords
kernel[grid](output=output, input=input)
```

## Validate correctness
```python
# Compare to PyTorch Native
x = torch.randn(128, 256, device='cuda', dtype=torch.float16)
output_triton = model_new(x)
output_torch = torch.nn.functional.softmax(x, dim=-1)

# Check discrepancies
diff = (output_triton - output_torch).abs().max()
print(f"Max difference: {diff.item()}")
assert diff < 1e-3, "Results mismatch!"
```
