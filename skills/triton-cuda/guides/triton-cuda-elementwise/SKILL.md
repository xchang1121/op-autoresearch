---
name: triton-cuda-elementwise
description: "Element-by-Elemental operator (element-wise) Optimization policy, including the vector Implementation and Integration technique for add/mul/relu/sigmoid/tanh/gelu/exp/log and so on. The CUDA internal nuclear code generation scenario for achieving the vector mode operator for active functions, element-by-element calculations, broadcast operations, etc."
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "elementwise"
  algorithms: "add, mul, relu, sigmoid, tanh, gelu, exp, log, div, sub, sqrt, pow"
---

# Element-wise operator Optimization

> Applies to operator on an element-by-element basis

## Apply operator

**Add, Mul, div, sub, pow
**Activation function**: relu, sigmoid, taunh (for `tl.extra.cuda.libdevice.tanh`), gelu, silu, swish
**Mathematical functions**: exp, log, sqrt, sin, cos,abs

## Optimizing Policy

### 1. Continuous memory access optimization

When tensor is stored continuously in the memory, a one-dimensional pointer can be used to walk through the memory to avoid multi-dimensional indexing costs.

**Programme 1: Continuous + 1-D visits (recommended)**

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

        elementwise_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
        return output_tensor

@triton.jit
def elementwise_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask)
    result = compute(data)  # Your calculation logic.
    tl.store(output_ptr + offsets, result, mask=mask)
```

**Strength**:
- `.contiguous()` One-time expense vs distance every visit
- Better combined access
- It's easier to optimize compiler

**Program 2: Using Stride Access (not recommended)**

Use only when `.contiguous()` cannot be called.

### 2. BLONK_SIZE Selection

- **Recommended value**: 256, 512, 1024
- **Principle**: balancing parallelity and resource occupancy
- **GPU Consider**
  - Bigger BLONK_SIZE → with fewer block start costs, but possibly lower
  - Smaller BLONK_SIZE → parallels more finer particle size, but start costs increase
  - Make sure Grid is large enough to make full use of GPU

### 3. Warp Configuration

Element-wise operator usually uses less warp:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
    restore_value=['output_ptr'],  # Must: list all output pointer parameter names
)
@triton.jit
def optimized_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask)
    result = compute(data)
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 4. Large Shape Process

Ensure that there are enough blocks to cover all elements when the size of the Shape is entered:

```python
@triton.jit
def large_elementwise_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # Each program handles multiple blocks (grid mode loop)
    for block_start in range(pid * BLOCK_SIZE, n_elements, num_pids * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)

# Start: Limit Grid size
num_blocks = min(triton.cdiv(n_elements, BLOCK_SIZE), 65535)
grid = (num_blocks,)
large_elementwise_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=1024)
```

### 5. vector load

For simple element-wise operator, a larger BLONK_SIZE can increase the volume of work per thread and increase the density of calculation:

```python
@triton.jit
def vectorized_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Larger BLONK_SIZE allows compiler to make a better vector
    data = tl.load(input_ptr + offsets, mask=mask)
    result = tl.maximum(data, 0.0)  # ReLU
    tl.store(output_ptr + offsets, result, mask=mask)
```

## Full example: ReLU

```python
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask)
    result = tl.maximum(data, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()

        output = torch.empty_like(x)
        n_elements = x.numel()

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        relu_kernel[grid](x, output, n_elements, BLOCK_SIZE)
        return output
```

## Full example: GELU

```python
import torch
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x_cubed)  # sqrt(2/pi) ≈ 0.7978845608
    result = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))

    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output
```

## Performance Checklist

- [ ] Is the input converted to a continuous memory?
- [ ] BLONK_SIZE is a 2-year-old?
- [ ] Did you use autotune to search for optimal configuration?
- [ ] For the big Shape, did you use the grid mode loop?
- [ ] Do memory access merge (codesced)?

## Common Errors

1. **Forgot to continue**: leading to non-consolidated visits and reduced performance
2. **BLONK_SIZE Too small**: start-up costs too high
3. **BLONK_SIZE Too big**: ocupancy lower
4. **Forget Mask**: Cross-border visits lead to mistakes
5. **Unnecessary Synchronization**: element-wise operator does not require Synchronization
