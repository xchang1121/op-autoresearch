---
name: triton-ascend-examples-mindspore
description: "An integrated example of the Triton Ascend kernel under MindSpore framework displays standard writings such as MindSpore customises operator registration, Primitive definition, and tensor uploads. When the target framework is mindspore, the example should be imported as a code structure reference."
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  framework: mindspore
---

# MindSpore + Triton Ascend Example Code

## MindSpore vs PyTorch Difference

| Features | PyTorch | MindSpore |
|------|---------|-----------|
| **Base group** | `torch.nn.Module` | `mindspore.nn.Cell` |
| **Forward function** | `forward` | `construct` |
| **tensor was created** | `torch.empty` | `mindspore.mint.empty` / `mindspore.mint.empty_like` |
| **device** | `device='cuda'/'npu'` | `mindspore.set_device("Ascend", 0)` / `mindspore.set_device("CPU")` |
| **data type** | `torch.float16` | `mindspore.float16` |
| **Get core** | `triton.runtime.driver.active.utils.get_device_properties` | `mindspore.runtime.get_device_limit(0)` |

## Example List

### 1. Victor Add (vector plus)
**MINDSpore Achieved**:
```python
import mindspore as ms
from mindspore import nn
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)

class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, a, b):
        c = ms.mint.empty_like(a)

        n_elements = a.size
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
        return c
```

### 2. MatMul (matrix multiplication)
**Key differences**:
```python
class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x0, x1):  # Note: Use construct Not forward
        B, C = x0.shape
        C2, D = x1.shape
        assert C == C2, f"Matrix dimensions do not match: {C} != {C2}"

        # Create MindSpore tensor
        output = ms.mint.empty((B, D), dtype=ms.float32)

        matmul_kernel[1, 1, 1](output, x0, x1, 1, B, C, D)
        return output
```

### 3. Layer Norm
**MindSpore special treatment**:
```python
class ModelNew(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape

        # MindSpore Parameter Initialization
        ms.set_seed(0)  # Note: Use ms.set_seed Not torch.manual_seed
        self.weight = ms.Parameter(ms.ops.ones(normalized_shape, ms.float32))
        self.bias = ms.Parameter(ms.ops.zeros(normalized_shape, ms.float32))

    def construct(self, x):
        M, N = x.shape
        output = ms.mint.empty_like(x)

        grid = (M,)
        layernorm_kernel[grid](
            x, output, self.weight, self.bias,
            N, self.eps, BLOCK_SIZE=triton.next_power_of_2(N)
        )
        return output
```

### 4. Softmax
```python
class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        n_rows, n_cols = x.shape
        output = ms.mint.empty_like(x)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)

        softmax_kernel[grid](x, output, n_cols, BLOCK_SIZE)
        return output
```

### 5. Double Kernel (double core call)
```python
class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        # First Kernel
        intermediate = ms.mint.empty_like(x)
        kernel1[grid](x, intermediate, ...)

        # Second Kernel
        output = ms.mint.empty_like(x)
        kernel2[grid](intermediate, output, ...)

        return output
```