---
name: triton-ascend-elementwise
description: "For Pure Element-by-Element(element-wise)Categoryoperator. WhenoperatorThe core calculation is right.tensorThis guide should be selected for each element to perform the same operation independently and without inter-element dependence, typicallyoperatorIncluding:relu, sigmoid, tanh, gelu, selu, leaky_relu, elu, swish, softplus, hardsigmoid, hardtanh, softsign, exp, log, sqrt, pow, add, mul, sub, div, abs, neg, clamp, cast(Type Conversion), where, fill, copyAnd so on. It's also relevant.scalarRadio(broadcast). This does not apply to cross-elements.(Likesum/mean/max)ormatrix multiplicationofoperator. IfoperatorComprises both element-by-component calculation and global attribution (e.g. loss function)MSELoss,HuberLoss,HingeLoss), should chooseelementwise-reduce-fusedGuide."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise"
---

# Guidance for the preparation of Element-wise operator

## Preparation Mode

Core features of Element-wise operator: Each output element relies only on input elements at the corresponding location and is non-intersectional.
Common writing is the display of tensor as 1D, with a staggered cycle, press block through all elements.

### Standardized

```python
@triton.jit
def elementwise_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, CORE_NUM):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        y = compute(x)  # Replace with Specific Calculations
        tl.store(output_ptr + offsets, y, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch
            import triton
            device = torch.npu.current_device()
            properties = triton.runtime.driver.active.utils.get_device_properties(device)
            self.VEC_CORE_NUM = properties.get("num_vectorcore", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        grid = (self.VEC_CORE_NUM,)
        elementwise_kernel[grid](x, y, n, BLOCK_SIZE=1024, CORE_NUM=self.VEC_CORE_NUM)
        return y
```

**Element**
- `.contiguous()` ensures continuous access to a one-dimensional pointer and avoids stide calculation
- `torch.empty_like` creation output (no zeros, save initial cost)
- The parameter signature and quantity of `forward` must match the original `Model.forward`

## Optimizing skills

### 1. Continuous memory access

A 1-dimensional, continuous offset access, with the highest rate of cache hits:
- tensor First `.contiguous()`
- Acquiring the total number of elements with `x.numel()`, ignoring the original shape

### 2. BLONK_SIZE Selection

- Recommended 1024-2048, Balancing flow efficiency and UB occupancy
- The amount of data can be reduced to 256-512 in a very long time.
- BLONK_SIZE does not need to be increased when the amount of data is large. The stagger cycle is automatically balanced.

### 3. Numerical stability

- Maximum excretion pre-exceeding `exp`
- Ensure non-negative before `sqrt`: `tl.maximum(x, 0.0)` or `tl.maximum(x, eps)`
- Intermediate calculation with float32 cum and final return to target accuracy

### 4. Combining multistep calculations

Continuous elementwise operations should be integrated into the same kernel to avoid multiple GM readings and writing:

```python
# I'm going to take a look at this.
y = tl.maximum(x, 0.0)  # relu
y = y * scale            # scale
y = y + bias             # add_bias
```

### 5. Broadcast processing

When an input is scalar or needs to be broadcast, load with constants in the Kernel external processing or in the Kernel:

```python
# scalar imported as kernel parameter
@triton.jit
def scale_kernel(x_ptr, out_ptr, scale_val, n, BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr):
    ...
    y = tl.load(x_ptr + offs, mask=mask, other=0.0) * scale_val
```
