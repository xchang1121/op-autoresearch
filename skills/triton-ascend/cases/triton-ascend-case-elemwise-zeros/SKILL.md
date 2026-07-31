---
name: triton-ascend-case-elemwise-zeros
description: "SmallshapetensorCreatezeros/arange/fullOptimization: Avoiding multi-nuclear start-up and dispatch costs by reducing the number of nuclears, single-nuclear processing performance is better than multi-nuclear parallels, applicableshapeSmall (thousands of elements)elementwiseCreatetensorscene"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Zeros Create tensor Optimization Case

## Task characteristics
- **Operating type**: Elemwise type, including the operation to create tensors for torch arange, full, Zeros, Zeros_like
- **Data size**: (2, 256, 16), data size smaller
- **data type**: float32
- **Task characteristics**: can be sequenced according to axle (which can flatten is an axle), outer parallel, inner layer vector

## Optimization: Small Shape nuclei processing

```python
# kernel code
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
tl.store(output_ptr + offsets, zeros, mask=mask)
```

### Optimizing content
- Resize parallels and improve performance by setting the size of BLONK_SIZE
- When Shape is small, the nuclear numbers are kept to a minimum and multi-nuclear start-up and dispatch costs are avoided.

### Summary
1. On the Ascend platform, when the size of the Shape is small, the number is kept to a minimum, which avoids multi-nuclear start-up and dispatch costs and achieves performance optimization
2. For a simple Elementwise operation, the elements of the multiple axes are expanded into a single axis, and they are split on this axis.
3. Allocating block to each liner, and considering multiple splits if the UB does not exist
