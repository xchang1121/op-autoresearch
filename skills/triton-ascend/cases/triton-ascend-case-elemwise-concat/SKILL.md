---
name: triton-ascend-case-elemwise-concat
description: "Slice + Concat integration operator optimization: avoid intermediate results storage and multiple memory access through precision slice loading (only part of load required) and index calculation ciphering (avoiding cat commands) for multi-entry integration operator scenes requiring after slice integration"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Slice + Concat Integration operator Optimization Cases

## Task characteristics
- **Operating type**: integration of operator, 6 slice + 1 concat in a kernel
- **Data dimensions**: 7 input sizes (128, 50, 128), slices [128, 32, 48, 48, 48, 48, 48] after W dimensions, output (128, 50, 400)
- **Task characteristics**: operator integration to avoid storage of intermediate results and multiple memory access

## Optimization 1: Accurate Slice Loading

```python
# Only the slices that the load needs, not the whole input
# Input 1: Only 128 elements in front of load
w_offs_1 = tl.arange(0, SLICE_1)  # SLICE_1=128
input_offs = base_in_offs + w_offs_1[None, None, :] * stride_in_w
data = tl.load(x1_ptr + input_offs, mask=mask_1, other=0.0)

# Input 2: Only the top 32 elements of load
w_offs_2 = tl.arange(0, SLICE_2)  # SLICE_2=32
input_offs = base_in_offs + w_offs_2[None, None, :] * stride_in_w
data = tl.load(x2_ptr + input_offs, mask=mask_2, other=0.0)
```

### Optimizing content
- Only part of the slice required for each input within Kernel (e.g. 128, 32, 48)
- Control precisely the number of elements of the load through `w_offs = tl.arange(0, SLICE_SIZE)`
- Reduce unnecessary memory access and increase memory utilization bandwidth

## Optimize 2: Index calculation achieves fusion

```python
# Collapse by adjusting the output index instead of using the triton's cat command
w_out_offset = 0

# Input 1 writing position: output [0:128]
output_offs = base_out_offs + (w_out_offset + w_offs_1)[None, None, :] * stride_out_w
tl.store(output_ptr + output_offs, data, mask=mask_1)
w_out_offset += SLICE_1  # Update As128

# Input 2 writing position: output [128:160]
output_offs = base_out_offs + (w_out_offset + w_offs_2)[None, None, :] * stride_out_w
tl.store(output_ptr + output_offs, data, mask=mask_2)
w_out_offset += SLICE_2  # Update As160
```

### Optimizing content
- Write different input data to different locations of the output by maintaining output offsets (w_out_offset) and dynamically adjusting the output address index
- Avoid using Triton's cat command to calculate the spell directly from the address
- Reduction of intermediate steps and additional data removal costs

### Summary
1. For concat operations, the slices required for load should be precise within the kernel to avoid full load data
2. Collapse data directly to target position by indexing without additional cat instruction
3. operator integration avoids storage and multiple memory access of intermediate results and enhances overall performance
