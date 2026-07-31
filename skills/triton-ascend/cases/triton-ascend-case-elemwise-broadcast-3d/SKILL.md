---
name: triton-ascend-case-elemwise-broadcast-3d
description: "Cross-axis 3D broadcasting optimization (last dimension small): vector efficiency is enhanced by a two-stage kernel strategy (starting with +reshape with 2D and then standard multi-nuclear processing), applied to cross-axis Broadcast and the last dimensionalally small (<20) resulting in a low impact of vector"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Cross-axis 3D Broadcast Optimization Case

## Task characteristics
- **Operating type**: cross-axis Broadcast, Broadcast First and Three Axes
- **Data size**(65536, 128, 16) / (1, 128, 1)
- **Characteristics**: first dimension large, third dimension small, cross-axis Broadcast

## Optimization: Two Stages Kernel Policy

When the last dimensional special hour (e.g. W = 16) in cross-axis Broadcast, if directly processed, results in a differential effect of vector.

### Phase 1: Broadcast Kernel (Multinuclear Parallel)

```python
# Forward (1, H, 1) Broadcast to (1, H, W)
input2_broadcast = torch.empty(1, H, W, dtype=input2.dtype, device=input2.device)

grid_broadcast = lambda meta: (meta['NUM_H_CORES'],)
broadcast_kernel_parallel[grid_broadcast](
    input2, input2_broadcast,
    input2.stride(1),
    input2_broadcast.stride(1), input2_broadcast.stride(2),
    H=H, W=W,
)
```

### Phase 2: Division Kernel (Reshape 2D)

```python
# Convert 3D questions to 2D
input1_flat = input1.reshape(B, HW).contiguous()  # (B, HxW)
input2_flat = input2_broadcast.reshape(1, HW).contiguous()  # (1, HxW)
output_flat = torch.empty(B, HW, dtype=input1.dtype, device=input1.device)

grid_div = lambda meta: (meta['NUM_CORES'],)
div_flatten_kernel[grid_div](
    input1_flat, input2_flat, output_flat,
    B, HW,
    input1_flat.stride(0), input1_flat.stride(1),
    input2_flat.stride(1),
    output_flat.stride(0), output_flat.stride(1),
)
```

### Optimizing content
1. **Phase I kernel**: the dimensions of broadcast will be needed first to map along the HD to multi-nuclear parallel processes
2. **Phase II kernel**: 2D for 3D reshape, 1D (B) map to polynucleus, nucleus to HW dimension (SUB_HW=512), and vector ' s dimension is significantly increased

This approach, by pre-broadcast+reshape, avoids the last small dimension of the vector efficiency problem.

## Generic optimization programme

### Continuous broadcast (near dimension)
By reshape the adjacent dimensions are combined into one dimension and converted to a single axle Broadcast.

### Cross-axis Broadcast
- First dimension map to multi-nuclear parallels.
- The other dimensions inside the core are divided according to needs

### Summary
When the last dimensional special hour in cross-axis Broadcast, two phases of kernel: start with +reshape as 2D, then perform standard multi-nuclear parallel processing to enhance vector efficiency.
