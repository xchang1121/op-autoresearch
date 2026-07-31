---
name: triton-ascend-optimization
description: "Triton Ascend Performance Optimizing Universal Policy: BLOCK_SIZE Selection (1024-2048 for elementwise, must be < 65536), grid conversion (use VEC_CORE_NUM / CUBE_CORE_NUM, 2D/3D grid for matmul / conv / reduce, 1D grid + inner loop for election / pointwise), 256Bignment for memory transporters, autotune brook-size papers, fp16/fp32 precision. Bind via keywords like matmul, elementwise, reduce, block_size, grid, autotune, alignment, fp16, fp32, tile, interleaved-loop, cube-core, vec-core."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Triton Ascend Performance Optimization Guide

## Optimizing Policy Checklist

- [ ] **Grid 1D**: `grid=(CORE_NUM,)` + kernel stagger cycle `for block_id in range(pid, total, CORE_NUM)`
- [ ] **Grid dimension selection**:
  - Consider 2D/3D grid for calculating the intensive operator (matrix multiplier, volume, volume etc.) using hardware movement advantages
  - For a large number of small-scale calculations (element-wise, pointwise, etc.), consider 1D grid + nuclear inner cycle to reduce start-up costs
- [ ] **nuclear kernel cycle**: Add extra cycles to scenes not required for compiler automatic multi-level flow of water
- [ ] **Try not to be BLONK_SIZE**: `ub overflow` shrinks from the larger tile; balance parallelity and resource occupancy in the nuclear inner cycle:
  - Try a small tectonic strategy so that reading and writing can go hand in hand.
  - Try a big split policy to increase UB usage
  - Column Multigroup Configuration, Add @triton.autotune
- [ ] **operator split**: Complex integration operator can be split into multiple Kernel sequences, sometimes better performance
- [ ] **Autotune**: column multigroup tile parameter configuration (excluding num_warps/num_stages)
- [ ] **Reduction with scalar cumulation**: each core scalar cumulation + single atomic writing
- [ ] **Memory alignment**: matmul up bandwidth by 512B
- [ ] **Avoid host end permute**: non-last dimension index processing in Kernel
- [ ] **Invisible broadcasts**: replace `tl.broadcast_to` with `a[:, None] * b` and reduce temporary tensor
- [ ] **load directly Mask**: `tl.load(ptr, mask=m, other=0.0)` better than load first and `tl.where`
- [ ] **Reduction of redundancy accuracy conversion**: avoid repeated conversions at fp16/fp32, i.e. `.to(float16)` and `.to(float32)`, multiple reuses at a time
- [ ] **Core number configuration**: grid number set to core number (VEC/CUBE) with start-up cost inverse when oversized
- [ ] **256B alignment**: Data removal in 256B to enhance bandwidth

## Reduction Optimization

Each core begins with a local scalar cumulation, with the final atom written:

```python
core_sum = 0.0
for block_start in range(pid, total_blocks, CORE_NUM):
    data = tl.load(...)
    core_sum += tl.sum(data, axis=0)
tl.atomic_add(output_ptr, core_sum)
```

## Numerical stability

### Spillproof.
```python
max_val = tl.max(scores, axis=0)
scores = scores - max_val
p = tl.math.exp2(scores)
```

### At the beginning of the defense value
- Ensure non-negative before any sqrt: `max(input, 0.)` or `max(input, eps)`

### accuracy Upgrade
- matmul with fp32 loader: `acc = tl.zeros([M, N], dtype=tl.float32)`
- And then we go back to target accuracy: `result = acc.to(tl.float16)`.
