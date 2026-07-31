---
name: triton-ascend-performance-improvement
description: |
  Triton Ascend performance optimizes operational experience. The generic optimisation model, which is derived from batch adaptation searches, covers tile optimization, memory loading optimization, rediction optimization, covert broadcasting, multi-pass consolidation, data access re-engineering, etc.
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
---

## Tile Size Selection Method

tile sizes are subject to hardware storage restrictions (hardware information documents for specific capacity reference):

**CUBE Path (matmul / tl.dot)**: Tyre must be capable of being placed in L0A/L0B/L0C
- Calculator: `BLONK_M × BLONK_K × size (dtype) ≤ L0A capacity '
- fp32 Two times more than fp16 and needs to be reduced accordingly
- K-dimensional 512B alignment to increase bandwidth utilization

**VEC Path (elementwise / reduce)**: All active tensor needs to be placed in UB
- Calculating formula: `BLONK_SIZE × sizeof(dtype) × tensor Active × multi_buffer coefficient ≤ UB capacity '
- compiler auto-multi-buffer will increase occupancy to 2 ~3 times
- Kernel intermediate variable (e. g. temporary buffer from `tl.where`) also occupies UB

**Optimal strategy**: Try from a larger file, downscaled in case of `ub overflow` / `cbuf overflow` 's compilation error. Complementing `@triton.autotune` 's Auto-Effort.

## Memory Load Optimization

Apply mask and fill values directly on `tl.load` instead of loading them unconditionally and filtering them with `tl.where`:

```python
# Suboptimal: load first and then where (more than one middle operation, possibly trigger vsel compilation error)
tile = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
tile = tl.where(mask, tl.load(ptr + offsets), 0.0)

# Recommendations: Apply mask directly on load
tile = tl.load(ptr + offsets, mask=mask, other=0.0)
```

## Invisible radio instead of visible

There is a need to expand the dimension (`[:, None]`) of Triton's covert radio to avoid `tl.broadcast_to` creating temporary matrices when low-dimensional tensor is broadcast to high-dimensional:

```python
# Suboptimal: Evidently expanded to complete matrix
a_broadcast = tl.broadcast_to(a_tile[:, None], (BLOCK_M, BLOCK_N))
c_tile = a_broadcast * b_tile

# Recommendations: Invisible broadcasts
c_tile = a_tile[:, None] * b_tile
```

## Reduction best practice

1. **scalarComposer**: for each core`core_sum = 0.0`Local accumulation, avoid.tensorIndexing issues
2. **single atom writing**: `tl.atomic_add(out_ptr, core_sum)` writing once after cycle
3. **Avoid host end permute**: non-last dimension reduce direct multidimensional index in Kernel

## Host-end projection

Precalculating the tensor's stride on the host end and entering it as a parameter in Kernel instead of inside Kernel, reduces referencing errors and supports compiler optimization:

```python
# hostend
stride_am, stride_ak = A.stride(0), A.stride(1)
kernel[grid](A_ptr, ..., stride_am, stride_ak, ...)
```

## Multiple Pass Merge to Single Pass

  When  operatorMultiple separate periods of time for the same datasoftmax of max→exp_sum→normalize,or topk) should be combined into a single cycle.HBMRead, the ratio is theoretically higher.passMerges to a lesser degree of degree of attribution (can be placed in a single dimension)BLOCKThe scene is the best.

## Strided Access Mode Reconstructing

When operator is involved in non-continuous moded access (e.g., access to matching calculations of adjacent elements), the data loading policy should be reset:
- Change to semantic grouping to place related elements in the same block
- Avoiding random access to `flat_idx % D` + across stride
- Handle pairing relationships within blocks after successive loads, using data locality

Ascend hardware has significant performance penalties for non-continuous access, and the post-restructuring performance gap can be several times to several dozen times.

## Two Stages Better than Single Pass

For the LayerNorthm/RMS Norm category operator, the two-stage programme (Statistics in the first instance means/var, second-time integration) is usually better than single pass (online statistics).
Reason: Increased tensor in the single pass cycle, increased UB pressure, reduced optimisation efficiency of compiler pipeline. In fact, 2-pass is 20% faster than single pass.
