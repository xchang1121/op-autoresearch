---
name: catlass-hardware-constraints
description: "CATLASS TileShape and on-chip cache capacity constraints: L1/L0A/L0B/L0C budget formula, fp16/fp32 Pingpong double buffering, 512B alignment and layout effects on Tile selection."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc_catlass
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "matmul, conv"
---

# CATLASS Hardware Containment and Tile Selection

Before changing `L1TileShape` / `L0TileShape`, whether or not the buffer zone is beyond the limit; most of the failure to compile `static_assert` is a problem.

## Cache on film (Atlas A2 Quantities, Pingpong STAGES=2)

| Buffer | Capacity | Purpose |
|------|------|------|
| L1 | 512 KB | A/B file double buffering |
| L0A | 64 KB | A file double buffering |
| L0B | 64 KB | B file double buffering |
| L0C | 128 KB | C-cumulators (fp32 bytes/elements when added) |

Atlas A5-level inter-generational capacity may differ, based on current `ArchTag` and warehouse files;**do not**hard-set the A2 formula to uncertified architecture.

## fp16 + Pingpong (Elemental Budget)

Note `L1 = (m1, n1, k1)`, `L0 = (m0, n0, k0)`, element type width `es` (fp16 is 2), excise `ac` (fp32 is 4), double buffering times**2:

```
L1 Volume of Elements:  m1*k1 + n1*k1          (multiplier) es*2 Byte required ≤ 512KB)
L0A Volume of Elements: m0*k0                  (× es*2 ≤ 64KB)
L0B Volume of Elements: k0*n0                  (× es*2 ≤ 64KB)
L0C Volume of Elements: m0*n0                  (× ac ≤ 128KB,fp32 (plus)
```

**Example (described algorithm)**: `L1=(128,256,256)`, `L0=(128,256,64)`, fp16, Pingpong
→ L1 saves `128*256*4 + 256*256*4` by word, and is required to certify, on a case-by-case basis, that it is below 512KB.

## fp32 When entering

L1 A/B is often 4 bytes; `L0C` may still be added by fp32. fp32 scenes tend to lower `k1` or `m1/n1`, otherwise L1 will be reached first.

## Selective principle (not relevant to specific benchmark number)

1. **M/N/K alignment**: multiple of priority 16; RowMajor often requests**512B alignment**(about 256 elements under fp16), otherwise the padding kernel approach should be taken instead of wriggling Tile.
2. **L0 relation to L1**: Regular `m0=m1`, `n0=n1`, `k0≤k1`; `k0 = k1/4` is the common starting point, not the only solution.
3. **Full capacity before performance**: L1 is recalculated before any increase in `k1`; `k1` is smaller than problem K, the frequency of the outer K cycle increases.
4. **Fragmentation**: RowMajor/ ColumnMajor is a different combination and priority should be given to ensuring removal efficiency**256 times L1**; zN etc. is less sensitive to alignment 256.
5. **Small M or very small K**: too large Tile → base blocks are too small and underutilized; should**reduce m1/n1 or adjust k1**and accompany Swizzle instead of replacing example names.

## 512B Alignment (RowMajor)

- Focus Matrix**Inner Axis**(e. g. RowMajor column orientation) Whether or not 512B alignment
- Unarranged: expand size in padding or opt for example families with padding at the pageline stage, instead of changing Tile on the error Kernel

## Fragments and recommendations Tile shape (start point, re-check capacity and load)

| A | B | L1 Start (Accountable) | Annotations |
|---|---|------------------|------|
| RowMajor | RowMajor | (128, 256, 256) | Universal |
| RowMajor | ColumnMajor | (128, 256, 256) | N Continuous removal friendly |
| ColumnMajor | ColumnMajor | (256, 128, 256) | M Directional Priority |
