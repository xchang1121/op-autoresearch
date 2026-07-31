---
name: catlass-matmul-optimization
description: "CATLASS Gemm Performance Modified: DispatchPolicy, Tile, Balance with subnucle load, Swizzle, when to use padding/Split-K/Preload. For AR to modify the type aliases in catlass_kernel.asc."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc_catlass
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "matmul"
---

# CATLASS Matrix Multiplication

in ARMedium Priority**`catlass_op/kernel/catlass_kernel.asc`** in `DispatchPolicy`,`L1TileShape`,`L0TileShape`,`BlockScheduler`(Swizzle) must be able to be compiled, andverify accuracyand profileIndicators are only meaningful.

## 1. First select the Kernel family (logical condition, not backnumber)

| Conditions | Direction |
|------|------|
| General alignment Gemm, no complex tailings | `BasicMatmul` + Pingpong |
| Requires D=A@B+X, Element-by-Element Integration, etc. | `MatmulEpilogue` or EVG path (see epilogue skill) |
| Inner axis not aligned 512B | Padding family Kernel, do not just screw Tile on the wrong template |
| K Large, Single-Zone K Dissatisfied and**Do enable multiple K returns** | Spit-K family; if configuration results in**spit factor 1**, equal to generic Gemm, no gain for exchange |
| Magnificent M/N, bandwidth Bottleneck | Preload + shuffleK |
| M or N very small and much smaller than the nucleus | **m1/ n1**to reduce L1 by increasing the number of blocks and adjusting Swizzle |

Pipeline binds example with M/N/K/layout/tail treatment**of**Reference against the above table instead of using the benchmark question.

## 2. DispatchPolicy

```cpp
// baseline
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

// Large matrix, lower lift bubbles.
using DispatchPolicy = Gemm::MmadAtlasA2Preload<true, true>;
```

Note: Preload will introduce extra running water with scalar/vector expenses,**small matrices may slow down**; based on profile.

## 3. Tile equals the AIC load

Number of basic blocks:

```
blocks = ceilDiv(M, m1) * ceilDiv(N, n1)
```

Target: `blocks` is close to the integer multiple of**AIC**, avoiding a small number of nuclears bearing the majority of the pieces.

Recommendations on the order of reference:

1. Ensure with `catlass-hardware-constraints` that L1/L0 does not spill
2. Adjust `m1`, `n1` (maintain `m0,m1` relationship with `n0,n1`) to the extent feasible
3. If `blocks` is still much smaller than the nucleus,**decrease**`m1` or `n1`, don't just increase Tile
4. And fine-tune Swizzle 's `offset`/ `direction`

## 4. Swizzle

`GemmIdentityBlockSwizzle<offset, direction>`:

| Situation | Starting point |
|------|------|
| M > N, A/B is RowMajor | direction = 0 |
| M < N, A/B for RowMajor | direction = 1 |

Once the direction is set, `offset` (e.g. 3 → 4→5) can observe the profile; each time only one match is easy to settle.

## 5. Performance associated with the PyTorch path

`latency_us` for AR is typically**`ModelNew.forward`end to end**(including dtype conversion, `npu_format_cast`, operator call). If profile shows time spent in Transdata instead of MMAD:

- Change only `.asc` Tile**Possible indicator of almost static**
- `catlass_op/src/catlass_torch.cpp` needs to be changed to reduce duplicate `npu_format_cast` or redundant copy (this path should be listed in `editable_files`)

## 6. Common error zone

- **`kernel.py` Symbol**or Python Logic without Tile →'s performance usually remains unchanged
- **Split-K is not a good name**: K is short or split=1 and applies `BasicMatmul`
- **Tile excessive**: blocks less than core → mass nuclear idle
- **Ignore alignment**: → accuracy or performance abnormal
- **In excess of L1**: translation failed or silently misconfigured - calculate capacity before submitting eval
