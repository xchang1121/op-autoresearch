# Universal Tiling Reference

> Tiling Design Principles, tmpBufSize formulae, common Tiling data structure fields

---

## Tiling Design Principles

**For direct formula calculations, no two-point search**

1. **a0TileBase is the smallest alignment unit**: `VECTOR_REG_WIDTH / sizeof(T)` (FP32=64), all Buffer sizes are several times their integer
2. **Limitation to minimize**: `a0Inner = Min (UB capacity limit, A0 dimension limit, polynuclear balance limit) '
3. **Conservative estimate**: `ubPerTileBase` is calculated using a0TileBase, the actual `tileA0Len ≤ estimate ' will not exceed UB
4. **API Parameters Limit Transmission**: `R_max = min(R_max, 255)` if `repeatTimes ≤ 255` of Reduce API is relevant to R

**Full load of vs split decision**: Full load = Load data + All Buffer ≤ UB_SIZE calculation processes. Unlike the middle Buffer of operator, the threshold formula differs from operator.

## tmpBufSize (sharedTmpBuffer) calculations

```cpp
uint32_t ComputeReduceBufSize(uint32_t rLengthAlign, uint32_t typeSize) {
    uint32_t perRepeat = 256 / typeSize;  // 64 for FP32
    uint32_t perBlock = 32 / typeSize;    // 8 for FP32
    uint32_t repeats = (rLengthAlign + perRepeat - 1) / perRepeat;
    uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * typeSize;
    return std::max(tmpBufSize, 4096u);   // Min 4KB
}
```

---

## Common Tiling Data Structure Fields

## Basic attribution Tiling (ReduceOpTilingData)

| Fields | Type | Meaning |
|------|------|------|
| `factorACntPerCore` | uint64 | Workload per A-axis |
| `factorATotalCnt` | uint64 | A-axis total working module |
| `ubFactorA` | uint64 | A-axis slice size for UB |
| `factorRCntPerCore` | uint64 | Workload per core R axis |
| `factorRTotalCnt` | uint64 | R-axis total working module |
| `ubFactorR` | uint64 | UB 's R-axis slice size |
| `groupR` | uint64 | R-axis grouping (>1 trigger Group Reduce)|
| `outSize` | uint64 | Output Buffer Size |
| `basicBlock` | uint64 | Enter UB buffer size |
| `resultBlock` | uint64 | Output/intermediate buffer size |
| `coreNum` | int32 | Use nuclei |
| `useNddma` | int32 | Whether to use NDDMA |
| `shape[8]` | uint64[] | Dimensions |
| `stride[8]` | int64[] | Step in every dimension |

## Arg Max Series Tiling

| Fields | Type | Meaning |
|------|------|------|
| `aSize` | uint64 | Volume of all dimensions before the axis of engagement |
| `rSize` | uint64 | Reduction-axis volume |
| `nextASize` | uint64 | Volume of all dimensions after a axis of engagement |
| `cutASize` | uint16 | A slice of UB |
| `cutRSize` | uint16 | R slices of UB |
| `cutNextASize` | uint16 | UB's nextA slice |
| `realCoreNum` | uint64 | Actual use of cores |
| `blkFactor` | uint64 | Size of blocks per nuclear main dimension |
| `blkTailFactor` | uint64 | The size of the tail core main dimension |
| `tilingKey` | uint64 | Policy Selection Key |
| `aRaMode` | uint64 | ARA Sub-Model (1-6) |
| `workSpaceSize` | uint64 | Group Reduce workspace |

## Norm Class Tiling (RmsNom/LayerNom)

| Fields | Type | Meaning |
|------|------|------|
| `num_row` | uint64 | & Enter Number of Lines |
| `num_col` | uint64 | Enter the number of columns |
| `num_col_align` | uint64 | Align the rear rows |
| `block_factor` | uint64 | Number per line |
| `row_factor` | uint32 | Lines per iterative processing |
| `ub_factor` | uint32 | Number of columns processed on an iterative basis |
| `reduce_mask` | uint32 | Return Mask Configuration |
| `epsilon` | float | Value stabilization constant |
| `avg_factor` | float | 1.0/num_col |

## Softmax Series Tiling

| Fields | Type | Meaning |
|------|------|------|
| `a` (or `totalA0Len`/`totalA1Len`) | uint64 | A-dimensional |
| `r` (or `totalRLen`) | uint64 | R-dimensional |
| `rAligned` | uint64 | R Align to Size |
| `ubFactor` | uint64 | UB Process Size |
| `aBlockFactor` | uint64 | Line A per core |
| `tilesPerCore` | uint64 | Number of files per core |
| `rLoopCount` | uint64 | R / VL_FP32 |
