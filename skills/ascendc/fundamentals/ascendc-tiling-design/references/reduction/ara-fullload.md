# Reduce Class operator - ARA Full Load Branch

> **Applicable scene**: A0>1 (non-tailaxis), R ≤ R_max (full load mode)

---

## Contents

- [I. Branch characteristics](#a branch feature)
- [II, Buffer Planning](#2 Buffer - Planning)
- [III, Tiling Parameter Calculation](#triling - Parameter Calculation)
- [IV. Kernel Achievement Point](#4kernel - Achievement Point)
- [V, common issue](#5 common issue)
- [Performanceoptimization recommendation](#Six.optimization recommendation)

---

## I. SPECIFIC STATE

| Features | Annotations |
|------|------|
| **Template type** | ARA template (Axes + R + A Axes) |
| **Shape Abstract** | (A1, R, A0) |
| **Load mode** | Full load (all R rows fully placed in UB) |
| **Conditions applicable** | A0>1, R ≤ R_max |
| **Data continuity** | Inconsistent R elements per column (Spacing A0) |
| **Reduce Results** | vector (A0 values) |

---

## II. Buffer Planning

### 2.1 FP32 scene (Kernel side)

```cpp
// Double Buffer Mode
constexpr uint32_t a0TileBase = 64;  // FP32

pipe->InitBuffer(inQueueX, 2, R * a0TileBase * sizeof(float));
pipe->InitBuffer(outQueueY, 2, a0TileBase * sizeof(float));  // ReduceOutcome (%)A0value)
pipe->InitBuffer(tmpBuf, tmpBufSize);

// UB = 2 × R ×0TileBase×4 + 2×0TileBase×4 +tmpBufSize
```

### 2.2 tmpBufSize Calculation (Host side)

```cpp
constexpr uint32_t VECTOR_REG_WIDTH = 256;
uint32_t perRepeat = VECTOR_REG_WIDTH / sizeof(float);  // 64
uint32_t perBlock = 32 / sizeof(float);  // 8

// tmpBufSize based on R × signed Cols
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);
uint32_t repeats = (R * alignedCols + perRepeat - 1) / perRepeat;
uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * sizeof(float);
tmpBufSize = std::max(tmpBufSize, 4096u);
```

---

## III. Tiling Parameter Calculation

### 3.1 Calculation of full load threshold (Host side)

```cpp
// Full load condition: 2 × R × 0 TileBase× 4 + 2 ×0TileBase× 4 + tmpBufSize ≤ UB_SIZE
constexpr uint32_t a0TileBase = 64;  // FP32
uint32_t tmpBufSize = 4096;
uint32_t overhead = 2 * a0TileBase * sizeof(float) + tmpBufSize;
uint32_t R_max = (UB_SIZE - overhead) / (2 * a0TileBase * sizeof(float));

R_max = std::min(R_max, 255u);  // API Limits repeatTimes ≤ 255
R_max = std::max(R_max, 1u);

// Estimation: The formula calculates about 375, but is limited by min (R_max, 255), actual R_max = 255
```

### 3.2 A0Inner calculation (minimised by three bounds)

```cpp
// Binding 1: UB Capacity Limit
uint64_t ubPerTileBase = 2 * R * a0TileBase * sizeof(float) + tmpBufSize;
uint64_t fixedOverhead = 2 * a0TileBase * sizeof(float) + tmpBufSize;
int64_t factorMax = (UB_SIZE - fixedOverhead) / ubPerTileBase;
if (factorMax < 1) factorMax = 1;

// Binding 2: A0-dimensional limits
int64_t a0FactorMax = (A0 + a0TileBase - 1) / a0TileBase;

// Binding 3: Multi-Nation Balance Limit
int64_t totalTilesMax = A1 * a0FactorMax;
int64_t a0InnerMax = (totalTilesMax + blockDim - 1) / blockDim;

// Take Minimum Value
int64_t a0Inner = std::min({a0InnerMax, factorMax, a0FactorMax});
a0Inner = std::max(a0Inner, 1L);

uint32_t tileA0Len = a0Inner * a0TileBase;
```

### 3.3 Polynuclear cut parameters (Host side)

```cpp
uint32_t a0Outer = (A0 + tileA0Len - 1) / tileA0Len;
uint32_t totalTiles = A1 * a0Outer;
uint32_t tilesPerCore = (totalTiles + blockDim - 1) / blockDim;
uint32_t usedCoreNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
uint32_t tailCoreTiles = totalTiles % tilesPerCore;
if (tailCoreTiles == 0 && totalTiles > 0) tailCoreTiles = tilesPerCore;
```

---

## IV. KERNEL IMPLEMENTS

### 4.1 Data flows

```
GM (A1, R, A0) → CopyIn → UB (R × tileA0Len)
    ↓
[ReduceMax/ReduceSum Pattern::Reduce::RA] → result (tileA0Len)
    ↓
UB (tileA0Len) → CopyOut → GM (A1, A0)
```

> **Core advantages of the full load model**: [R, TileA0Len] data are fully stored in UB.
> If operator needs multistep calculations (e.g. ReduceMax and ReduceSum), the intermediate result is directly re-referenced to UB data.
> CopyIn is only needed once and no re-loading of data from GM (multi-wheel scans are required only in split mode because only partial data are moved in each time).
> Note: `isReuseSource=false` (default value) for ReduceMax/ReduceSum does not destroy source data, and subsequent steps can be continued.

### 4.2 Core API Call

```cpp
// Number of rows after alignment (key!)
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// srcShape must use signedcols
uint32_t srcShape[] = {R, alignedCols};

// ReduceMax.
ReduceMax<float, Pattern::Reduce::RA>(resultLocal, xLocal, tmpLocal, srcShape, true);

// Or ReduceSum.
ReduceSum<float, Pattern::Reduce::RA>(resultLocal, xLocal, tmpLocal, srcShape, true);
```

**Key points**:
- Using `Pattern::Reduce::RA` - Acquire along the first dimension (R-dimensional)
- `srcShape[1]` must use `alignedCols` (32 bytes aligned)
- Output is vector (tileA0Len value)

**Why choose Patterson::Reduce::RA**:

The data for each nuclear processing component `(R, A0_inner)`, after multi-nucleocution, are placed in the UB as follows:
```
[row0All of it.A0_inner, row1All of it.A0_inner, ..., row{R-1}All of it.A0_inner]
```
is the 2D matrix for `(R, alignedCols)`.

Pattern::Reduce::RA
- **R**=Reduce dimension (under first dimension of the Convention)
- **A**=Allign dimension (retain second dimension)

For each `a0` position (0 to A0_inner-1) take the R-values and output A0_inner results.

**Note: The Level 2 API `Reduce<T>(dst, src, tmp, count)` cannot be used because it can only process continuous data and can only return to the Axis-1.

### 4.3 Data handling (key!)

**GM→UB used DataCopyPad**:

```cpp
// BlockCount=R Line, blockLen=tileA0Len*sizeof(float)
DataCopyExtParams copyParams;
copyParams.blockCount = R;
copyParams.blockLen = tileA0Len * sizeof(float);
copyParams.srcStride = (A0 - tileA0Len) * sizeof(float);  // Cross A0
copyParams.dstStride = 0;

DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
```

**UB→GM uses DataCopyPad**:

```cpp
DataCopyExtParams copyParams;
copyParams.blockCount = 1;
copyParams.blockLen = tileA0Len * sizeof(float);
copyParams.srcStride = 0;
copyParams.dstStride = 0;

DataCopyPad(yGm[offset], resultLocal, copyParams);
```

### 4.4 Non-matching

```cpp
// TileA0Len may not be 32 byte alignment
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// Buffer size with signedcols
pipe->InitBuffer(inQueueX, 2, R * alignedCols * sizeof(float));

// Reduce API using signedcols
uint32_t srcShape[] = {R, alignedCols};
```

---

## V. common issue

| Problem | Reason | Solutions |
|-----|------|---------|
| accuracy is too big. | srcShape uses A0 instead of signedcols | Use the number of columns after alignment |
| Compiler error: no watching ReduceMax | API arguments do not match | Use interface with Pattern, srcShape={R, alignedCols} |
| Data Error | DataCopy srcStride/ dstStride Error | correctly calculate the distance across |
| FP16 accuracy | Moderate calculation accuracy is insufficient | Calculate using FP32 middle |
| UB Inadequate capacity | Buffer, it's not rational to plan. | Calculate UB usage correctly |

---

## VI. PERFORMANCEoptimization recommendation

1. **Double Buffer**: Open with `InitBuffer(que, 2, size)`, CopyIn /Compute/CopyOut
2. **FP16 Mixed accuracy**: Sum/Mean/Prod etc. under contract scenario: BF16/FP16 input, FP32 calculation, BF16/FP16 output
3. **A0Inner Optimization**: three bounds minimized to ensure load balance
