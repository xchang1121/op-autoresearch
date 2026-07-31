# Reduce Class operator - ARA Row-Spit branch

> **Applicable scene**: A0>1 (non-tail axis), R > R_max (distribution mode)

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
| **Load mode** | Split (R lines unattended, split chunk processing) |
| **Conditions applicable** | A0>1, R > R_max |
| **Critical direction** | Line direction (in R-minute chunk) |
| **Data continuity** | Inconsistent R elements per column (Spacing A0) |
| **Reduce Results** | vector (A0 values) |

---

## II. Buffer Planning

### 2.1 FP32 scene (Kernel side)

```cpp
// Single Buffer mode (in chunk processing)
constexpr uint32_t a0TileBase = 64;  // FP32

pipe->InitBuffer(inQueueX, 1, R_chunk_size * a0TileBase * sizeof(float));
pipe->InitBuffer(outQueueY, 1, a0TileBase * sizeof(float));  // Final result
pipe->InitBuffer(globalResultBuf, 1, a0TileBase * sizeof(float));  // Global results
pipe->InitBuffer(chunkResultBuf, 1, a0TileBase * sizeof(float));  // chunkResult
pipe->InitBuffer(tmpBuf, 1, tmpBufSize);

// UB = R_chunk×a0TileBase×4 + 3×a0Tile×4 +tmpBufSize
```

### 2.2 tmpBufSize Calculation (Host side)

```cpp
// tmpBufSize based on R_chunk_size × signedCols
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);

constexpr uint32_t VECTOR_REG_WIDTH = 256;
uint32_t perRepeat = VECTOR_REG_WIDTH / sizeof(float);  // 64
uint32_t perBlock = 32 / sizeof(float);  // 8

uint32_t repeats = (R_chunk_size * alignedCols + perRepeat - 1) / perRepeat;
uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * sizeof(float);
tmpBufSize = std::max(tmpBufSize, 4096u);
```

- TileA0Len is the number of columns that each nuclear needs to process after the splitting of A0.
---

## III. Tiling Parameter Calculation

### 3.1 Divisional determination (Host side)

```cpp
// R_max calculation (note: this is a single buffer, different from the full load mode formula, overhead and division)
constexpr uint32_t a0TileBase = 64;  // FP32
uint32_t tmpBufSize = 4096;
uint32_t overhead = 3 * a0TileBase * sizeof(float) + tmpBufSize;
uint32_t R_max = (UB_SIZE - overhead) / (a0TileBase * sizeof(float));
R_max = std::min(R_max, 255u);
R_max = std::max(R_max, 1u);

if (R > R_max) {
    // ARA-Row-Spit mode
    loadMode = LOAD_SPLIT;

    // chunk
    uint32_t R_chunks = (R + R_max - 1) / R_max;
    uint32_t R_chunk_size = R_max;
    uint32_t R_last_chunk_size = R - (R_chunks - 1) * R_chunk_size;
}
```

### 3.2 A0Inner Calculation (Multinuclear Slicing of A0 on the Host side)

```cpp
// Row-Split Buffer used: R_chunk×a0TileBase×4 +3×a0TileBase×4 +tmpBufSize
uint64_t ubPerTileBase = R_chunk_size * a0TileBase * sizeof(float)
                       + 3 * a0TileBase * sizeof(float) + tmpBufSize;
uint64_t fixedOverhead = 3 * a0TileBase * sizeof(float) + tmpBufSize;

int64_t factorMax = (UB_SIZE - fixedOverhead) / ubPerTileBase;
if (factorMax < 1) factorMax = 1;

int64_t a0FactorMax = (A0 + a0TileBase - 1) / a0TileBase;
int64_t totalTilesMax = A1 * a0FactorMax;
int64_t a0InnerMax = (totalTilesMax + blockDim - 1) / blockDim;

int64_t a0Inner = std::min({a0InnerMax, factorMax, a0FactorMax});
a0Inner = std::max(a0Inner, 1L);

uint32_t tileA0Len = a0Inner * a0TileBase;
```

---

## IV. KERNEL IMPLEMENTS

### 4.1 Data flows

```
GM (A1, R, A0) → min R chunk Processing
    ↓
Chunk 0: GM[0:R_chunk, 0:tileA0Len] → UB
    ↓
[ReduceMax/ReduceSum Pattern::Reduce::RA] → chunkResult_0 (tileA0Len)
    ↓
[Update globalResult] → globalResult = merge(globalResult, chunkResult_0)
    ↓
Chunk 1, 2, ... (Repeat)
    ↓
UB → GM (A1, A0)
```

**Why choose Patterson::Reduce::RA**:

The data for each nuclear processing component `(R_chunk, A0_inner)`, after multi-nucleocution, are placed in the UB as follows:
```
[row0All of it.A0_inner, row1All of it.A0_inner, ..., row{R_chunk-1}All of it.A0_inner]
```
is the 2D matrix for `(R_chunk, alignedCols)`.

Pattern::Reduce::RA
- **R**=Reduce dimension (under first dimension of the Convention)
- **A**=Allign dimension (retain second dimension)

For each `a0` position, take the R_chunk value and output the A0_inner result.

**Note:**Unable to use Level 2 API because it can only process continuous data.

### 4.2 Core API Call

#### ReduceMax Dispersed

```cpp
// Number of columns after alignment
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// Initialize global maximum value -∞
LocalTensor<float> globalResultLocal = globalResultBuf.Get<float>();
Duplicate<float>(globalResultLocal, -INFINITY, alignedCols);

for (uint32_t chunkIdx = 0; chunkIdx < R_chunks; chunkIdx++) {
    uint32_t rStart = chunkIdx * R_chunk_size;
    uint32_t rCount = (chunkIdx == R_chunks - 1) ? R_last_chunk_size : R_chunk_size;

    // Load R chunk
    DataCopyExtParams copyParams{static_cast<uint16_t>(rCount),
                                  static_cast<uint32_t>(tileA0Len * sizeof(float)),
                                  static_cast<uint32_t>((A0 - tileA0Len) * sizeof(float)),
                                  0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    // shape will be [rCount, alignedCols] with `[rCount, tileA0Len]` valid data.
    DataCopyPad(xLocal, xGm[rStart * A0], copyParams, padParams);

    // ReduceMax for this chunk (Pattern::Reduce::RA)
    uint32_t srcShape[] = {rCount, alignedCols};
    LocalTensor<float> chunkResultLocal = chunkResultBuf.Get<float>();
    ReduceMax<float, Pattern::Reduce::RA>(chunkResultLocal, xLocal, tmpLocal, srcShape, true);

    // Update globalResult (maximum value by element)
    Max<float>(globalResultLocal, globalResultLocal, chunkResultLocal, alignedCols);
}

// Output final result
DataCopyPad(yGm[offset], globalResultLocal, {1, tileA0Len * sizeof(float), 0, 0});
```

### 4.3 Key Care Points

1. **cross chunk merger**:
   - ReduceMax: Maximum value by element with `Max<float>`
   - ReduceSum: add elements by elements using `Add<float>`

2. **Boundaries Process**: Last R chunk may be smaller than R_chunk_size

3. **Initialization**:
   - ReduceMax: Initialize to the minimum values of `-INFINITY` or data type
   - ReduceSum: Initialize as `0`

4. **Data Access**: need to correctly calculate srcStride over A0

---

## V. common issue

| Problem | Reason | Solutions |
|-----|------|---------|
| accuracy drop | chunk merge logical error | Use Max/Add to merge elements by elements |
| Output Error | Last R chunk size processing error | Use R_last_chunk_size |
| Poor performance | Over and over again. | Optimizing chunk policy |
| Buffer Insufficient | Buffer, it's not rational to plan. | Rational allocation of GM/UB space |
| accuracy is too big. | srcShape uses A0 instead of signedcols | Use the number of columns after alignment |

---

## VI. PERFORMANCEoptimization recommendation

1. **Reduced data access**: Minimize GM visits
2. **chunk Optimization**: R_chunk_size = R_max to ensure that each chunk fully utilizes UB
3. **pipeline**: chunk of different tiles can be processed in parallel
4. **Element-by-Element merger**: using `Max<float>` and `Add<float>` instead of Reduce
