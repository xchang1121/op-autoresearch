# Reduce Class operator - AR Col-Spit branch

> **Applicable scene**: A0=1 (endaxis), R > present (distribution mode)

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
| **Template type** | AR Template (Axes + R) |
| **Shape Abstract** | (A1, R) |
| **Load mode** | Split (unable to leave whole row, split chunk processing) |
| **Conditions applicable** | A0 = 1, R > [full load threshold](#3.1 full load threshold) |
| **Critical direction** | Column Orientation (in R-point chunk) |
| **Data continuity** | R elements per row continuous |
| **Reduce Results** | scalar (1 value) |

---

## II. Buffer Planning

### 2.1 FP32 scene (Kernel side)

```cpp
// Single Buffer mode (in chunk processing)
pipe->InitBuffer(inQueueX, 1, chunkCols * sizeof(float));
pipe->InitBuffer(outQueueY, 1, 32);  // ReduceOutcome (%)1individualscalar)
pipe->InitBuffer(chunkResultBuf, 1, 32);  // chunkIntermediate result
pipe->InitBuffer(tmpBuf, 1, tmpBufSize);

// UB = chunkCols×4 + 32 + tmpBufSize
```

### 2.2 ChinkCols Calculation (Host side)

```cpp
// UB-based capacity calculation: Maximum number of columns to support when 1 line calculation is completed
chunkCols = std::min(chunkCols, R);
uint32_t numChunks = (R + chunkCols - 1) / chunkCols;
uint32_t lastChunkSize = R - (numChunks - 1) * chunkCols;
```

---

## III. Tiling Parameter Calculation

### 3.1 Divisional determinations

Explanation: When you complete the calculation of one line of data in the UB, the maximum number of columns to be supported is chunkcols. If R > chunkcols is required to be divided.

### 3.2 Polynuclear cut parameters (Host side)

```cpp
// Split by A1 (line)
uint32_t rowsPerCore = (A1 + blockDim - 1) / blockDim;
uint32_t usedCoreNum = (A1 + rowsPerCore - 1) / rowsPerCore;
uint32_t tailCoreRows = A1 % rowsPerCore;
if (tailCoreRows == 0 && A1 > 0) tailCoreRows = rowsPerCore;
```

---

## IV. KERNEL IMPLEMENTS

### 4.1 Data flows

```
GM (A1, R) → min chunk Processing
    ↓
Chunk 0: GM[0:chunkCols] → UB
    ↓
[ReduceMax] → chunkResult_0
    ↓
[Update globalResult] → globalResult = merge(globalResult, chunkResult_0)
    ↓
Chunk 1, 2, ... (Repeat)
    ↓
UB → GM (A1)
```

### 4.2 Core API Call

#### ReduceMax Distribution Achieved (Kernel side)

```cpp
// Initialize global maximum value -∞
float globalMax = -INFINITY;

for (uint32_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    uint32_t chunkStart = chunkIdx * chunkCols;
    uint32_t chunkSize = (chunkIdx == numChunks - 1) ? lastChunkSize : chunkCols;

    // Load chunk. use `DataCopyPad` in case last chunk is not 32 bytes aligned.
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(chunkSize * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[chunkStart], copyParams, padParams);

    // ReduceMax for this chunk
    ReduceMax<float>(chunkResultLocal, xLocal, tmpLocal, chunkSize, false);
    float chunkMax = chunkResultLocal.GetValue(0);

    // Update `globalMax`. DO NOT use `std::` function.
    if (chunkMax > globalMax) {
        globalMax = chunkMax;
    }
}

// use `globalMax` to complete the reset calculation...
```

### 4.3 Key Care Points

1. **cross chunk merger**:
   - ReduceMax: use `if` or 'Trimenc 'to take maximum value
   - ReduceSum: use `+=` API cumulative
2. **Border processing**: last chunk may be smaller than chunkcols. Use `CopyDataPad` when moving in/out
3. **Initialization**:
   - ReduceMax: Initialize to the minimum values of `-INFINITY` or data type
   - ReduceSum: Initialize as `0`

---

## V. common issue

| Problem | Reason | Solutions |
|-----|------|---------|
| accuracy drop | chunk merge logical error | Ensure correct merger (Max takes maximum value, Sum add) |
| Output Error | Last chunk size processing error | Use lastChunkSize instead of chunkCols |
| Poor performance | Over and over again. | Optimize the chunk strategy and reduce the number of times |
| Buffer Insufficient | cunkCols Calculator Error | Based on UB capacity correct calculation |

---

## VI. PERFORMANCEoptimization recommendation

1. **chunk Optimization**: Select the best chunk size based on UB capacity to minimize the number of chunks
2. **Avoiding unnecessary copies of data**: consolidation directly on chunk results and reduction of intermediate storage
