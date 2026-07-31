# Reduce Class operator - AR Full Load Branch

> **Applicable scene**: A0 = 1 (tail axis), R ≤ present (full load mode)

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
| **Load mode** | Full load (full line in UB) |
| **Conditions applicable** | A0 = 1, R ≤ [full load threshold](#3.1 full load threshold) |
| **Data continuity** | R elements per row continuous |
| **Reduce Results** | scalar (1 value) |

---

## II. Buffer Planning

### 2.1 FP32 scene (Kernel side)

```cpp
// Double Buffer Mode
pipe->InitBuffer(inQueueX, 2, R * sizeof(float));
pipe->InitBuffer(outQueueY, 2, 32);  // ReduceOutcome (%)1individualscalar)
pipe->InitBuffer(tmpBuf, tmpBufSize);

// UB = 2 × R × 4 + 2 × 32 + tmpBufSize
```

### 2.2 tmpBufSize Calculation (Host side)

```cpp
uint32_t ComputeReduceBufSize(uint32_t rLengthAlign, uint32_t typeSize) {
    uint32_t perRepeat = 256 / typeSize;  // 64 for FP32
    uint32_t perBlock = 32 / typeSize;     // 8 for FP32
    uint32_t repeats = (rLengthAlign + perRepeat - 1) / perRepeat;
    uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * typeSize;
    return std::max(tmpBufSize, 4096u);  // Min 4KB
}
```

---

## III. Tiling Parameter Calculation

### 3.1 Full load threshold

Explanation: The maximum number of columns R_max is supported when one line of data is calculated in the UB.

### 3.2 Polynuclear cut parameters (Host side)

```cpp
// Split by A1 (line)
uint32_t rowsPerCore = (A1 + blockDim - 1) / blockDim;
uint32_t usedCoreNum = (A1 + rowsPerCore - 1) / rowsPerCore;
uint32_t tailCoreRows = A1 % rowsPerCore;
if (tailCoreRows == 0 && A1 > 0) tailCoreRows = rowsPerCore;
```

### 3.3 Alignment (Kernel side)

```cpp
// Calculates the number of columns after alignment
uint32_t alignedCols = ((R * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// Use DataCopyPad to process non-matching (copy only one line). If UB is richer, you can make multiple batch copies)
DataCopyExtParams copyParams{1, static_cast<uint32_t>(R * sizeof(float)), 0, 0, 0};
DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
```

---

## IV. KERNEL IMPLEMENTS

### 4.1 Data flows

```
GM (A1, R) → UB (R)
    ↓
[ReduceMax/ReduceSum] → result (1)
    ↓
UB (1) → GM (A1)
```

### 4.1 Core API Call (Kernel side)

> **Recommended**: AR fully used**Level 2 interface (line-by-line)**, simpler and unmatched

```cpp
for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;  // ⚠️ Key: Length after alignment

    // ReduceMax - Use number of valid data
    ReduceMax<T>(resultLocal, xLocal[rowOffset], tmpLocal,
                 static_cast<int32_t>(rLength), false);

    // Or ReduceSum.
    ReduceSum<T>(resultLocal, xLocal[rowOffset], tmpLocal,
                 static_cast<int32_t>(rLength), false);

    // Output result
    T result = resultLocal.GetValue(0);
    // Follow-up
}
```

**Key points**:
- `rowOffset` calculation: Use `rLengthAlign` (stored by 32 byte for each line in UB)
- API `count` parameter: use `rLength` (processing valid data only, excluding padding)
- Buffer size: with `rLengthAlign` (required to accommodate data after alignment)

### 4.2 pipeline Design

**Double Buffer mode**(`InitBuffer(que, 2, size)`):
```
Tile N:   CopyIn(row0) → Compute(row0) → CopyOut(row0)
Tile N+1:              CopyIn(row1) → Compute(row1) → CopyOut(row1)
```

---

## V. common issue

| Problem | Reason | Solutions |
|-----|------|---------|
| Output All 0 | Buffer Incorrect Initialization | Check AllocTensor/FreeTensor pairs |
| FP16 accuracy | Moderate calculation accuracy is insufficient | Calculate using FP32 middle |
| Inconsistent scene accuracy error | Count with rLengthAlign instead of rLength | Count parameters with `rLength` (number of valid data) |
| Multiple-line Data Error | Rowofset Calculator Error | Rowofset with `rLengthAlign` (UB-Systemed) |

---

## Performance/accuracy optimization recommendation

1. **Double Buffer**: Open with `InitBuffer(que, 2, size)`, CopyIn /Compute/CopyOut
2. **FP16 Mixed accuracy**: Sum/Mean/Prod etc. under contract scenario: BF16/FP16 input, FP32 calculation, BF16/FP16 output
