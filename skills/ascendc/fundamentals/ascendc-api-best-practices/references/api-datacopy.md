# DataCopy / DataCopyPad User Guide

Complete guidance for GM ↔ UB data handling.

---

## Contents

1. [selection rule](#selection rule)
2. [32 Byte Alignment Requirements](#32 Byte Alignment Requirements)
3. [DataCopyPad Arguments Detailed](#dataCopyPad - Detailed Parameters)
4. [Use scene example](#use scenario example)
5. [stride parameter elaboration](#stride-parameter elaboration)
6. [common errors and debugging](#common errors and debugging)

---

## Selection Rule

**Principle: priority for DataCopyPad**

| scene | API | Reason |
|-----|-----|------|
| **Unor uncertain alignment** | `DataCopyPad` | Automatically handle alignment/non-matching, avoid boundary bug |
| **Data volume strict 32 byte alignment** | `DataCopy` or `DataCopyPad` | DataCopy is available when alignment is established, DataCopyPad is safer |

### ⛔ ️ blacklist API (prohibited use of production codes)

| API | Prohibited grounds | Allow scenes only |
|-----|---------|-----------|
| `GlobalTensor::SetValue(idx, val)` | It's extremely inefficient, and it's written down individually. | **Only used for debugging** |
| `GlobalTensor::GetValue(idx)` | Extremely inefficient. Modulars read each. | **Only used for debugging** |

```cpp
// ❌ Ban: Production code use SetValue/GetValue
for (uint32_t i = 0; i < size; i++) {
    xGm.SetValue(i, value);    // ⛔️ Very inefficient.
    T val = xGm.GetValue(i);   // ⛔️ Very inefficient.
}

// ✅ Correct: Bulk handling using DataCopyPad
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);

// ✅ allows single-point validation when debugging
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));  // Debug only
```

**Why first? DataCopyPad?**

1. Automatically handle non-matching without manual judgement
2. CopyIn and Copyout apply
3. Tiling may create an unmatched file size when designing
4. Disparities in performance under alignment are negligible

---

## 32 Byte Alignment Requirements

**DataCopy requires 32 byte alignment**, non-matching leads to data errors.

| data type | Aligning Elements | Minimal alignment bytes |
|---------|-----------|--------------|
| half (2 bytes) | 16 | 32 |
| float (4 bytes) | 8 | 32 |
| int32_t (4 bytes) | 8 | 32 |
| fp8 (1 byte) | 32 | 32 |

---

## DataCopyPad Arguments Detailed

### isPad Arguments

| isPad | Meaning |
|-------|------|
| `false` | framework Autofill, no fill value specified by the user |
| `true` | Fill with the user-defined `paddingValue` |

### BlockLen fill behaviour when not aligned

**GM → UB(CopyIn)**:

| Conditions | isPad | dummy fill value |
|-----|-------|-------------|
| leftPadding=0, rightPadding=0 | false | **First element value** |
| leftPadding=0, rightPadding=0 | true | paddingValue |
| leftPading≠0 or rightPading≠0 | false | Random value |
| leftPading≠0 or rightPading≠0 | true | paddingValue |

**UB → GM(CopyOut)**:
- framework Auto-Procedure Non- Alignment
- Automatically discard dummy when moving to GM

### UB End Start Address 32B Alignment (Easier to Pedal)

`DataCopyPad(GM, UB, ...)` and `DataCopyPad(UB, GM, ...)`**UB end start addresses must be aligned with 32 bytes**(blockLen may not be 32B aligned, but no start address).

When indexing to UB Buffer byline, the number of bytes must be 32 times:

```cpp
// ❌ cols * sizeof(elem) is not 32 times, row * cols may not be aligned
// For example, fp8 + cols = 4 bytes per line, only row ∈ {0, 8, 16,...} meet 32B alignment
DataCopyPad(gmOut[off], ubBuf[row * cols], copyParams);
```

Fixing mode: Introduced striding buffer, reordering irregular widening data to 32B alignment in each row:

```cpp
// ✅ Reorder with strided buf to ensure 32B alignment for UB src for each line
auto stridedBuf = strideBuf_.Get<elem_T>();
for (int row = 0; row < mEff; ++row) {
    for (int j = 0; j < cols; ++j) {
        stridedBuf.SetValue(row * 32 + j, ubBuf.GetValue(row * cols + j));
    }
}
DataCopyPad(gmOut[off], stridedBuf[row * 32], copyParams);  // src Each line 32B Alignment
```

**Symptoms of the wrong code**
- `AIV error 80: The UB address accessed by the VEC instruction is not aligned`
- Chain trigger `AIC error: timeout or trap error. subErrType: 0x4`

### BlockCount Parameter Limit

`DataCopyPad` field maximum value of `blockCount`, 4095, exceeds the need for batch handling. Host side Tiling calculation must clip:

```cpp
constexpr uint32_t MAX_BLOCK_COUNT = 4095;
tileRows = std::max(1u, std::min(tileRows, MAX_BLOCK_COUNT));
```

---

## Use scene example

### Scenario 1: Unmatched CopyIn, does not care about filling values

```cpp
// cols = 5 (FP32), blockLen = 20 bytes, not aligned
// Subsequent calculation only handles cols elements, dummy ignored
AscendC::DataCopyParams copyParams{1, cols * sizeof(float), 0, 0};
AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);

// Subsequent calculation only handles cols elements
AscendC::ReduceMax(tmpReduce, xLocal, tmpReduce, cols, false);
```

### scene 2: Unmatch CopyIn, specify the fill value

```cpp
uint32_t padElements = paddedCols - cols;
AscendC::DataCopyPadExtParams<float> padParams{true, 0, padElements, 0.0f};
AscendC::DataCopyExtParams copyParams{1, cols * sizeof(float), 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

### Scenario 3: Unmatched Copyout

```cpp
// CopyOut Auto-Procedure Unmatched, discard dummy while moving to GM
AscendC::DataCopyParams copyParams{1, cols * sizeof(float), 0, 0};
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### Full example: multi-line batch handling

```cpp
__aicore__ inline void CopyInBatch(uint32_t startLocalRow, uint32_t rowsThisTile)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = rowsThisTile;
    copyParams.blockLen = cols * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = paddedColsT - cols;
    padParams.paddingValue = 0;

    AscendC::DataCopyPad(xLocal, xGm[startLocalRow * cols], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}
```

### Scenario 4: line-by-line stand-alone mode (recommended by Softmax/LayerNom)

**Applies to scene**: operator independently calculated by line, etc., Softmax / LayerNom, without crossing Reduce.

**Core element**: blockCount mode + UB alignment storage

```cpp
// Synchronization point.
uint32_t rLength = 13;                           // Number of valid data
uint32_t rLengthAlign = (rLength + 7) / 8 * 8;   // Align to 8 Element()FP32 Down 32 bytes)

// Synchronization point.
AscendC::DataCopyPad(xLocal, xGm[offset],
    {static_cast<uint16_t>(rows),           // blockCount: Lines
     static_cast<uint32_t>(rLength * sizeof(T)), // blockLen: Valid data length (unmatched!)
     0, 0},                                  // stride: Continuous storage
    {false, 0, 0, 0});                       // padParams: Autoprocessing

inQueueX.EnQue(xLocal);
auto xIn = inQueueX.DeQue<T>();

// Synchronization point.
for (uint32_t row = 0; row < rows; row++) {
    // Key: UB offset rLengthAlign, not rLength!
    uint32_t rowOffset = row * rLengthAlign;

    // Reduce API Only rLength (number of valid data)
    AscendC::ReduceMax<T>(rowTmp, xIn[rowOffset], reduceTmp,
        static_cast<int32_t>(rLength), false);
    // ... Sub, Exp, ReduceSum, Div
}

// Synchronization point.
AscendC::DataCopyPad(yGm[offset], yOut,
    {static_cast<uint16_t>(rows), static_cast<uint32_t>(rLength * sizeof(T)), 0, 0, 0});
```

**Key comparison table**

| Parameter Position | Use rLength | Use rLengthAlign |
|---------|-----------|-----------------|
| DataCopyPad blockLen | ✓ | ✗ |
| Reduce API count | ✓ | ✗ |
| Sub/Exp/Div count | ✓ | ✗ |
| UB rowOffset | ✗ | ✓ |
| Buffer Size | ✗ | ✓ |

**UB Data Layout Icon**:

```
GM(continuous storage):  [row0: 13Elements][row1: 13Elements][row2: 13Elements]...
                         ↓ DataCopyPad blockCount Mode
UB(Classed storage):  [row0: 13+3=16][row1: 13+3=16][row2: 13+3=16]...
                         ↑
                  Each line padding to 8 Element Alignment
```

---

## Details of the length parameters

**stride parameter units depend on the operational location**:

| Organisation | Stride Unit | Annotations |
|-----------|------------|------|
| GlobalTensor (GM) | **Bytes** | Byte interval of adjacent data blocks |
| LocalTensor (UB) | **dataBlock (32 bytes)** | 32-byte block spacing of adjacent data blocks |

**stride meaning**: spacing between adjacent data blocks (range of the former tail to the latter head)

### UB → GM multi-line mover (Copyout)

```cpp
// For each row of UB: [cols valid data] [padElements working]
// Adjacent row interval = paddedColsT - cols element
copyParams.blockCount = rowsThisTile;
copyParams.blockLen = cols * sizeof(T);
copyParams.srcStride = (paddedColsT - cols) * sizeof(T) / 32;  // UB stride Units: 32Bytes
copyParams.dstStride = 0;  // GM stride Units: Bytes

AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### Common Errors

```cpp
// ❌ error: srcStride understood line length
copyParams.srcStride = paddedColsT * sizeof(T) / 32;  // This could lead to an output error.

// ✅ Correct: srcStride is interval
copyParams.srcStride = (paddedColsT - cols) * sizeof(T) / 32;
```

---

## Common error and debugging

### Error 1: CopyIn/Copyout

```cpp
// ❌ error
AscendC::DataCopy(xLocal, xGm, 4);  // cols=4 (16 bytes)Data error

// ✅ Correct
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

### Error 2: CopyIn uses DataCopyPad, CopyOut with DataCopy

```cpp
// ❌ error: CopyIn and Copyout both need to process incoherent
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
AscendC::DataCopy(yGm, yLocal, 4);  // Output Error

// ✅ Correct: Use DataCopyPad on both sides
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### Debug Steps

When data error occurs:

1. **Authentication of CopyIn and Copyout**
   - Testing the correct handling with "CopyIn → Copyout"
2. **Checks for 32 byte alignment of data**
3. **Unmatched scene: CopyIn and Copyout used DataCopyPad**

### Case in action: SoftmaxV5

**Question**: FP32 cols=4,5,6,7 misdirected, cols=8 normal

**Roots**:
1. CopyIn uses DataCopyPad but isPad=false (fill random values)
2. Copyout handles non-matched output with DataCopy

**Solve**: CopyIn and Copyout used DataCopyPad
