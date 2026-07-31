# API Optimization Guide for arithmetical operations

> **Applicable scenario**: Optimistic means of achieving the best by using algorithms to calculate API (Add/Sub/Mul/Div), avoiding unnecessary broadcast buffer and command expenses.

---

## Contents

- [Overview](# OVERVIEW)
- [Scene 1: scalar Operations (single line)](#Scene 1 scalar Operations Line)
  - [Programme comparison](# programme comparison)
  - [API interface](#api-interface)
  - [full example](#full example)
- [Scene 2: Broadcast operation (multiline)](#Scene 2 broadcast operation multiline)
  - [Programme comparison-1](#programme comparison--1)
  - [core principles](# core principles)
  - [Short processing](# batch processing)
- [Scene 3: Half accuracy plus minus accuracy Optimum](#accuracy 3: minus accuracy Optimum)
  - [Question Roots](# Problem Roots)
  - [default policy](#default policy)
  - [Standard paradigm](#Standard paradigm)
  - [Kernel Integration Point](#kernel - Integration Point)
- [Performance comparison](#Performance comparison)
- [API applies](#Api applies)
- [Annual errors](#Annual errors)

---

## Overview

API (Add/Sub/Mul/Div) supports two modes of use:

| Mode | API | Apply scene | Buffer Requirements |
|-----|-----|---------|------------|
| **scalar operation** | `Adds/Muls` | Single-line processing (Softmax AR template) | 32B |
| **Broadcasting operation** | `Sub/Div + BinaryRepeatParams` | Multiline processing (Softmax ARA template) | alignedCols×4 |

**Key optimization**:
- Single line: Avoid Duplicate with `Adds/Muls`
- Multiline: Use `src1RepStride=0` to avoid a line-by-line cycle

---

## Scenario 1: scalar Operations (one line)

### Programme comparison

**Question**: tensor needs to execute `x - scalar` or `x / scalar` for each element

**Typical scene**:
- Softmax AR template: `x - max_val` (value stable)
- Softmax AR template: `exp(x) / sum` (consolidation)
- Layer Norm: `x - mean` (centralized)
- BatchNorm:`x * gamma + beta`

**Programme comparison**:

| Programme | Commands | Buffer Requirements | Recommended level |
|-----|--------|------------|--------|
| Duplicate + Sub | Article 2 | `rLength × sizeof(T)` | ⭐⭐ |
| Duplicate + Div | Article 2 | `rLength × sizeof(T)` | ⭐⭐ |
| **Adds(-scalar)** | **1 Article** | **32B** | **⭐⭐⭐⭐⭐** |
| **Muls(1/scalar)** | **1 Article** | **32B** | **⭐⭐⭐⭐⭐** |

### API Interface

**Adds (scalar plus)**
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src,
    const T& scalarValue,
    const int32_t& count);

// Function: dst[i] = src[i] + scalarValue
// Example: Adds(dst, src, -maxVal, count)/ / Subtract Add
```

**Muls (scalar multiplier)**:
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src,
    const T& scalarValue,
    const int32_t& count);

// Function: dst[i] = src[i] * scalarValue
// Example: Muls(dst, src, 1.0/sum, count) // Division Multiplication
```

### Full Example

#### Before Optimizing (Sub/Div + Duplicate)

```cpp
// Buffer Initialization
uint32_t broadcastBufSize = rLengthAlign * sizeof(T);  // For example:512B (rLength=128, FP32)
pipe.InitBuffer(broadcastBuf, broadcastBufSize);
pipe.InitBuffer(reduceBuf, reduceBufSize);

// Compute
LocalTensor<T> broadcastLocal = broadcastBuf.Get<T>();

for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;

    // Step 1: ReduceMax
    ReduceMax<T>(broadcastLocal, xLocal[rowOffset], reduceTmpLocal, rLength, false);

    // Step 2: Duplicate + Sub (buffer required)
    T maxVal = broadcastLocal.GetValue(0);
    Duplicate<T>(broadcastLocal, maxVal, rLength);  // Command 1
    Sub<T>(yLocal[rowOffset], xLocal[rowOffset], broadcastLocal, rLength);  // Command 2

    // Step 3: Exp
    Exp<T>(yLocal[rowOffset], yLocal[rowOffset], rLength);

    // Step 4: ReduceSum
    ReduceSum<T, true>(broadcastLocal, yLocal[rowOffset], reduceTmpLocal, rLength);

    // Step 5: Duplicate + Div (buffer required for broadcast)
    T sumVal = broadcastLocal.GetValue(0);
    Duplicate<T>(broadcastLocal, sumVal, rLength);  // Command 3
    Div<T>(yLocal[rowOffset], yLocal[rowOffset], broadcastLocal, rLength);  // Command 4
}

// Total: 6 directives/lines required broadcast Buf (512B for rLength = 128)
```

#### Optimized (Adds/ Muls + scalar)

```cpp
// Buffer Initialization (broadcast Buf)
uint32_t scalarBufSize = 32;  // Minimum alignment requirements, storage only 1 individualscalar
pipe.InitBuffer(scalarBuf, scalarBufSize);
pipe.InitBuffer(reduceBuf, reduceBufSize);

// Compute
LocalTensor<T> scalarLocal = scalarBuf.Get<T>();

for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;

    // Step 1: ReduceMax
    ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmpLocal, rLength, false);

    // Step 2: Adds (direct scalar operation, no broadcast)
    T maxVal = scalarLocal.GetValue(0);
    Adds<T>(yLocal[rowOffset], xLocal[rowOffset], -maxVal, rLength);  // Command 1

    // Step 3: Exp
    Exp<T>(yLocal[rowOffset], yLocal[rowOffset], rLength);

    // Step 4: ReduceSum
    ReduceSum<T, true>(scalarLocal, yLocal[rowOffset], reduceTmpLocal, rLength);

    // Step 5: Muls (separate multiplication, direct scalar operation)
    T sumVal = scalarLocal.GetValue(0);
    T invSumVal = (T)1.0 / sumVal;  // CPU End Calculating 1/sum
    Muls<T>(yLocal[rowOffset], yLocal[rowOffset], invSumVal, rLength);  // Command 2
}

// Grand total: 4 directives/lines, savings in classcast Buf (480B for rLength = 128)
```

---

## Scenario 2: Broadcast operations (multi-line)

### Programme comparison

**Question**: The same scalar operation (e.g. `x - max`, `exp / sum`) is required for multiline data

**Programme comparison**:

| Programme | API Call | Buffer Requirements | Recommended level |
|-----|---------|------------|--------|
| Line-by-line cycle | R times | alignedCols×4 | ⭐⭐ |
| Single broadcasts (R ≤ 64) | 1 time | alignedCols×4 | ⭐⭐⭐⭐⭐ |
| Batch broadcasts (R > 64) | ceil (R/64) | alignedCols×4 | ⭐⭐⭐⭐⭐ |

### Core principles

**Binary RepeatParams.src1RepStride=0

```cpp
struct BinaryRepeatParams {
    uint8_t dstBlkStride;    // I'm not sure what I'm talking about.dst of block Step length
    uint8_t src0BlkStride;   // I'm not sure what I'm talking about.src0 of block Step length
    uint8_t src1BlkStride;   // I'm not sure what I'm talking about.src1 of block Step length
    uint8_t dstRepStride;    // It's not like we're in the middle of nowhere.dst of block Step length
    uint8_t src0RepStride;   // It's not like we're in the middle of nowhere.src0 of block Step length
    uint8_t src1RepStride;   // =0 Making it happen.
};
```

**Working principles**
- `dstRepStride = alignedCols/8`: each iterative, dst forwards `alignedCols` elements
- `src0RepStride = alignedCols/8`: Src0 Forwards `alignedCols` Element
- `src1RepStride = 0`: every iterative, src1**Do not move**, repeat reading the same location

**Effect**:
```
Organisation 0: dst[0:cols]     = src0[0:cols]     - src1[0:cols]
Organisation 1: dst[cols:2cols] = src0[cols:2cols] - src1[0:cols]  ← Repeat Read
Organisation 2: dst[2cols:3cols]= src0[2cols:3cols]- src1[0:cols]  ← Repeat Read
```

### Batch processing

#### Option 1: Line-by-line cycle (inefficient)

```cpp
for (uint32_t r = 0; r < R; r++) {
    Sub(dstLocal[r * alignedCols], srcLocal[r * alignedCols], scalarLocal, alignedCols);
}
// API Call: R times
```

#### Option 2: Single broadcasts (efficient, R ≤ 64)

```cpp
uint64_t mask = alignedCols;
uint8_t repeatTime = R;

Sub(dstLocal, srcLocal, scalarLocal, mask, repeatTime,
    {1, 1, 1, alignedCols/8, alignedCols/8, 0});
// API Call: 1 call
// Performance enhancement: R multiple
```

#### Programme 3: Batch broadcasting (efficiency, R > 64)

```cpp
constexpr uint32_t BATCH_SIZE = 64;
uint32_t totalBatches = (R + BATCH_SIZE - 1) / BATCH_SIZE;  // ceil(R/64)

for (uint32_t batch = 0; batch < totalBatches; batch++) {
    uint32_t startRow = batch * BATCH_SIZE;
    uint8_t repeatTime = (startRow + BATCH_SIZE <= R) ? BATCH_SIZE : (R - startRow);
    uint32_t offset = startRow * alignedCols;

    Sub(dstLocal[offset], srcLocal[offset], scalarLocal,
        mask, repeatTime, {1, 1, 1, alignedCols/8, alignedCols/8, 0});
}
// API Call: ceil (R/64)
// Performance enhancement: about 64 times
```

---

## Scenario 3: Half accuracy plus minus accuracy Optimization

### The root causes of the problem

Half-accuracy (FP16 = 10-bit end, BF16 = 7-bit) will be at the same risk for the two orders of magnitude "**big**small**, Add and Sub:

```
a = 1024.0, b = 0.0625
  Add<half>  : 1024.0     ← b Abandoned.     Sub<half>  : 1024.0     ← b Abandoned.
  Add<float> : 1024.0625  ← Correct.         Sub<float> : 1023.9375  ← Correct.
```

Critical margin (notable degradation threshold):FP16 ≈ 2¹⁰=1024,BF16 ≈ 2⁷=128;total loss threshold approximately2×(end count implied)1bits). PlusNThreshold threshold divided by√N.

### Default Policy

**spec does not explicitly "input the same level" up to FP32**. The generic operator caller distribution is unknown and is not controlled in the event of a disability/aggregation/consolidation/quantitative inverse. Add and Sub apply the same rule, with only a different threshold value for BF16 and FP16 (see below).

| Spec declares input of the same magnitude? | Recommended realization | Rationale |
|---------------------|---------|------|
| No (default) | `Cast → Add/Sub<float>(in-place) → Cast` | Overwrite All Distributions |
| Yes (mask superimpose, standardized probabilities, etc.) | Direct `Add/Sub<half>` | The two inputs themselves are only 10/7 bits accuracy and no additional losses are included in the single operation; no √N amplified |

### Standard Model

`Add/Sub<float>(dst, src0, src1)` supports dst and src aliases, only**K=2 copies**FP32 temporary space (dst reuse src0Fp32):

```cpp
// The left of Get<T>(len) is the number of elements; offset by tensor [N]
auto src0Fp32 = tmpBuf.Get<float>(TILE);
auto src1Fp32 = src0Fp32[TILE];

// half → floatUse it.CAST_NONE;float → halfUse it.CAST_ROUND
AscendC::Cast<float, half>(src0Fp32, src0, AscendC::RoundMode::CAST_NONE, count);
AscendC::Cast<float, half>(src1Fp32, src1, AscendC::RoundMode::CAST_NONE, count);
AscendC::Add<float>(src0Fp32, src0Fp32, src1Fp32, count);   // in-place;Sub Same thing.
AscendC::Cast<half, float>(dst, src0Fp32, AscendC::RoundMode::CAST_ROUND, count);
```

Cost:+3Directives (total)4Article:2 Cast↑ + 1 Add/Sub + 1 Cast↓),+K×count×sizeof(float) UB.BF16Path will be`half`Replace with`bfloat16_t`It's okay.

> **API aliases binding decision K**: `Add/Sub<float>` supports dst and src aliases on Victor, so K=2; Reduce Class API prohibits dst=tmpBuffer, which is not comparable.

### Kernel Integration Points

> The ascending accuracy path requires K=2 copies of provisional FP32 Buffer, Add/Sub<float> supports dst/src alias dst reuse src0Fp32. accuracy converts RoundMode as detailed in [api-precision.md] (api-precision.md).

---

## Performance Comparison

### scalar Operations (one line)

| Item | Before Optimizing | Optimised | Improvement |
|-----|--------|--------|------|
| **Guidances/lines** | Article 6 | Article 4 | **-33%** |
| **Buffer Size** | 512B (rLength=128) | 32B | **-94%** |
| **UB Savings** | - | ~480B | Could be used for larger rowsPerLoop |

### Broadcast operation (multi-line)

| R (lines) | Line-by-line cycle | Single broadcasts | Battery broadcasts | Performance enhancement |
|---------|---------|---------|---------|---------|
| 32 | 32 times | 1 time | - | **32×** |
| 64 | 64 times | 1 time | - | **64×** |
| 100 | 100 times | - | 2 times | **50×** |
| 128 | 128 times | - | 2 times | **64×** |
| 200 | 200 times. | - | 4 times | **50×** |

### Half-accuracy plus minus (FP16/BF16 Add/Sub)

The route to accuracy is relatively direct, `Add/Sub<half>`: +3 commands (4 in total), +2× Count ×sizeof (float) UB. Apply the scene [Scene 3 default policy](#default policy).

### Example (Softmax ARA branch)

**Scene**: R=128, signed Cols=64, FP32

| Operation | Before Optimizing | Optimised | Raise |
|-----|--------|--------|------|
| Sub (x-max) | 128 times | 2 times | 64× |
| Div (exp/sum) | 128 times | 2 times | 64× |
| Total** | **256 times** | **4 times** | **64×** |

---

## Application of API

Binary Operations for all supporting `BinaryRepeatParams` API:

| API | Purpose | Single Line Optimization | Multiline Optimization |
|-----|------|---------|---------|
| **Add** | Add | Adds | src1RepStride=0 |
| **Sub** | Subtract | Adds(-val) | src1RepStride=0 |
| **Mul** | Multiplication | Muls | src1RepStride=0 |
| **Div** | Division | Muls(1/val) | src1RepStride=0 |
| **Max** | Maximum value | - | src1RepStride=0 |
| **Min** | Min | - | src1RepStride=0 |

---

## Common Errors

| Error | Reason | Solutions |
|-----|------|---------|
| Could not close temporary folder: %s | `mask > 64` (FP32) | Batch processing or back looping |
| Data Error | `src1RepStride` is not set to 0 | Confirm parameters: `{..., 0}` |
| Partial Line Correct | calculator error | `offset = startRow * alignedCols` |
| Cross-border collapse | calculation error for repeatTime | Use Triple Operations |
| Buffer Insufficient | Use Duplicate Schemes | Change to Adds/ Muls |
| dst == tmpBuffer | Reduce API Limit | Use different buffer |
| FP16/BF16 plus minus accuracy lost | Directly `Add/Sub<half>` | accuracy: `Cast→FP32 Add/Sub(in-place)→Cast` |
| Half-accuracy plus minus | Temporary Buffer Insufficient | Save `2 × count × sizeof(float)`, Add/Sub reuse src0Fp32 |
| `Get<T>(len)` takes out an abnormal length | Wrong number of bytes as elements | `len` is the number of elements, not bytes |

---

## Checklist

When using algorithms to calculate API, ensure that:

**scalar Operations (one line)**:
- [ ] Use `Adds(-scalar)` instead of `Duplicate + Sub`
- [ ] Use `Muls(1/scalar)` instead of `Duplicate + Div`
- [ ] scalar division converted to multiplication (CPU end calculation 1/scalar)

**Radio operations (multilines)**:
- [ ] alignedCols ≤ 64 (FP32) / ≤ 128 (FP16)
- [ ] Use `src1RepStride = 0` for broadcast
- [ ] Use batch processing for R > 64
- [ ] Ofset correct calculation: `offset = startRow * alignedCols`

**Semi-accuracy plus minus (FP16/BF16 Add/Sub)**:
- [ ] Default to raise accuracy; direct `Add/Sub<half>` is only allowed when spec explicitly "input equals"
- [ ] Temporary Buffer set aside `K × count × sizeof(float)`, `Add/Sub<float>` for aliases K=2, resc0Fp32 (in-place)
- [ ] The left of `Get<T>(len)` is the number of elements; offset by `tensor[N]`
- [ ] Cast direction: `half→float` with `CAST_NONE`, `float→half` with `CAST_ROUND`

---

## References

- [ Binary Repeat Params Structure] (../../../asc-devkit/docs/api/context/BinaryRepeatParams.md)
- [Adds API](../../../asc-devkit/docs/api/context/Adds.md)
- [Muls API](../../../asc-devkit/docs/api/context/Muls.md)
- [Sub API](../../../asc-devkit/docs/api/context/Sub.md)
- [Div API](../../../asc-devkit/docs/api/context/Div.md)
