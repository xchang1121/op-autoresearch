# Reduce Patterson Interface Detailed

Advanced use of the Pattern interface for cross-line batch Reduce.

---

## Two Reloading Forms

### Form 1: Visible transfer of shared TmpBuffer (recommended)

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(
    const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer,  // Visible Input
    const uint32_t srcShape[],
    bool srcInnerPad
);
```

### Form 2: framework automatic application for temporary space

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(
    const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& srcTensor,
    const uint32_t srcShape[],
    bool srcInnerPad
);
```

> **⚠ ️ Form 2 must reserve temporary space**or runtime UB crosses the border. For more details, see [Temporary Space Reserve](#Temporary Space Reserve).

---

## Pattern type

| Pattern | Direction | Enter shape | Output shape | Purpose |
|---------|-----|---------|---------|------|
| `Pattern::Reduce::AR` | Follow the last dimension (column direction) | (R, C) | (R,) | Each line is about one value. |
| `Pattern::Reduce::RA` | Along the first dimension (line direction) | (R, C) | (C,) | Each column is about 1 value |

---

## Description of parameters

| Parameters | Type | Annotations |
|-----|------|------|
| `T` | half/float | data type |
| `pattern` | Pattern::Reduce::AR/RA | Reunification Mode |
| `isReuseSource` | bool | Reuse Source Operations (Default false) |
| `dstTensor` | LocalTensor\<T\> | Output tensor |
| `srcTensor` | LocalTensor\<T\> | Enter tensor |
| `sharedTmpBuffer` | LocalTensor\<uint8_t\> | Temporary cache (Form 1) |
| `srcShape` | uint32_t[] | `{rows, alignedCols}`,**alignedCols must be 32 byte aligned** |
| `srcInnerPad` | bool | A2/A3 Chip only supports `true` |

---

## Temporary space reserved

**Both forms require temporary space**:

| Modalities | Retention Method | Strengths | Recommended level |
|-----|---------|------|-------|
| **Form 1** | `InitBuffer(tmpBuf, tmpSize)`+ Visible Input | Memory Controllable, Reusable | ⭐⭐⭐⭐⭐ |
| **Form 2** | `InitBuffer(tmpBuf, tmpSize)` (used automatically by framework) | The code is simple. | ⭐⭐⭐ |

**Temporary space size calculations**:

```cpp
#include "kernel_operator.h"

uint32_t maxSize, minSize;
AscendC::GetReduceMaxMaxMinTmpSize(srcShape, sizeof(T), isReuse, maxSize, minSize);

// Use maxSize (safe) or minSize (saving memory)
pipe->InitBuffer(tmpBuf, maxSize);
```

Reference document: `asc-devkit/docs/api/context/GetReduceMaxMaxMinTmpSize.md`

---

## Full Example

### Example 1: ReduceMax (AR/RA Patterson)

```cpp
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();
AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

uint32_t srcShape[] = {rows, alignedCols};  // alignedCols I have to. 32 Byte Alignment
constexpr bool isReuse = true;

// AR Patterson: About 1 value → outputrows per line
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, isReuse>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);

// RA Pattern: About 1 value per column → output signedcols
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, isReuse>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);
```

### Example 2: ReduceSum (framework automatic application)

```cpp
// ⚠ ️ has to set aside temporary space in advance.
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();

uint32_t srcShape[] = {rows, alignedCols};

AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
    dstLocal, srcLocal, srcShape, true);
```

---

## Comparative summary

| Contrast | Form 1 (obvious inflow) | Form 2 (framework application) |
|-------|-----------------|------------------|
| tmp Arguments | ✅ Visible Inflow | Automatic application for ❌ framework |
| Preserve Space | ✅ is fine when called. | ⚠ ️**has to stay in InitBuffer** |
| Memory management | Manually managed, reusable | Pre-encumbered, easily missed. |
| Recommended level | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Recommended use of form 1**to avoid the omission of reserved space leading to runtime errors.
