# Reduce API User Guide

Line-by-line Reduce selection and use of the API selection for cross-reduce.

---

## Contents

1. [Interface Selection](# Interface Selection)
2. [Level 2 interface (line-by-line processing)](#level-2 -- interface-by-line processing)
3. [Pattern interface (cross-line batch)](#pattern -- interface cross-line batch)
4. [Annual errors](#Annual errors)
5. [best practice](#best practice)

---

## Interface Selection

| scene | Interface | Parameters | Alignment Requirements | Typical uses |
|-----|------|------|----------|---------|
| Separately, line by line | Level 2 | `(dst, src, tmp, count)` | **Not available** | Softmax, LayerNorm |
| Bulk processing across rows | Pattern | **Two forms**(see below) | 32 Bytes | ReduceSum axis=-1 |

**The selection principle**
- Independent line-by-line calculation of →**Level 2 interface**(simplistic, no matching requirement)
- Need to cross line Reduce →**Pattern interface**(higher performance, recommended form 1)

---

## Level 2 interface (line-by-line)

### API Signature

```cpp
AscendC::ReduceMax<T>(dst, src, tmpBuffer, count, calIndex);
AscendC::ReduceSum<T, isSetMask=true>(dst, src, tmpBuffer, count);
AscendC::ReduceMin<T>(dst, src, tmpBuffer, count, calIndex);
```

**Parameters**:
- `dst`: Output LocalTensor (1 element)
- `src`: Enter LocalTensor (count elements)
- `tmpBuffer`: Temporary Buffer (**types must be the same as T**)
- `count`: Number of elements (int32_t)
- `calIndex`: Whether or not to calculate the index (bool, default false)

### tmpBuffer type requirements

**tmpBuffer type must be the same as dst/src**:

```cpp
// ❌ error: tmpBuffer type not matched
AscendC::LocalTensor<uint8_t> tmpBuffer = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax(rowTmp, src, tmpBuffer, count);  // Compiler error!

// ✅ Correct: tmpBuffer type must be the same as T
AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();
AscendC::ReduceMax(rowTmp, src, reduceTmp, count);
```

### Full example: Softmax-by-line

```cpp
__aicore__ inline void ProcessRow(
    AscendC::LocalTensor<T>& xLocal,
    AscendC::LocalTensor<T>& yLocal,
    uint32_t rowIdx)
{
    uint32_t rowOffset = rowIdx * rLengthAlign;  // ⚠️ Use it. rLengthAlign

    AscendC::LocalTensor<T> rowTmp = rowBuf.Get<T>();
    AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();

    // ReduceMax (count = rLength, number of valid data)
    AscendC::ReduceMax<T>(rowTmp, xLocal[rowOffset], reduceTmp,
        static_cast<int32_t>(rLength), false);

    T maxVal = rowTmp.GetValue(0);
    AscendC::Duplicate<T>(rowTmp, maxVal, rLength);
    AscendC::Sub<T>(xLocal[rowOffset], xLocal[rowOffset], rowTmp, rLength);

    // 2. Exp
    AscendC::Exp<T>(xLocal[rowOffset], xLocal[rowOffset], rLength);

    // 3. ReduceSum
    AscendC::ReduceSum<T, true>(rowTmp, xLocal[rowOffset], reduceTmp,
        static_cast<int32_t>(rLength));

    T sumVal = rowTmp.GetValue(0);
    AscendC::Duplicate<T>(rowTmp, sumVal, rLength);
    AscendC::Div<T>(yLocal[rowOffset], xLocal[rowOffset], rowTmp, rLength);
}
```

---

## Pattern interface (lined batch)

The Pattern interface has**two overload formats**as detailed in [api-reduce-pattern.md] (api-reduce-pattern.md).

### Quick start.

```cpp
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();
AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

uint32_t srcShape[] = {rows, alignedCols};  // alignedCols I have to. 32 Byte Alignment

// Recommended use form 1: Visible transfer to tmpLocal
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);
```

### Key points

| Points | Annotations |
|-----|------|
| **Alignment requirements** | `alignedCols` must be 32 bytes aligned |
| **Pattern Type** | `Pattern::Reduce::AR` (in column direction), `Pattern::Reduce::RA` (in row direction) |
| **Recommended form** | Form 1 (visible transfer of shared TmpBuffer) |
| **Temporary space** | Both forms need to be reserved, as detailed below.[api-reduce-pattern.md](api-reduce-pattern.md) |

### Non-recognizing data processing

```cpp
// ✅ Option 1: Change to a Level 2 interface (no matching requirement)
AscendC::ReduceMax<T>(dst, src, tmp, rLength, false);

// ✅ option 2: Fill in with DataCopyPad to align
uint32_t alignedCols = ((rLength * sizeof(T) + 31) / 32) * 32 / sizeof(T);
AscendC::DataCopyPadExtParams<T> padParams;
padParams.isPad = true;
padParams.rightPadding = alignedCols - rLength;
DataCopyPad(dstLocal, srcGm, copyParams, padParams);

uint32_t srcShape[] = {1, alignedCols};
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, true);
```

---

## Common Errors

### Error 1: tmpBuffer type not matched

```cpp
// ❌ error
AscendC::LocalTensor<uint8_t> tmpBuffer = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax(rowTmp, src, tmpBuffer, count);

// ✅ Correct
AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();
AscendC::ReduceMax(rowTmp, src, reduceTmp, count);
```

### Error 2: RowOffset with rLength instead of rLengthAlign

```cpp
// ❌ error: single-line, multi-line failed
uint32_t rowOffset = rowIdx * rLength;

// ✅ Correct
uint32_t rowOffset = rowIdx * rLengthAlign;
```

### Error 3: Unmatched data with Patterson interface

```cpp
// ❌ Error: rLength=13, not 32 byte alignment
uint32_t srcShape[] = {1, rLength};
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, false);

// ✅ Option 1: Change to the Level 2 interface
AscendC::ReduceMax<T>(dst, src, tmp, rLength, false);

// ✅ option 2: Fill in the alignment with DataCopyPad (see above)
```

### Error 4: Reduce API count pass rLengthAlign

```cpp
// ❌ error: count should be the number of valid data
AscendC::ReduceMax(rowTmp, src, tmp, rLengthAlign, false);

// ✅ Correct: count only valid numbers
AscendC::ReduceMax(rowTmp, src, tmp, rLength, false);
```

### Error 5: Pattern interface format 2 forgot to set aside temporary space

```cpp
// ❌ error: runtime UB crossed border or result error
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, true);

// ✅ Option 1: Use Form 1 (Recommended)
AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(dst, src, tmpLocal, srcShape, true);

// ✅ option 2: set aside temporary space (for details, see api-reduce-pattern.md)
```

### Error 6: Reduce dst start address is not 8 bytes aligned

`ReduceMax<float>` / `ReduceSum<float>` API and others require**dst start address 8 byte alignment**(matched with fp32 or 2 elements). Under the "group return" scenario (number of groups per row, with only 4 bytes of results) there is a high risk of an odd number of

```cpp
// ❌ dst uses string 1 fp32: 4 bytes per group.
// Write dstBuf[r *groupsPerRow + g] Only 4B alignment for g odd numbers
const uint32_t groupsPerRow = 4;
AscendC::ReduceMax<float>(dstBuf[r * groupsPerRow + g], src, tmp, 32, false);  // g=1,3 → 4B Alignment
```

Fix: dst buffer uses 2 fp32 (eight bytes per group):

```cpp
// ✅ stride 2 fp32 → each dst results 8 bytes, and any g meets 8B alignment
AscendC::ReduceMax<float>(dstBuf[r * groupsPerRow * 2 + g * 2], src, tmp, 32, false);
```

synchronise index times 2 for downstream reading.

**Symptom**: Reduce API returns silent error (results left with old values or written in the wrong place), not necessarily immediately trip.

---

## best practice

### Parameter Control Table

| Parameter Position | Use rLength | Use rLengthAlign |
|---------|-----------|-----------------|
| DataCopyPad blockLen | ✓ | ✗ |
| Reduce API count | ✓ | ✗ |
| Sub/Exp/Div count | ✓ | ✗ |
| UB rowOffset | ✗ | ✓ |
| Buffer size calculation | ✗ | ✓ |

### Decision-making process

```
Yes. Reduce Operation?
    │
    ├─ Separately (line by line)Softmax/LayerNorm)
    │     └─→ Level 2 Interface
    │           - No Matching Request
    │           - count = rLength
    │
    └─ Cross-line Batch Reduce
          └─→ Pattern Interface (format)1 (Recommended)
                - Yes. 32 Byte Alignment
                - Visible Management tmp buffer
```

### Buffer Allocation

```cpp
uint32_t tileSize = rowsPerLoop * rLengthAlign * sizeof(T);
uint32_t rowBufSize = rLengthAlign * sizeof(T);
uint32_t reduceBufSize = 32 * 1024;

pipe->InitBuffer(inQueueX, 1, tileSize);
pipe->InitBuffer(outQueueY, 1, tileSize);
pipe->InitBuffer(rowBuf, rowBufSize);
pipe->InitBuffer(reduceBuf, reduceBufSize);
```

---

## API Document Access Priority

1. ⭐ ⭐ ⭐**Official API Document**: `asc-devkit/docs/api/context/ReduceMax.md`
2. ⭐ ⭐ ⭐**Official Example Code**: `asc-devkit/examples/03_libraries/05_reduce/`
3. Pattern interface details: [api-reduce-pattern.md](api-reduce-pattern.md)

---

## Example of reference

- Example of `asc-devkit/examples/03_libraries/05_reduce/reducemax/reducemax.asc` - Pattern interface
- `asc-devkit/docs/api/context/ReduceMax.md` - Official API document
