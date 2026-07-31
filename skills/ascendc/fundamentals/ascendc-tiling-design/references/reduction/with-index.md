# Reducation with Index Tracking

> Index tracking is an**-transformation**of Reduction: the location of the extreme (or other condition) is recorded at the same time as the return.
>
> **Tiling methodology is identical**: 3D abstraction, AR/ARA determination, full/segregated determination, multi-nucleotide — all replicating standard Retract branch documents ([ar-fullload.md] (ar-fullload.md), [ara-fullload.md] (ara-fullload.md), etc.].
>
> **This document only describes the incremental differences tracked by the index**: API replacement, additional constraints, Buffer increments.

---

## Summary of Differences to Standard

| Branch | Standard | Index tracking variants |
|--------|---------------|-------------|
| AR-FullLoad | `ReduceMax(dst, src, tmp, count)` | `ReduceMax(dst, src, tmp, count, calIndex=true)` |
| AR-ColSplit | chunk → Reduce Max → scalar merger | cunk → `ArgMaxV1` → cross-index offset |
| ARA-FullLoad | `Pattern::Reduce::RA` → vector. | `Compare(LE) + Select(TT) + Select(TS)` Line-by-line iterative |
| ARA-RowSplit | Same ARA-FL + cross chunk `Max` | Merge with ARA-FL+ Cross chunk Company+Select |

**Additional constraints**(all branches share):

| Constraints | Annotations |
|------|------|
| **Index type** | The index must be stored in float and, finally, Cast is in32 |
| **Index scope** | float32Exactly[0, 2^24]Integer number;half <= 65535 |
| **Multiple polar values** | returns**the first**polar index (LE/GE program automatically guaranteed) |
| **Compare 256 Bytes Alignment** | Count elements for 256 bytes (fload: 64 element multiples),**non**32 bytes |
| **DataCopyPad rightPadding** | rightPadding < = 32 bytes (maximum 8 floats), large padding volume cannot use this parameter |
| **Select 8K reserved** | DAV_2201 Mode 1/2 has to set aside 8K UB, framework, which is usually managed automatically |

---

## Index variant for the AR branch

> Tiling Parameter Calculation, Multinuclear Cuts, DataCopyPad Configuration: See [ar-fullload.md] (ar-fullload.md)/[ar-colsplit.md] (ar-colsplit.md).
> The following is only a description of the API differences that the index tracks under the AR branch.

### AR-FullLoad:ReduceMax(calIndex=true)

Data continuum, using the `calIndex=true` parameter for Level 2 API:

```cpp
AscendC::LocalTensor<float> dstVal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> sharedTmpBuffer = tmpQueue.AllocTensor<float>();

AscendC::ReduceMax<float>(dstVal, srcVal, sharedTmpBuffer, count, true);
// CalIndex=true at the same time return value and index

// Output format: dst[0] = maximum, dst [1] = index
float maxVal = dstVal.GetValue(0);
float idxRaw = dstVal.GetValue(1);
uint32_t maxIdx = *reinterpret_cast<uint32_t*>(&idxRaw);  // Type conversion!
```

**tmpBuffer calculates**(calIndex=true larger):

```cpp
uint32_t tmpSize = AscendC::GetReduceMaxMinTmpSize<T>(count, true);
```

**Buffer Planning**(vs standard AR-FullLoad variance):

| Buffer | Size | Purpose | Standard Reaction |
|--------|------|------|----------------------|
| srcQueue | count×sizeof(T) | Enter Data | Same |
| dstQueue | **2×sizeof(T)** | Output value + index | Standard only 1×sizeof(T) |
| tmpQueue | tmpBufSize×sizeof(T) | Intermediate calculation | Same, but bigger when calIndex=tru |

### AR-ColSpit: ArgMaxV1 + Cross-Script Index Merge

  When  RIt's too big to load it all.chunkProcess. Every piece used`ArgMaxV1`Independent for local max+Index, cross-film.

```cpp
// ArgMaxV1: Find maximum value and subscript for continuous R_slice elements
// dst_indice: Output index (0-based, relative to the beginning of the film)
// dst_values: output maximum
ArgMaxV1(dst_indice, dst_values, src, batchSize, R_slice);
```

**Cross-section consolidation logic**:

1. **First**: ArgMaxV1 → Initial (maxValue, maxIndex)
2. **Follow-up to each film**: Arg MaxV1 → (chunkValue, chunkIndex)
   - chunkValue > maxValue → Update maxValue, maxIndex = chunkIndex +**Initial offset**
   - Otherwise retain the original value
3. **tail**: probably smaller than cutRSize, same treatment

> **Key**: The index returned by ArgMaxV1 is**internal deviation**(starting with 0) with the global starting position of the piece to be added to the merger.

> **ArgMin variance**: for comparison condition from `>` to `<`.

> **ARA-RowSpit scenario**: the division merges with the same logic, with the difference being only in data handling (DataCopyPad's blockCount=R_chunk row, with srcStridide).

---

## Index variant of the ARA branch

> Tiling Parameter Calculation, Multinuclear Cuts, DataCopyPad Configuration: See [ara-fullload.md] (ara-fullload.md)/[ara-rowsplit.md] (ara-rowsplit.md).
> The following is only a description of the index tracking API differences under the ARA branch.

### Core difference: Compare+Select Replaces Pattern::Reduce::RA

Standard ARA Reduction returns the entire `(R, alignedCols)` matrix at a time with `Pattern::Reduce::RA`.
Index tracking cannot be done using Patterson API (no index output), moving to line-by-line `Compare+Select` iterative.

### API binding (DAV_2201)

| Binding Item | Specific limitations | Impact |
|--------|---------|------|
| **Select dst type** | DAV_2201 only supports half/float,**not in 32** | Index must be stored in float, before output Cast is in 32 |
| **Compare count alignment** | The space occupied by the count elements must**256 byte alignment**(float: 64 element multiples) | a0Aligned =ceil (A0/64)*64, not 32 byte alignment |

### Recommended scheme: LE Invert + TENSOR_SCALAR

**Core technique**: Compare reverses mask polarity with LE (and not GT) so that bit=1 means "retains the old value", so `VSEL_TENSOR_SCALAR_MODE` uses XZ0XQ as scalar to import the current line index into Select and saves each round of `Duplicate` operations. Three commands/wheels, five buffers in the cycle.

**algorithmic logic**

```
Compare(LE): xLocal[r] <= maxLocal
  → bit=1: Current line does not exceed the previous maximum → Keep old value
  → bit=0: Current rows greater than the old max   → Update new value

Select(maxLocal, cmpLocal, maxLocal, xLocal[r], TENSOR_TENSOR):
  → bit=1: Reservations maxLocal(Maximum old)
  → bit=0: Remove xLocal[r](New maximum)

Select(idxLocal, cmpLocal, idxLocal, rowIdxFloat, TENSOR_SCALAR):
  → bit=1: Reservations idxLocal(old index) tensor)
  → bit=0: Remove rowIdxFloat(New Index) scalar)
```

**"First extreme" semantic guarantee**: LE set up (bit=1) when `xLocal[r] == maxLocal` is in place, keeping the old index - consistent with numpy.argmax behaviour.

**reference implementation**

```cpp
__aicore__ inline void Compute()
{
    AscendC::LocalTensor<float>   xLocal   = inQueueX.DeQue<float>();
    AscendC::LocalTensor<int32_t> yLocal   = outQueueY.AllocTensor<int32_t>();
    AscendC::LocalTensor<float>   maxLocal = maxBuf.Get<float>();
    AscendC::LocalTensor<float>   idxLocal = idxBuf.Get<float>();      // Indexing float Storage!
    AscendC::LocalTensor<uint8_t> cmpLocal = cmpBuf.Get<uint8_t>();

    // Initialization: First row as initial maximum, indexed to 0.0f
    AscendC::DataCopy(maxLocal, xLocal, a0Aligned);
    AscendC::Duplicate<float>(idxLocal, 0.0f, a0Aligned);
    AscendC::PipeBarrier<PIPE_ALL>();  // DataCopy(MTE) + Duplicate(V) Cross pipe

    // LE Invert + TENSOR_SCALAR optimize cycle
    float rowIdxFloat = 1.0f;  // Use it. float Thruster Avoidance aicore in uint→float cast
    for (uint32_t r = 1; r < R; r++) {
        AscendC::Compare(cmpLocal, xLocal[r * a0Aligned], maxLocal,
                         AscendC::CMPMODE::LE, a0Aligned);
        AscendC::Select(maxLocal, cmpLocal, maxLocal, xLocal[r * a0Aligned],
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, a0Aligned);
        AscendC::Select(idxLocal, cmpLocal, idxLocal, rowIdxFloat,
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, a0Aligned);
        rowIdxFloat = rowIdxFloat + 1.0f;
    }

    // fload index → in32 output
    AscendC::Cast(yLocal, idxLocal, AscendC::RoundMode::CAST_ROUND, a0Aligned);
    outQueueY.EnQue<int32_t>(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

### Buffer Planning (3 of 5 vsReducation)

| Buffer | Type | Size | Purpose | Queue | Standard |
|--------|------|------|------|-------|---------------|
| inQueueX | float | R×a0Aligned×4 | Enter Data | TQue VECIN, InitBuffer num=2 | Same |
| outQueueY | int32 | a0Aligned×4 | Output Index | TQue VECOUT, InitBuffer num=2 | Output value (non-indexed) |
| maxBuf | float | a0Aligned×4 | Current max | TBuf VECCALC | **New** |
| idxBuf | float | a0Aligned×4 | Current Index (float storage) | TBuf VECCALC | **New** |
| cmpBuf | uint8_t | max(a0Aligned/8, 32) | Compare Results Mask | TBuf VECCALC | **New** |

### UB Calculation

```
UB_USED = 2 × (R × a0Aligned × 4)       // inQueueX (num=2 Open Double Buffer)
        + 2 × (a0Aligned × 4)           // outQueueY (num=2 Open Double Buffer)
        + a0Aligned × 4                  // maxBuf
        + a0Aligned × 4                  // idxBuf
        + max(a0Aligned / 8, 32)         // cmpBuf(32 Byte Alignment)
```

### Compare 256 Byte Alignment Policy

**Alignment calculation**(not 32 bytes):

```cpp
// fload type: multiplies up to 64 (256 bytes / 4 bytes = 64 elements)
uint32_t a0Aligned = ((A0 + 63) / 64) * 64;
```

| Original A0 | a0Aligned | Pad Volume |
|---------|-----------|--------|
| 32 | 64 | 32 |
| 36 | 64 | 28 |
| 64 | 64 | 0 |
| 65 | 128 | 63 |

**Pad area does not need prefilling**: CopyOut will only export the curA0Len active elements, and the pad area results are not written back to GM. Save prefilling reduces one wide vector writing operation and one `PipeBarrier<PIPE_ALL>()`.

---

## Min-Index variant

The only difference with Max-Index is the Compare mode reverse:

| | Max-Index | Min-Index |
|--|--------|--------|
| Compare Mode | **LE** | **GE** |

The remaining logic (Select, Buffer, alignment, index type) is identical.

---

## Performance optimization techniques

### Skills 1: LE/GE Invert + TENSOR_SCALAR mode

**Rationale**: By inverseComparea comparison of the direction of theGT→LE / LT→GEIt's not a good idea.mask in bit=1Organisation"Keep old value"(Most locations),bit=0Organisation"Update new value"(minus position). Index updatedSelectUse it.`VSEL_TENSOR_SCALAR_MODE`:
- Bit=1 → takes value from tensor (idxLocal) (retains old index)
- Bit=0 → takes value from scalar (rowIdxFloat) (update new index)

**Proceeds**: Province Duplicate + 1 buffer + 3 Directives/ Wheels

**Conditions applicable**: Line-by-line scenario of ARA branch that requires Selact's TENSOR_SCALAR model to support float.

### Skills 2: float loader replacement Cast

`static_cast<float>(uint32_t)` is not supported in aicore. Gradient +1.0f with fload variable:

```cpp
float rowIdxFloat = 1.0f;
for (uint32_t r = 1; r < R; r++) {
    // Use rowIdxFloat instead of status_cast<float>(r)
    ...
    rowIdxFloat = rowIdxFloat + 1.0f;
}
```

---
