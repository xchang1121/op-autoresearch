# API Quick Reference

Ascend C API uses the Core Decision Index.

---

## Contents

1. [Core Principles expedited](#Core Principles expedited)
2. [decision tree: What am I supposed to use?](# decide what api should I use)
3. [See detailed documents by scene](#see detailed documents by scene)

---

## Core principles fast track

### 1. DataCopy vs DataCopyPad

**Principle: priority for DataCopyPad**

| scene | API | Reason |
|-----|-----|------|
| All GM ↔ UB handles | `DataCopyPad` | Harmonized alignment/non-matching |
| Determines that data is strict 32 byte alignment | `DataCopy` | Simple scene available |

**Detailed use**:[api-datacopy.md](api-datacopy.md)

### 2. Cast RoundMode Selection

| Convert direction | RoundMode | Reason |
|---------|-----------|------|
| half → float | `CAST_NONE` | Low → High accuracy, no loss |
| float → half | `CAST_ROUND` | High → Low accuracy to round |
| half → int32_t | `CAST_ROUND` | Quantified scene |
| int32_t → float | `CAST_NONE` | Integer → Floating Point |

**Detailed use**:[api-precision.md](api-precision.md)

### 3. TBuf vs TQue Selection

| scene | Type | Annotations |
|------|------|------|
| MTE2/MTE3 Move buffer zone | `TQue<VECIN/VECOUT>` | `InitBuffer(que, num, size)` |
| Pure Victor calculates the buffer zone | `TBuf<VECCALC>` | `InitBuffer(buf, size)` |

**Detailed use**:[api-buffer.md](api-buffer.md)

### 4. pipeline Sync

**Principle**: EnQue/ DeQue must synchronize MTE and Victor

**Core model**:
```
CopyIn → EnQue → DeQue → Compute → EnQue → DeQue → CopyOut
```

**Detailed use**:[api-pipeline.md](api-pipeline.md)

### 5. Victor API limit

**Core limit**: maximum value when repeattime is uint8_t 255

**Process**: Host side limit R_max or Kernel side in batch processing

**Detailed use**:[api-repeat-limits.md](api-repeat-limits.md)

### 6. Reduce API Selection

| scene | Interface | Annotations |
|-----|------|------|
| Independent on a line-by-line basis | Level 2: `ReduceMax(dst, src, tmp, count)` | No matches, count pass rLength |
| Cross-line BatchReduce | Pattern: `ReduceMax<T, Pattern::AR>(...)` | 32 byte alignment required |

**Detailed use**:[api-reduce.md](api-reduce.md) | [PatternInterface Details](api-reduce-pattern.md)

---

## Decision tree: What am I supposed to use?

### Q1: Need GM ↔ UB data?

```
Yes. → DataCopyPad(Recommended)
   → DataCopy(when only certain)32Byte Alignment)

Yes → Go on.
```

### Q2: Need accuracy conversion?

```
half → float → CAST_NONE
float → half → CAST_ROUND
Other → Access api-precision.md
```

### Q3: Need to allocate the UB buffer zone?

```
Involving MTE Removal → TQue + InitBuffer(que, num, size)
Pure Vector Calculate → TBuf + InitBuffer(buf, size)
```

### Q4: Data error/random value encountered?

```
1. Check if missing EnQue/DeQue → api-pipeline.md
2. Inspection DataCopyPad Parameters → api-datacopy.md
3. Inspection Reduce API of tmpBuffer Type → api-reduce.md
 4. Check for multi-line processing rowOffset Calculate → api-reduce.md
 5. Inspection repeatTime Whether to overflow → api-repeat-limits.md
```

### Q5: Need to mix accuracy calculations (FP16 input, FP32 intermediate calculation)?

```
Access api-precision.md MixingaccuracyMode
```

---

## View detailed documents by scene

| scene | Documentation | Core content |
|-----|------|---------|
| Data handling (GM↔UB) | [api-datacopy.md](api-datacopy.md) | DataCopyPad Parameters, Strude Calculating, Unmatched |
| accuracy Conversion/Mixing accuracy | [api-precision.md](api-precision.md) | Cast RoundMode, FP16 Mixed accuracy Mode |
| UB buffer zone management | [api-buffer.md](api-buffer.md) | TBuf/TQue Selection, Double Buffer, Batch Removal |
| pipeline Sync | [api-pipeline.md](api-pipeline.md) | EnQue/DeQue Synchronization Mechanism, Scheduling |
| repeatTime limit | [api-repeat-limits.md](api-repeat-limits.md) | Batch processing |
