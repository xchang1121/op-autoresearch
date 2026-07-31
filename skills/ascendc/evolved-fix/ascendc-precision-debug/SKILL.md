---
name: ascendc-precision-debug
description: Ascend C operator accuracy debugs skill to provide a diagnosis and solution to the problem of accuracy. Trigger: Output anomalies (all zero, random, uninitialized), accuracy authentication failure (rtol/atol below standard), FP16 accuracy below expectations, post-Cast data error, need to check pipeline Sync (EnQue/DeQue) or DataCopy alignment.
---

# Ascend C operator accuracy debug

## Core concepts

> **accuracy debug = Understand + Analyse + Position + Fix**

1. **data type limit**: FP16 approximately 3-4 bits, FP32 about 6-7 bits.
2. **Identification of numerical stability problems**: large-scale consumption of decimals, catastrophic offsetting.
3. **Mastery of scientific debugging methods**: from minimal recurrence to root cause analysis.

## Time to use

**Applicable**: accuracy authentication failed (rtol/atol did not meet the standard), output was all zero or random, FP16 was above FP32, specified range error was large, pipeline synchronized, DataCopy alignment problem.

---

## Debug preset request ⭐ ⭐ ⭐

> Before entering debugging**must**complete the following three steps

### 1. Fixed Minimum Recoverable Example

| Item | Annotations | Example: |
|------|------|------|
| Shape | tensor shape | `{8, 16}` |
| Dtype | data type | `float16` |
| Fixed value | Specific Values | `[1.0, 2.0, -0.5, ...]` |

**Selection principle**: priority simplistic → priority 32 byte alignment → priority FP32 → over boundary values

**💡 Recommended Practice**: Debugging is suggested to be validated with at least two dtypes (e.g. FP16 and FP32)**the same size + the same data**. If one dtype passes through another failure, it can be rapidly reduced to the corresponding diagnostic model below.

### 2. Search asc-devkit ⭐

> **Intuitive changes to codes are prohibited**

**Retrieving order**
1. Search `asc-devkit/examples/` for similar operator.
2. View `asc-devkit/docs/api/context/` API documents.
3. A comparison between official and current achievements.

### 3. Clear cache and temporary files

```bash
rm -rf build input output
mkdir -p build/input build/output
```

---

## Quick Decision Tree

```
[Precheck] Fixed examples? RetrievedAPI  Cleaned up the cache
    │
    └─ Yes → Let's finish the pre-step.
    └─ Yes. → Go on.
        │
        ├─ [I don't think so.0Step] ⭐ The code is changed and the output is completely unchanged?
        │   ├─ Yes. → Cleaning build/ and kernel_cache Try again after
        │   └─ Yes → Go on.
        │
        ├─ [I don't think so.0.5Step] ⭐ More dtype Cross-validation
        │   ├─ FP32Passed butFP16/BF16Failed → accuracyNot enough. See the diagnosis below.
        │   ├─ BF16Passed butFP16/FP32Failed → API fallback Path difference or accuracyThreshold differences, see below diagnostic model
        │   └─ It's all through./Failed All → Go on.
        │
        ├─ [Step one.] Checking data handling ⭐⭐⭐
        │   ├─ Whether the output is all 0 Or a random error?
        │   │   ├─ Yes. → InspectionpipelineSyncEnQue / DeQue)⭐⭐⭐
        │   │   │       └─ DataCopy And then straight to the calculations?→ Add EnQue/DeQue
        │   │   │       └─ Provisional certification: plus PipeBarrier, confirm the sync problem if correct
        │   │   ├─ Inspection DataCopy Whether or not 32 Byte Alignment
        │   │   │       └─ Inconsistent → Change DataCopyPad
        │   │   └─ Check for use GlobalTensor.SetValue
        │   │           └─ Yes. → Change LocalTensor.SetValue + DataCopyPad Move Out to GM
        │   └─ Validation: Use "CopyIn → CopyOut" Test handling
        │
        ├─ [Step two.] Comparative analysis
        │   └─ Compare official examples with current achievements → Discrepancies detected
        │
        └─ [Step three.] Type of diagnostic problem
            ├─ All the results are bad. → Formula/Constant/APISelection
            ├─ Individual value error → Border conditions/Zero./Spill
            └─ errorThe whole thing is big. → FP16accuracyNot enough. → TryFP32Intermediate calculation
```

---

## Symptoms - causes quick check

| Symptom | Possible causes | Diagnosis |
|------|----------|----------|
| **Output is all 0 or random error** | pipeline Synchronisation Missing / DataCopy Unmatched / GlobalTensor. SetValue | Check EnQue / DeQue, Data Alignment, Change to LocalTensor. SetValue + DataCopyPad ⭐ ⭐ ⭐ |
| `sum = 0, max_err = input level ` | The output is not written. | Check output queue type (VECIN vs VECOUT) |
| `sum=0, max_err≈0` | Output fully 0/ not initialized | Check UB spill, Buffer distribution |
| `Specific parameters range failed ' | Threshold/Boundary Error | Validate threshold calculation, check branch conditions |
| `Unmatched data failed ' | DataCopy Alignment Problem | Change to DataCopyPad |
| {\cHFFFFFF}{\cH00FFFF} {\cHFFFFFF}{\cH00FFFF} {\cHFFFFFF}{\cH00FFFF} {\cHFFFFFF}{\cH00FFFF} {\cHFFFFFF}{\cH00FFFF} | accuracy is inadequate | Intermediate calculation using FP32 |
| `Cast post-data error ' | RundMode Error | half → float for CAST_NONE, float → half for CAST_ROUND |
| **BF16 passed but FP16/FP32 failed** | (1) Part API does not support BF16, BF16 takes a simpler fallback path; (2) BF16 does not have the same characteristics as FP16/FP32 accuracy (BF16 matissa 7bit vs FP16 10bit), accuracy threshold or spill behaviour difference | First check for API fallback branch differences, then check if accuracy threshold values (rtol/atol) fit each dtype |
| **FP32 passed but FP16/BF16 failed** | Half-accuracy calculation of accuracy is insufficient | accuracy: Cast → FP32 Calculates → Cast returns half accuracy |
| **The output remains completely unchanged with the change in code** | Binary not updated / compiler cache | Cleans up bild/ and $HOMEZ1XQ/ Try again |

### Diagnosis mode: "FP32 passed but FP16/BF16 failed."

This is the most common diagnostic signal for accuracy: FP16/BF16 usually shares the same Cast-to-FP32 calculation path in Ascend C (`if constexpr (std:is_same_v<T, bfloat16_t>) || std::is_same_v<T, half>) ', FP32 by stating that the core algorithm is correct, the problem is at the semi-accuracy conversion point.

```
Confirm.: FP32It's through.FP16/BF16Failed
    │
    ├─ Check Intermediate Calculatoraccuracy
    │   ├─ FP16/BF16 Cast→FP32 Is the calculation path correct?
    │   ├─ Is there an outstanding upgrade?accuracy..intermediate (e.g. direct) Add<half>)?
    │   └─ Authentication: will FP16 All calculations in the middle of the path FP32Watch the results.
    │
    ├─ Inspection Cast RoundMode
    │   ├─ half→float Should be used CAST_NONE
    │   ├─ float→half Should be used CAST_ROUND
    │   └─ Authentication: Comparison Cast Values before and after
    │
    ├─ Inspection Pipeline Sync
    │   ├─ FP16/BF16 There's an extra path. Cast Operation,Cast Whether or not after EnQue/DeQue?
    │   ├─ FP32 Path None Cast, there's no such thing as natural sync.
    │   └─ Authentication: In Cast Then add PipeBarrier, confirm the sync problem if correct
    │
    └─ Inspection Buffer Size
        ├─ FP16/BF16 Additional path required Cast buffer(2× innerDim × sizeof(float))
        ├─ FP32 Path not needed Cast buffer
        └─ Authentication: Inspection Tiling in castBuf Size Count
```

### Diagnosis mode: "BF16 passed, but FP16/FP32 failed."

This is a valuable diagnostic signal, which may result from two types of causes:

**Reason 1: API does not support the resulting fallback path difference**. Part Ascend C arithmetic/ reporting API does not support BF16, the developer achieves a simpler fallback path for BF16. BF16 fallback uses the complex API called in the FP16/FP32 path by stating that the logic is correct.

**Reason 2: accuracy threshold / range difference**. BF16 (mantissa 7bit, index range equal to FP32) and FP16 (mantissa 10bit, index range smaller) accuracy characteristics: BF16 is not easy to spill but the tail number accuracy is low, FP16 is easy to spill but the tail number accuracy is high. If the verification threshold (rtol / atol) is not differentiated, or if the spill of FP16 does not occur and BF16 does not exist, then BF16 has passed but failed.

```
Confirm.: BF16It's through.FP16/FP32Failed
    │
    ├─ [Reason1] API fallback Path Difference
    │   ├─ Searching code if constexpr (std::is_same_v<T, bfloat16_t>) Branch
    │   ├─ BF16 Let's go. fallback Path vs FP16/FP32 Where's the difference in the main path?
    │   ├─ List FP16 / FP32 Use in path but BF16 Path not used API
    │   ├─ Verify differences on a case-by-case basis API Parameters (%2)mask,repeatTime,stride)
    │   └─ Temporary General FP16 The path should read BF16 Same. fallback Achieved and observed through
    │
    ├─ [Reason2] accuracythreshold / Difference in range of values
    │   ├─ FP16 Is it spilling?FP16 max ≈ 65504,BF16 Index range equal FP32)
    │   ├─ Validate threshold (%2)rtol/atol) Whether to press dtype Distinction?
    │   │   ├─ BF16: rtol=1e-2 Level (%1)mantissa only 7bit)
    │   │   └─ FP16: rtol=1e-3 Level (%1)mantissa 10bit)
    │   └─ Check if the intermediate calculation is due FP16 HigheraccuracyThe request exposed the algorithm's flaws.
    │
    └─ Cross-validation
        ├─ Inspection asc-devkit Document confirmation relevant API Supported or not BF16
        ├─ If... API Not supported BF16 → Prioritize by cause1Check it out.
        └─ If... API Support BF16(BF16/FP16 Go the same path)→ Prioritize by cause2Check it out.
```

---

## Common trap-search.

| A trap. | Symptom | Solutions |
|-----|------|----------|
| **pipeline sync is missing** | Output all 0 or random error | DataCopy must be followed by EnQue/ DeQue Sync ⭐ ⭐ ⭐ |
| **DataCopy Unmatched** | Small-scale data full of 0/Abnormal | Use DataCopyPad ⭐ ⭐ ⭐ |
| **GlobalTensor.SetValue** | All output is 0 | Move to GM ⭐ ⭐ ⭐ with LocalTensor. SetValue + DataCopyPad |
| **Cast RoundMode** | Post-Cast data mess. | half→float for CAST_NONE, float→half for CAST_ROUND ⭐ |
| FP16 accuracy is inadequate | Simple calculations have error. | Critical middle value FP32 |
| Exp/ log spill | Show Inf or NN | Minus the maximum and then calculate. |
| Subtract offset | When a ≈b is a-b error big | Use a numerically stable equivalent formula |
| Reduce error | Reduce, the result is bigger than the element by element, error. | Use FP32 loader |
| Zero risk. | NN or abnormally large | Add epsilon protection |

### pipeline Synchronization Debug

**Core issue**: DataCopy / DataCopyPad is an anecdotal DMA operation, doing Vector calculations directly on moving data and possibly reading incomplete data!

```cpp
// ❌ error: direct after AllocTensor
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
Compute(x);  // Wrong. Probably read the data on the uncompleted removal.

// ✅ Correct: DeQue to calculate later
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();  // Waiting for removal to complete
Compute(xIn);
```

**Provisional debugging method**:
```cpp
DataCopy(x, gm, size);
PipeBarrier<PIPE_ALL>();  // Temporary plus, if the result correctly indicates the problem of synchronization
Compute(x);
```

**If Pipe Barrier can solve the problem, it means synchronization.**→ Rehabilitation Program: change to EnQue/ DeQue Mechanism

| Error | Get it right. |
|-----|---------|
| Data is available after AllocTensor | AllocTensor only assigns memory without waiting for removal. |
| DataCopy. It's synchronized. | DataCopy is a step away. DMA, return immediately. |
| No, EnQue/DeQue can work. | Must sync with EnQue/ DeQue or PipeBarrier |
| Pipe Barrier, good performance. | Pipe Barrier, full pipeline pause, performance poor. |

For further details, see[references/common-traps.md](references/common-traps.md)

---

## Debug Policy Level

```
Debug Method
    │
    ├─ A fast-track approach (prior to trying),≤7(a) Number of reports
    │   ├─ errorDistribution analysis → IdentificationerrorMode
    │   ├─ Printf Organisation → Zoom Out
    │   ├─ DumpTensor 7Step → kernel Interpolation stake + CPU golden Paragraph-by-paragraph comparison ⭐
    │   └─ Common trap screening. → It's a drug.
    │
    └─ Secondary debugging (feasibility)
        └─ Quick Method Attempt≥7Switch as soon as the second or method is exhausted
```

> **Key principle**: do not try blindly more than 7 times

### DumpTensor 7 Steps

kernel interpolation debugging standard tool: insert `DumpTensor` at the CopyIn / Compute / CopyOut key point, matching CPU gold with the paragraph-by-paragraph desc number (100 /200/300) and fast-tracking anomalies at which stage of the data stream. Application: Output error, NaN/Inf, need to track the data at all stages of the CopyIn → Comte → CopyOut.

For further details, see[references/ascendc-dumptensor.md](references/ascendc-dumptensor.md)

---

## Problem positioning methods

### 1. Contrast method (comparison to working code)

Finds the normal working code, compares the differences by line

### 2. Boundary dichotomy

Record critical points of passage/failure, analyze branch selection

### 3. Numerical Authentication Method

Do not believe in the formula, use the code to calculate the actual value.

### 4. Buffer Debug Elements

| Problem | Performance | Solutions |
|------|------|----------|
| VECIN for output | Output equals input | The output must be queued with VECOUT |
| Double Buffer | Threshold error | ×2 when calculating thresholds |

See the detailed location process at [references/diagnosis-workflow.md] (references/diagnosis-workflow.md)

---

## accuracy standard source priority

1. **Priority 1**: well-defined accuracy requirements in operator Development Plan
2. **Priority 2**: China for official accuracy standard document
3. **Priority 3**: This Skill Default (Faceground only)

| data type | rtol | atol |
|---------|------|------|
| FP16 | 1e-3 | 1e-4 |
| FP32 | 1e-5 | 1e-6 |
| INT | - | 0 |

---

## Agent User Guide

### Debug count rules

```
Counter = 0
Every time you try a fast-track method,errorAnalysis/Printf/Tighten up the trap.→ Counter+1
Time counter >= 7 Or the fast-track approach is exhausted. → Toggle Debugging Now
```

> **Empirical proposal for 💡**: if repeated attempts do not make progress (the number of failed examples does not decrease), it is recommended that:
> 1. Check if compiled caches (`rm -rf build/ $HOME/atc_data/kernel_cache/`) have been cleared
> 2. Verify if modified binary sh256 really changes
> 3. Switch to a completely different debug strategy (e.g., binary downgrade to minimum working path)

### Debugging summary requirements

### Checklist

**Debug phase**:
- [ ] Fixed Minimum Recoverable Example
- [ ] Retrieval asc-devkit confirmed API usage ⭐
- [ ] Cleared caches and temporary files
- [ ] **Pipeline synchronization issue** (`DataCopy` ordering around `EnQue`/`DeQue`) ⭐⭐⭐
- [ ] **Zero-initialization issue** (`DataCopy` alignment or `GlobalTensor.SetValue` → `LocalTensor.SetValue` + `DataCopyPad`) ⭐⭐⭐
- [ ] Compare official examples with current achievements
- [ ] Number of attempts < 7
- [ ] Toggle diagonal debugging immediately to the threshold

---

## References

### Workflow
- [diagnosis-workflow.md] (references/diagnosis-workflow.md) - Full diagnostic workflow
- [binary-search-debug.md] (references/binary-search-debug.md) - Debug Detailed Guide

### Problem diagnosis
- [common-traps.md] (references/common-traps.md) - Common accuracy trap
- [best-practices.md] (references/best-practices.md) - best practice

### Debug Tool
- [printf-debug.md] (references/printf-debug.md) - Printf debugging
- [data-comparison.md] (references/data-comparison.md) - Data Contrast
- [tools-reference.md] (references/tools-reference.md) - Tool and Command Reference
- [ascendc-dumptensor.md] (references/ascendc-dumptensor.md) - DumpTensor 7-step method (includes API, error mode)

### Field cases
- [case-studies.md] (references/case-studies.md) - Field debugging case
