# AscendC DumpTensor Debugging

Systematic approach to debug AscendC kernels using DumpTensor API.

## Quick Start

```cpp
// Add DumpTensor after key computation points
DumpTensor(inputLocal, 100, 32);   // After CopyIn
DumpTensor(tmpLocal, 200, 32);     // After computation
DumpTensor(outputLocal, 300, 32); // Before CopyOut
```

## 7-Step Debugging Workflow

### 1. Add DumpTensor at Key Points

Insert after: CopyIn, each Compute step, before CopyOut.

```cpp
// After DataCopy
LocalTensor<T> inputLocal = inQueue.DeQue<T>();
DumpTensor(inputLocal, 100, 32);

// After computation
Adds(tmpLocal, inputLocal, 1.0f, tileLength);
DumpTensor(tmpLocal, 200, 32);

// Before CopyOut
DumpTensor(outputLocal, 300, 32);
```

### 2. Use Systematic Desc Numbering

| Range   | Stage         |
|---------|---------------|
| 100-199 | Inputs        |
| 200-299 | Intermediates |
| 300-399 | Outputs       |

Increment by 10 within each range.

### 3. Add CPU Golden Prints

Match NPU dump points with CPU prints:

```cpp
// CPU reference
printf("[CPU-100] input: %.6f, %.6f\n", input[0], input[1]);
printf("[CPU-200] tmp: %.6f, %.6f\n", tmp[0], tmp[1]);
printf("[CPU-300] output: %.6f, %.6f\n", output[0], output[1]);
```

### 4. Verify Input Data First

Always confirm inputs are correct before debugging computation.

Checklist:

- Input values match CPU golden
- No NaN/Inf in inputs
- Data alignment correct
- Shape/size match expectations

### 5. Segment Verification

Verify each stage independently:

1. **CopyIn** → If wrong, fix DataCopy/DataCopyPad
2. **Compute** → If wrong, debug computation logic
3. **CopyOut** → If wrong, fix output stage

### 6. Analyze Error Patterns

See [ascendc-dumptensor-refs/error-patterns.md](ascendc-dumptensor-refs/error-patterns.md) for common patterns and root causes.

### 7. Apply Fix and Verify

Re-run with DumpTensor to confirm fix works.

## References

- [API Reference](ascendc-dumptensor-refs/api-reference.md) - DumpTensor API details and best practices
- [Error Patterns](ascendc-dumptensor-refs/error-patterns.md) - Common error patterns and root causes

---

## Use trap ⭐ ⭐ ⭐

DumpTensor itself is bound by the AscendC pipeline rules and misreads the diagnosis by misreading the wrong data.

### 1. Must be after DeQue Dump, not after AllocTensor

```cpp
// ❌ error: Dump before removal, read last residual / uninitialized
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
DumpTensor(x, 100, 32);   // It's not what you read. gm Value
inQueue.EnQue(x);

// ✅ Correct: DeQue until removal is completed
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();
DumpTensor(xIn, 100, 32);  // The data is ready by now.
```

If it is not convenient to add EnQue/DeQue (e.g. temporary plugs), use `PipeBarrier<PIPE_ALL>()` as a back-up - to determine whether or not to supplement the sync after confirming the correct results.

### 2. Multinuclear scenes must include blockIdx in desc

All cored dumps will be written to the same log at the time of multi-checking, and no blockIdx number will result in a completely indistinguishable result.

```cpp
// ❌ error: read only desc cannot tell which core
DumpTensor(inputLocal, 100, 32);

// ✅ Correct: add blockIdx to desc
uint32_t desc = 100 + GetBlockIdx() * 1000;   // core 0: 100, core 1: 1100, ...
DumpTensor(inputLocal, desc, 32);
```

When debugging a single nuclear issue, use `if (GetBlockIdx() == 0)` to limit dump to nuclei 0 and avoid log explosions.

### 3. dumpSize Control

- Default 32 Element is sufficient for diagnostic mode (NAN/ All / Offset can be judged by the first few values)
- Big tensor complete dump fills the log buffer in an instant and affects Kernel time series, covering up the original bug
- `dumpSize` Do not exceed the actual length of tensor or cross the border

```cpp
uint32_t dumpSize = std::min(tileLength, 32u);
DumpTensor(outputLocal, 300, dumpSize);
```

### 4. Debug complete must be removed

DumpTensor introduces significant time-sequencing costs that may change the behaviour of pipeline, which must be removed or macro-switched after positioning:

```cpp
#ifdef DEBUG_DUMP
DumpTensor(outputLocal, 300, 32);
#endif
```

---

## Output Reader

Output form is the following (the actual format is based on CANN version):
```
[DumpTensor] block_idx=0 desc=100 size=32 dtype=float32
  0.123, 0.456, -0.789, ...
```

Operational process:
```bash
# Run the example.
./run_op > dump.log 2>&1

# Press desc to extract a phase
grep "desc=100" dump.log    # Input
grep "desc=200" dump.log    # Centre
grep "desc=300" dump.log    # Output

# Multinuclear scenario separated by nuclear.
grep "block_idx=0" dump.log
```

Align NPU output with CPU gold and desc number and compare it to the fast-positioned anomaly.

---

## Debug Method Selection

DumpTensor is not the only one.

| Methodology           | scene                              | Strengths                  | Limits                    |
|----------------|-----------------------------------|----------------------|-------------------------|
| **DumpTensor** | NPU mode, read LocalTensor data     | Look directly at the UB real value     | Time series is expensive and requires pipeline synchronization |
| `PRINTF`       | NPU mode, watch scalar/ control stream      | Light                 | Can not get folder: %s: %s |
| `printf`       | CPU Simulation Mode                      | Consistent with normal C++ debugging   | Can not verify NPU actual behaviour    |
| Debug 2       | We know there's a bug, but it's not working.         | I'm sure it'll hold back.           | Slow, many times compiled and run       |

The "debug strategy level" that connects the father skill:
- Start with the DumpTensor 7-step method (≤7 tried) and debug it immediately if position fails
- DumpTensor, after seeing the data anomaly, drugged him with the skill "Symptomology-Probative Spacing" test.
- Remember `build/` and `$HOME/atc_data/kernel_cache/` after changing the code, otherwise the dump will be wrongly assumed not to work.

---

## Checklist

Before the plug:
- [ ] Plugged after DeQue, not after AllocTensor
- [ ] Multi-nuclear scenario added blockIdx to desc
- [ ] dumpSize ≤ 32 (unless it is confirmed that more is required)
- [ ] CPU golden has been output with the same desc
- [ ] Compile cache cleared

After positioning:
- [ ] The earliest stage in which an anomaly has been identified (CopyIn / Compute / Copyout)
- [ ] Matched with [error-patterns.md] (ascendc-dumptensor-refs/error-patterns.md)
- [ ] Post-recovered dump validation passed, removed/maxis off DumpTensor
