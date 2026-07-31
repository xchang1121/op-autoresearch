---
name: ascendc-api-best-practices
description: Ascend C API uses best practice and Blacklists. This covers the correct use of core APIs such as arithmetic / attribution / data handling / Buffer / accuracy conversion / pipeline Sync / Transpose. Trigger: operator has encountered uncertainty about API usage, API parameter error reporting (matching / repeattimes), needing to search for blacklists and limitations.
---

# Ascend C API best practice

## API Category Index

| API Category | Coverage API | Documentation | Typical scene. |
|---|---|---|---|
| **Calculation** | Add, Sub, Mul, Div, Adds, Muls | [api-arithmetic.md](references/api-arithmetic.md) | Softmax, Layer Norm, Radio |
| **Convention of return** | ReduceMax, ReduceSum, WholeReduce*, BlockReduce* | [api-reduce.md](references/api-reduce.md), [api-reduce-pattern.md](references/api-reduce-pattern.md) | Softmax, LayerNorm, ReduceMean |
| **Data removal** | DataCopy, DataCopyPad | [api-datacopy.md](references/api-datacopy.md) | Non-regulated, multi-dimensional handling |
| **Transpose / Gather** | TransDataTo5HD, Gather | [api-transpose.md](references/api-transpose.md) | Traspose, gather/permute |
| **Buffer Management** | TBuf, TQue | [api-buffer.md](references/api-buffer.md) | DoubleBuffer, UB Planning |
| **accuracy conversion** | Cast | [api-precision.md](references/api-precision.md) | FP16 /BF16/FP32 Mixed accuracy |
| **pipeline Synchronization** | EnQue, DeQue, SetFlag, WaitFlag | [api-pipeline.md](references/api-pipeline.md) | Multilevel pipeline, Event Synchronization |
| **repeatTimes Limit** | All vector inrinsic | [api-repeat-limits.md](references/api-repeat-limits.md) | repeatTimes ≤ 255, batch processing |
| **API Restrictions / Alignment** | Compare, wait. | [api-restrictions.md](references/api-restrictions.md) | 256B Alignment Containment, Disable API |
| **Host Runtime** | aclrtSetDevice, aclrtGetDeviceInfo | [api-host-runtime.md](references/api-host-runtime.md) | device Initialization, Numerical Query |
| **Rapid reference** | All above-mentioned API parameter speedsheets | [api-quickref.md](references/api-quickref.md) | — |

## Typical scene index

| scene | Association Documents | Key skills |
|---|---|---|
| **Softmax / LayerNorm** | api-reduce, api-reduce-pattern, api-arithmetic | Reduce and broadcast back to vector, Adds/ Muls for Div |
| **Line-by-line (AR template)** | api-arithmetic | Adds/Muls, UB Savings |
| **Multi-broadcast (ARA template)** | api-arithmetic | `BinaryRepeatParams.src1RepStride=0`, batch |
| **Semi-accuracy plus minus (FP16/BF16)** | api-arithmetic, api-precision | Default to raise accuracy FP32 unless spec says equal weight |
| **Unmatched data** | api-datacopy | DataCopyPad, avoid DataCopy crashing 32B |
| **Mixed accuracy** | api-precision | FP16 input +FP32 calculation |
| **pipeline Optimization** | api-pipeline, api-buffer | DoubleBuffer + EnQue/ DeQue pair |

## ⛔ API Blacklist (absolutely prohibited)

| API | Prohibited grounds | Alternatives |
|---|---|---|
| `GlobalTensor::SetValue()` | Extremely inefficient (uniform DMA) | `DataCopyPad` |
| `GlobalTensor::GetValue()` | Ibid. | `DataCopyPad` |
| `DataCopy(GM↔UB)` Non-32B Alignment | Cross-border visits led to UB OOB | `DataCopyPad` |

Debugging using: `AscendC::printf` single point validation (debug construction only, production remove).

## Performance backtracking.

These writings are usually run-off, but they can be seen in profiling as Scalar sound, access debris or Vector command expansion:

| Inverse Mode | Typical performance | Replace Priority |
|---|---|---|
| GM per element `GetValue/SetValue` | Garther/scatter/resize Extremely Slow | UB Stagging + Continuous `DataCopy` |
| One small `CopyOut` per line | Small statutes, multiple outputs, operator lanch, still slow. | Batch back after UB saves multiple lines |
| fp32 Enter still doing `Adds(x, 0)` | VEC high rate of activity add/cast | Directly use input LocalTensor as a calculation source |
| fp16/bf16 Simple operator Forced return to/from fp32 | Cast. It's mostly time-consuming. | accuracy allowed and native dtype path |
| Repeat for each file `div/mod` | Scalar command is dense | host tilingProjectedstride/base/mode |
| `SetFlag/WaitFlag` without dependence | Victor and Scalar wait a lot | `PipeBarrier` or delete redundant sync |
| All of them are gone. | Small, slow, too. | same-shape/scalar/last-dimFast Path |
| DataCopyPad Overwrite Main Path | MTE command fragmentation | Main path is `DataCopy`, tail |

Example: Replace scalar GM reading and writing with continuous block handling.

```cpp
// Discrepancies: A GM scalar visit for each element.
for (int32_t i = 0; i < len; ++i) {
    auto v = xGm.GetValue(base + i);
    yGm.SetValue(outBase + i, v);
}

// Okay: Move to UB and write back in bulk.
auto local = buf.Get<T>();
DataCopy(local, xGm[base], len);
DataCopy(yGm[outBase], local, len);
```

Example: aligned main path separated from tail pad.

```cpp
int32_t mainLen = len / elemsPer32B * elemsPer32B;
if (mainLen > 0) {
    DataCopy(local, gm[offset], mainLen);
}
if (mainLen < len) {
    DataCopyPad(local[mainLen], gm[offset + mainLen], len - mainLen, padParams);
}
```
