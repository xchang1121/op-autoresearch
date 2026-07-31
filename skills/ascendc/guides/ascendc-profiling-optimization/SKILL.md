---
name: ascendc-profiling-optimization
description: "AscendC programing to optimise action: VEC/MTE/CUBE/SCALAR sound, Bank condition, double buffer, L2 Cache, and inter-nuclear load imbalance. This applies to a scenario where operator is already correct but not performing adequately."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC profiling and Optimization

Use this skill after operator has passed correctness verification. Do not rewrite when there are still accuracy errors, crashes, or ABI problems; use `ascendc-precision-debug` or `ascendc-crash-debug` to compress to a verifiable version.

## 1. Profiling signal to optimize action

| Main signal | Common bottlenecks | Priority Action |
|---|---|---|
| Victor's share is high. | vector calculation limit | Merge UB calculation phase, reduce Cast, replace high-cost vector sequence |
| MTE2 High Time | GM to UB movement restricted | Increase single handling particle size, check 32B alignment, enable double buffer |
| MTE3 High Time | UB to GM write back restricted | Reduce the number of intermediate returns and write the final results as soon as possible. |
| Cube ratio high | Matrix Calculating Limited | Raise L1/L0 reuse, keep emilogue integration |
| Scalar's share is high | Indexing and start-up expenses restricted | Move cycle nonvariant to host tilling, remove common Shape fast path |
| There's a big difference in time-consuming nuclear systems. | block/tail distribution uneven | Recheck `blockDim`, each nuclear length and tail formulae |
| UB Bank confidence high | UB Visit to Conflict | Adjust LocalTensor Offset, Slide or Padding |

## 2. Vector Bound

Priority check:

- Whether multiple UB computing stages can be completed before one `CopyOut`.
- Whether there is a duplication between fp16/bf16 and fp32 Cast.
- Whether sensitive mathematical links are raised to accuracy only in the necessary position.
- Whether `where`, Compare, Multiplication Mask introduces additional temporary tensor.
- Reduction uses the current SDK more suitable vector to contract API instead of scalar loop.

If the syntax requires an intermediate result of fp32, do not change to fp16 for speed.

## 3. MTE Bound

Check order:

1. Whether the number of elements of `DataCopy` is large enough to cover the cost of DMA setup each time.
2. Whether the GM address and the UB address satisfy as much as possible the 32B alignment.
3. Whether the non-matched path uses `DataCopyPad` only in tail or special Shape.
4. Whether the Queue Depth leads to MTE2 and Vector serial execution.
5. Enter if there is a cross file reuse that you can read less than once.

The goal is to allow overlap between removal and calculation. If the trail still shows a line, check the queue pairing and dependency, and then dial file size.

## 4. Scalar Bound

Small Shape, Broadcast, index/gather class operator is easily dominated by scalar index costs.

Usual handling:

- In the host tilling, it is expected to be a general transition.
- Create a special fast path for contiguous, scalar Broadcast, last-dim Broadcast.
- For small workload lower `blockDim` to avoid over-starting nuclear.
- Avoids double-counting of `div/mod` chains in Kernel; changes to tasking if necessary.

## 5. Bank Conflict

When profiling points to a UB bank or a bank-group condition:

- Let HF LocalTensor's initial offset at least staggered 32B.
- Avoids reading and writing hotspot operations in the same UB area.
- Reconfirmation of the units of vector repecat stride, block stride.
- Padding will increase UB occupancy, only if the conflict returns more than the loss of capacity.

## 6. L2 and Cache Policy

Cache hint is only used when the data is available for reuse:

- Read-only input that is read over and over again allows the normal cache policy to be enabled.
- One-time streaming data should not contaminate L2.
- Mathmul-like operator gives priority to reducing GM access through L1/L0 reuse.

## 7. Batch optimization of discipline

Only one performance assumption is changed for each round of the batch:

```text
op:
baseline correctness:
profile symptom:
changed variable:
tested shapes:
speedup/regression:
next action:
```

Do not mix tolerance changes, dtype over-cover contractions and kernel optimizations in the same round. If an optimisation raises only single sape but significantly retreats most sapes, it should retreat or break into a sape-specific path.

## 8. Sample feedback and narrow path

Sample-by-sample performance tables can be used to locate degradation paths, but not to disguise nudity Shape ad hocization as a generic kernel optimization. A narrow path is considered only when a dtype, rank, playout, calibration or numerical distribution pattern clearly dominates total consumption, and the common path rewriting risks are higher.

- Priority is given to the conditions as semantic rules: dtype, rank, contigouous, same-shape, last-dim mode, alignment, special value pattern; nudity Shape constant is used only as a cover for temporary diagnosis or known input to the collection.
- A narrow path must remain fully sape-covered, cannot shrink dtype/layout support, cannot bypass correctness verification.
- Each time a narrow path is introduced, with a comparison of the full and sample-by-sampling indicators; if the total indicator is not good, the rolling back should be done even if the individual sample is faster.
- If the narrow path stabilizes the gains clearly, then consider moving it up to the generic tilling, kernel branch or host dispatch rule, instead of continuing to stack more naked sapep judgments.
- The library can only be used as a proxy for degradation pathways or as a performance reference; it is suitable to save dtype conversions, complex broadcasts, mini-prescriptions or unusual distributions of special values, at equal cost to the completion of the AscendC generic optimization.

It's recommended that narrow path conditions be written as " semantics" instead of a full sape write to die:

```cpp
static bool IsLargeSameShapeInt8(const at::Tensor& a, const at::Tensor& b) {
  return a.scalar_type() == at::kChar &&
         b.scalar_type() == at::kChar &&
         a.dim() == b.dim() &&
         a.sizes() == b.sizes() &&
         a.is_contiguous() &&
         b.is_contiguous() &&
         a.numel() >= (1 << 24);
}

static bool IsLastDimBroadcastHalf(const at::Tensor& x,
                                   const at::Tensor& bias) {
  return x.scalar_type() == at::kHalf &&
         bias.scalar_type() == at::kHalf &&
         x.dim() >= 2 &&
         bias.dim() == 1 &&
         bias.size(0) == x.size(x.dim() - 1) &&
         x.is_contiguous() &&
         bias.is_contiguous();
}

at::Tensor op(const at::Tensor& x, const at::Tensor& y) {
  if (IsLargeSameShapeInt8(x, y)) {
    // The library realizes that only the clearly degraded mode is covered by the bypass, and the rest of the input remains on the AscendC main path.
    return at::maximum(x, y);
  }
  return launch_ascendc_kernel(x, y);
}
```

A special value mode should also be written as a semantic condition. For example, all-zero, all-NaN, education security, single-segment can have a fast path, but the output semantic integrity must be ensured:

```cpp
if (is_all_zero && op_is_multiplicative_or_activation_zero_preserving) {
  return at::zeros_like(input);
}

if (is_same_size_resize && input.scalar_type() == output_dtype) {
  return input.contiguous();
}
```

### 8.1 Degrade from per-shape to general subbarrel

Per-shape Optimizes the correct way to open it not by "writes which is which is which is which is which is which." First, the slow sample is attributed to a reusable semantic sub-bin. The common sub-barrel includes:

| Degraded sample characteristics | More general sub-barrel conditions | Common handling |
|---|---|---|
| It's so big, same-shape int8/uint8 elementwise slow | Integral dtype, same-shape, contigouous, Numel | Avoid fp32 round-trip, add native integer path or library sidewalk |
| rank higher half/ bf16 element by element operator | No, no-float dtype, rank > = 4, contigouous, no radio | tile tile, reduce past buffer, dtype special path if necessary |
| `(N, D)` and `(D,)` slow broadcast | Last-dim Broadcast, Bias contigouous, D alignment or close alignment | Most bordercast mode, line-repeated small input in kernel |
| Little D, slow down. | Reduction dim small, row many. | Multiple batches, or bypassing resynchronisation with scalar Statutes |
| Non last-dim Statute Slow | "Reduce Axis in discontinuity, stride, big." | Conversion of layout, subaxis specific paths, or use of libraries to achieve degradation pathways |
| index/scatter/gather Specific case slow | Indexing to a continuum, activity reduce, single security | DataCopy, Divisions Merge, Avoid Atoms, or Read rewrite |
| The special value sample is very slow | all-zero,all-NaN,constant,identity segment | Semantic speed path directly fills, copies or skips calculations |

Recommended process:

1. First press `gen_us / reference_us` or absolutely time-consuming to find the dominant sample.
2. Slow sample records dtype, rank, contigouous, bruadcast, reduce axis, Numel, alignment, special value distribution.
3. Finds a common condition that explains many slow samples and then writes host dispatch or Tiling Mode.
4. If only one sample is to be explained, the conditions are to be written in the narrowest synonym, and the model it represents is to be described in the note, rather than specifying the shape.
5. Only one new drum is activated each time to observe whether the full sample has been slowed down.

host side encodes semantic subbins into tilling mode, avoids recurring analysis of Shape:

```cpp
enum class ElemMode : int32_t {
  kGeneric = 0,
  kSameShapeContiguous = 1,
  kLastDimBroadcast = 2,
  kLargeIntegralSameShape = 3,
};

static ElemMode ClassifyElementwise(const at::Tensor& x,
                                    const at::Tensor& y) {
  const bool sameShape = x.sizes() == y.sizes();
  const bool bothContig = x.is_contiguous() && y.is_contiguous();
  const bool integral = at::isIntegralType(
      at::promote_types(x.scalar_type(), y.scalar_type()),
      /*includeBool=*/false);

  if (sameShape && bothContig && integral && x.numel() >= (1 << 24)) {
    return ElemMode::kLargeIntegralSameShape;
  }
  if (sameShape && bothContig) {
    return ElemMode::kSameShapeContiguous;
  }
  if (x.dim() >= 2 && y.dim() == 1 &&
      y.size(0) == x.size(x.dim() - 1) && bothContig) {
    return ElemMode::kLastDimBroadcast;
  }
  return ElemMode::kGeneric;
}

tiling.mode = static_cast<int32_t>(ClassifyElementwise(x, y));
tiling.inner = x.size(x.dim() - 1);
tiling.total = x.numel();
```

deviceend reuse Mode select light branch:

```cpp
if (tiling->mode == static_cast<int32_t>(ElemMode::kSameShapeContiguous)) {
  ProcessContiguous();
} else if (tiling->mode == static_cast<int32_t>(ElemMode::kLastDimBroadcast)) {
  ProcessLastDimBroadcast();
} else {
  ProcessGeneric();
}
```

When a full Shape constant must be used for the time being, it should be wrapped behind a more exterior synonym and protected only as a background:

```cpp
static bool IsKnownPathologicalLarge5DHalf(const at::Tensor& x) {
  if (x.scalar_type() != at::kHalf || !x.is_contiguous() || x.dim() != 5) {
    return false;
  }
  // Only serve as the end of a fixed input pool; the following should be replaced with the rank/numel/tile mode rule.
  return x.numel() > (1 << 22) && x.size(3) % 32 != 0;
}
```

## 9. Commonly reusable optimisation paradigm

The more stable pattern of benefits from the optimal record of multiple categories of elementwise, reduction, index, nonmalization and geometry operator is as follows:

- **Remove redundant conversion and constant calculations**: delete fp32 add-zero copy, repeat `ToF32/FromF32`, useless `Muls/Adds`, dead branch and unused TQe. If dtype meets the computational requirements, enter LocalTensor as the source of operator.
- **Integration of adjacent vector phase**: replace `Muls+Add` with `Mad`, merge intermediate steps of softplus/action, or merge scale, bias, cast from epilogue before last writing.
- **The length of the UB budget**:tie should be calculated by dtype, scratch Butcher, queue depth and double buffering requirements; the benefits usually come from reduced tile times, but to prevent UB spills and queue capacity from crowding out each other.
- **Leaves removal and calculation overlap**: give priority to try Que depth 2/3, double/triple buffer, advance CopyIn next file, and reduce handwritten `SetFlag/WaitFlag`. If you line up only at the same stage, adding buffer depth will not automatically yield benefits.
- **Synchronization only where necessary**: Where the Statute or scalar can be used after rereading `PipeBarrier`, it is usually lighter than the flag of the event. Check that subsequent reading and writing do not depend on trans-current water before deleting sync.
- **Quantified small work units**: narrow lines softmax, argmax, Cross entropy, Foreach and SwigLU small lines, common gain from combining multiple rows/tensor to a UB file or a kernel call, with low start-up, synchronization and Copyout costs.
- **Replace scalar with a continuous block for reading and writing**: in index /gather/scatter/resize, the return is usually much greater than fine-tuning when it is possible to change `GetValue/SetValue` to a batch of DataCopy, a continuous CopyOut, group grouping light, or writing back after UB has been saved.
- **Move index algorithms out of the inner layer**: move `div/mod`, profile, row base, w-table base, security base, fixed dimension judgement to the beginning of a host tilling, Init or watch; use an incremental amount of asfset and int32 counters as far as possible.
- **Models using special values and structures**: all-zero, all-NaN, single-segment, sustainability security, Same-size reze, two-class cross enterprise, small reducation, etc., can have independent fast paths, but conditions must come from semantics rather than accidental Shape.
- **Reduced GM round-trip and intermediate writeback**: priority caches when input, gamma, cos/sin, row Cache, partial subs are available in UB; final results are written as much as possible to avoid the middle tensor writing GM first and read back.
- **Rational choice of scalar or vector's Statute**: little last-dim hardware vector's Statute may be flooded with synchronous costs, but scalar's cumulative costs are faster; vector Reduce is preferred in large, multi-line or bulk Statutes.
- **There is also a performance assumption for the clean-up of dead codes**: deletion of unused include, members, intry, helper, Mode branch sometimes reduces the compilation product and improves the schedule, but should be validated as a low-risk step rather than as a substitute for real bottleneck optimization.

### 9.1 Imputation of redundancy copy, cast and constants

Common bad tastes are used to unify dtype paths, and float input also does `Adds(x, 0)` or `Cast(float->float)` first. This will account for one more UB Buffer and one more vector pass.

```cpp
// Poor: float32 path is also copied.
auto xLocal = xQ.DeQue<T>();
auto xf = calcBuf.Get<float>();
if constexpr (std::is_same_v<T, float>) {
  Adds(xf, xLocal, 0.0f, count);
} else {
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
}

// Better: float32 directly uses input; nonfloat conversions.
auto xLocal = xQ.DeQue<T>();
if constexpr (std::is_same_v<T, float>) {
  ComputeFloat(xLocal, count);
} else {
  auto xf = calcBuf.Get<float>();
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
  ComputeFloat(xf, count);
}
```

Similarly, fixed coefficients can be prefolded in host or `Init` to avoid repetition of scalar multipliers per file:

```cpp
// Host/Init phase
tiling.effScale = scale * baseLog;
tiling.effShift = shift * baseLog;

// Kernel Phase
Muls(tmp, x, tiling.effScale, count);
Adds(tmp, tmp, tiling.effShift, count);
```

### 9.2 Budget Selection by UB

The length of the file should not only be input-sized, but also include dtype, scratch buffer, queue depth and double buffer. Experience, use the security budget and fine-tune the results by sample.

```cpp
static int64_t AlignDown(int64_t x, int64_t align) {
  return x / align * align;
}

int64_t PickTile(int64_t ubBytes, int64_t elemBytes,
                 int64_t inputBuffers, int64_t outputBuffers,
                 int64_t scratchBuffers, int64_t queueDepth) {
  int64_t buffers = (inputBuffers + outputBuffers) * queueDepth + scratchBuffers;
  int64_t usableBytes = ubBytes * 8 / 10;  // Leave Queue and temporary object balances.
  int64_t tile = usableBytes / (buffers * elemBytes);
  return std::max<int64_t>(256, AlignDown(tile, 256));
}
```

If a operator has both fp32 and fp16/bf16 paths, per-dtype file is usually required:

```cpp
tiling.tileLength =
    dtype == DTYPE_FLOAT ? PickTile(ubBytes, 4, 2, 1, 3, 2)
                         : PickTile(ubBytes, 2, 2, 1, 4, 2);
```

### 9.3 Removal and calculation overlap

The key to the double/triple buffer is not "to increase the number of buffer," but the loop order really allows the next file to overlap with the current file Compute/CopyOut.

```cpp
constexpr int32_t BUFFER_NUM = 2;

for (int32_t i = 0; i < tileNum + BUFFER_NUM; ++i) {
  if (i < tileNum) {
    CopyIn(i % BUFFER_NUM, i);
  }
  if (i >= 1 && i - 1 < tileNum) {
    Compute((i - 1) % BUFFER_NUM, i - 1);
  }
  if (i >= 2) {
    CopyOut((i - 2) % BUFFER_NUM, i - 2);
  }
}
```

If Compute is waiting for CopyIn to be finished, CopyOut is waiting for Compute to be finished, Trace will still show a string. Check the EnQue/DeQue sequence and event dependence of the queue and then adjust the file.

### 9.4 Synchronized noise reduction

After the Statute, just to make sure that vector's results are read back to scalar, `PipeBarrier` tends to be lighter than `SetFlag/WaitFlag`:

```cpp
ReduceSum(sumLocal, xLocal, tmpBuf, count);
PipeBarrier<PIPE_V>();
float sum = sumLocal.GetValue(0);
```

Do not mechanically delete all syncs. If you follow up over the MTE/VEC/Scalar stream to read and write the same buffer, you still need to keep the correct queue sync or event sync.

### 9.5 Quantified rows and multiple outputs

The small lines softmax, argmax, cross enterprise, foreach type operator are often dominated by Kernel launch, barrier and CopyOut expenses. The results can be saved in UB and then rewritten.

```cpp
constexpr int32_t ROW_BATCH = 16;
auto outLocal = outBuf.Get<float>();

for (int32_t rb = 0; rb < rows; rb += ROW_BATCH) {
  int32_t activeRows = Min(ROW_BATCH, rows - rb);
  for (int32_t r = 0; r < activeRows; ++r) {
    outLocal.SetValue(r, ComputeOneRow(rb + r));
  }
  DataCopy(outGm[rb], outLocal, activeRows);
}
```

Foreach class Tensor [] operator uses the same simple mode for most inputs, giving priority to combining these inputs with a kernel call; only irregular or special values are diverted separately.

### 9.6 Replace scalar reading and writing with continuous blocks

The index class operator is susceptible to degradation to each element `GetValue/SetValue`. When the index map forms a continuous segment on an axis, the continuous segment should be moved to UB and then written back in bulk.

```cpp
// Poor: Each element reads and writes individually.
for (int32_t i = 0; i < count; ++i) {
  T v = xGm.GetValue(base + i);
  yGm.SetValue(outBase + i, v);
}

// Better: a continuous period of direct removal.
DataCopy(local, xGm[base], count);
DataCopy(yGm[outBase], local, count);
```

For rather/scatter, you first identify the rank2, dim0, last-dim patterns, structure reducation, etc., and then write a blocked path for these modes; preserve the full semantics of the generic path.

### 9.7 Index algorithms

The inner layer of geometry, resize, grid, scatter type operator `div/mod` and multi-level stide is very expensive. Turn the no-change item to the start of the watch or host tilling:

```cpp
// Poor: every output point in the inner layer doubles.
int64_t n = linear / (OH * OW);
int64_t rem = linear % (OH * OW);
int64_t oh = rem / OW;
int64_t ow = rem % OW;
int64_t inBase = ((n * C + c) * IH + h0) * IW;

// Better: Move along lines, with only incremental updates in the inner layer.
int64_t rowBase0 = ((n * C + c) * IH + h0) * IW;
int64_t rowBase1 = ((n * C + c) * IH + h1) * IW;
for (int32_t ow = owStart; ow < owEnd; ++ow) {
  int64_t wBase = wTableBase + ow;
  ComputePixel(rowBase0, rowBase1, wBase);
}
```

If sufficient range can be demonstrated, the counter and offset will be given priority to `uint32_t` / `int32_t` to reduce the pressure of 64-bit integer algorithms.

### 9.8 Reuse UB data and reduce GM returns

The same line of data or coefficients are often required for operator in the same category. If UB capacity allows, pre-emption to a specific buffer:

```cpp
// Pass 1: Read x, cumulative subsq, while cache x.
DataCopy(xLocal, xGm[rowBase], D);
DataCopy(xCache, xLocal, D);
ReduceSum(sumsq, Square(xLocal), tmp, D);

// Pass 2: Directly reuse xCache, no longer read x from GM.
float rscale = Rsqrt(sumsq.GetValue(0) / D + eps);
Muls(xLocal, xCache, rscale, D);
Mul(outLocal, xLocal, gammaLocal, D);
DataCopy(yGm[rowBase], outLocal, D);
```

Cache whole lines may not work if D is small; if D is large, split file caches may squeeze out the necessary scratch. First, use the UB budget to decide whether to start.

### 9.9 scalar Path to the Statute

The hardware vector statute is suitable for large or multi-line batch statutes; small last-dim may be flooded with synchronous and temporary buffer costs.

```cpp
if (D <= 32) {
  float sum = 0.0f;
  for (int32_t i = 0; i < D; ++i) {
    float v = static_cast<float>(xLocal.GetValue(i));
    sum += v * v;
  }
  outLocal.SetValue(0, PostProcess(sum));
} else {
  Mul(tmp, xLocal, xLocal, D);
  ReduceSum(sumLocal, tmp, reduceTmp, D);
  PipeBarrier<PIPE_V>();
  outLocal.SetValue(0, PostProcess(sumLocal.GetValue(0)));
}
```

### 9.10 Numerical sensitive statutes and quantitative boundaries

Reduction after `sqrt/div/round/cast` links, tile length and partial sum
Trees are not just performance parameters, they change the last two ulps and then push the int8/uint8 quantitative results over.
Half-integer boundary. The path is to be adjusted to one variable at a time and to record each candidate's file
Hard-mismatch numbers; do not ignore integer output exact date because the overall MERE/MARE is very small.

```cpp
// The tile is not just an insinuation parameter; it changes the partial sum tree.
constexpr uint32_t TILE = 256;
for (uint64_t off = 0; off < dim; off += TILE) {
  uint32_t count = Min<uint64_t>(TILE, dim - off);
  Mul(tmp, xLocal, xLocal, count);
  ReduceSum(partial, tmp, reduceTmp, count);
  partialBuf.SetValue(chunk++, partial.GetValue(0));
}
ReduceSum(total, partialBuf, reduceTmp, chunk);
```

If a quantitative operator is only equal to a small number of elements `+/-1`, priority is given to:

- Whether the sum/amax tree shape differs from the reference path;
- Scale is calculated by scalar or read back by vctor `Div/Muls`;
- round Mode is consistent with reference, especially half-bundary;
- Whether the order of clamp and past changed the NAN/ Inf or boundary values.

Attention, matmul epilogue or volume inverse concentrate all failure points in small-value/
Cancellation, `abs(out) < eps -> 0`, can only be used as a diagnostic tool, not as a direct means.
Generic restoration. It often lowers the worst relative to error, but puts more non-zero values to zero.
Makes the whole MERE or small-value mismatch variable. Only if the threshold comes from a clear semantic
(e. g. full mask rows, padding contributions, known zero input segments) and also lower MERE/MARE
It should only be kept with the number of mismatch.

The special value branch avoids the temporary use of `inf/inf` and `0/0` algorithms in the AICore Scalar path
NN; this type of writing may trigger an anomaly or instability on different compile/hardware combinations.
NAN/ Inf, priority is given to the use of clear bit-pattern for writing, entering raw values for dissemination, or for special values
Converts the envelope to a separate helper and validates it with a case that only covers special values.

When handwritten scrolling, pooling, or scalar kernel type, paddy cross-border branch is not mechanical
`continue` is followed by an equal value multiplied by 0. This is right for generic limited input, but is the right weight, the right weight, the right weight, the right weight, the right weight, the right weight, and the right value.
The library achieves the possibility of retaining the number of actions on the other side of the peding position that includes NAN/ Inf
`0 * inf -> NaN`This special value spreads.`mere=0/mare=0` but NaN maskIt's not consistent.
Priority check if the padding branch skips a non-limited operation; terminological conditions are clearly disseminated during repair
quiet-NAN, do not make NAN on an ad hoc basis with illegal arithmetic.

### 9.11 Boundaries cleared by dead code

The deletion of the dead code reduces the sample, memory pressure or binary volume of the template, but it is not a major optimisation tool.

- Unattainable mode branch and entry.
- Only member variables consumed in `Init` that are no longer used thereafter.
- No longer use TQue/TBuf, include and helper.
- The dead branch of `if constexpr` that is not related to the current dtype path.

After cleaning, check whether CMake, register entry, host wrapper and Kernel entry are consistent and avoid deleting ABI or coverage together.
