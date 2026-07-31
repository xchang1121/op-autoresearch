---
name: ascendc-reduce-broadcast
description: "The standard formulation for the Reduce + Broadcast model (softmax / playnorm / rmsnorm / log-softmax) is how to use ReduceMax/Sum, HoleReduce, BlockReduce, how to get scalar back to vector and how to avoid the bombing of the NAN cascade and accuracy."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "softmax, layernorm, rmsnorm, instance_norm, log_softmax"
---

# AscendCReduce +Broadcast

The common skeletons of operator, such as softmax, playernorm, RMSNorum, are**per-row reduce →, broadcast back to elementwise**with reduce. AscendC is very flexible on this "reduce" step, but many fail come from the following three traps:

1. **Wrong entry**(`ReduceMax` returns LocalTensor or scalar?
2. **Scalar will not be broadcast back to vector as a subtraction/separation**
3. **Numerical stability damage**(exp without loss of max → spill; fp32 cumulative → accuracy bomb)

Ben Skill gives a couple of recognized stabilitys, Pattern.

## 1. Overview provided by AscendC

These are in the `/Ascend/cann-8.5.0/aarch64-linux/asc/include/basic_api/kernel_operator_vec_reduce_intf.h`:

| API | Enter Shape | Output Shape | Purpose |
|---|---|---|---|
| `BlockReduceMax/Sum` | (N,) | (N/64,) | For each 64 element group reduce, output is still LocalTensor |
| `PairReduceSum` | (N,) | (N/2,) |   Next to the two sides   |
| `WholeReduceMax/Sum` | (N,) with `repeatTimes` | (repeatTimes,) | **batched row reduce, commonly** |
| `ReduceMax/Sum` | (N,) | The LocalTensor | Make the whole input into one scalar, write to dst[0] |
| `GetReduceMaxMinCount` | — | scalar | Last REduceMax scalar to `T &` repository |

**To judge which one:**

- Single line reduce → `ReduceMax/Sum` (dst is 1 element LocalTensor)
- Okay, watched and reduce → `WholeReduceMax/Sum` plus `repeatTimes=N`
- Phased Hierarchical (64-wise, then whole) → `BlockReduceMax` and `ReduceMax`

## 2. Standard way to stabilize softmax

Formula: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.**No minus max will spill**(even input to 89.0, `exp(89) ≈ 4.5e38`, fp32 limit 3.4e38).

```cpp
template <typename T>
__aicore__ inline void SoftmaxRow(
    const LocalTensor<T>& xIn,     // (D,) Current line input
    const LocalTensor<T>& yOut,    // (D,) Current Line Output
    const LocalTensor<float>& fp32Buf,  // At least. 2*D individual fp32 slot
    uint32_t D)
{
    auto x = fp32Buf;                       // [0,  D)  upcast After x
    auto e = fp32Buf[D];                    // [D, 2D) exp(x - max)

    // 1) Upcast to fp32 (Ensure accuracy)
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Adds(x, xIn, 0.0f, D);
    } else {
        AscendC::Cast(x, xIn, AscendC::RoundMode::CAST_NONE, D);
    }

    // 2) row-max → scalar
    AscendC::ReduceMax<float>(/*dst=*/e, /*src=*/x, /*workLocal=*/x, D, false);
    //                          ^^Borrow.e[0]As1Elementsdst
    AscendC::PipeBarrier<PIPE_V>();          // Wait. reduce And then it came down. e[0]
    float maxVal;
    AscendC::GetReduceMaxMinCount<float>(maxVal);   // Take it. scalar to GPR

    // 3) x - max → e
    AscendC::Adds(e, x, -maxVal, D);

    // 4) exp
    AscendC::Exp(e, e, D);

    // 5) row-sum → scalar
    AscendC::ReduceSum<float>(x, e, x, D);   // dst Borrow. x[0] Use it.
    AscendC::PipeBarrier<PIPE_V>();
    float sumVal;
    AscendC::GetReduceMaxMinCount<float>(sumVal);

    // 6) divided by sum
    float invSum = 1.0f / sumVal;
    AscendC::Muls(e, e, invSum, D);

    // 7) Downcast to target dtype
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Adds(yOut, e, 0.0f, D);
    } else {
        AscendC::Cast(yOut, e, AscendC::RoundMode::CAST_RINT, D);
    }
}
```

Key non-variant:
- **fp32 Intermediate calculation**, even if IO is fp16/bf16. fp16 `exp` is highly vulnerable to spillage.
- **Can't get out of**before reducing max (even if the note says "it should be all right.")
- `ReduceMax`After that, we have to`PipeBarrier<PIPE_V>()`  To make  `GetReduceMaxMinCount`Read correct.scalar;missingbarrierReads old values or0And then...`exp(x - 0)`Spill→AllInf → softmax NaN(softmaxThe most common stage"AllNaN"The accident.
- `Muls(invSum)` is faster than `Div(sum)` by 5×, and Div of AscendC is imitated.

## 3. LayerNorum Standard

Formula: `y = (x - mean) / sqrt(var + eps) * gamma + beta`, where `mean = E[x]`, `var = E[x^2] - mean^2`.

```cpp
// fp32Buf needs at least 3*D float slot
auto x   = fp32Buf;                  // [0,  D)
auto x2  = fp32Buf[D];               // [D, 2D)
auto out = fp32Buf[2*D];             // [2D,3D)

// upcast
AscendC::Cast(x, xIn, ..., D);

// Meaning: Direct in a compactor with RepeatReducesum 1/D scale
AscendC::Muls(x2, x, 1.0f / D, D);              // x / D
AscendC::ReduceSum<float>(out, x2, x2, D);
AscendC::PipeBarrier<PIPE_V>();
float mean;  AscendC::GetReduceMaxMinCount<float>(mean);

// var = E[(x - mean)^2]
AscendC::Adds(x2, x, -mean, D);                 // x - mean
AscendC::Mul(out, x2, x2, D);                   // (x - mean)^2
AscendC::Muls(out, out, 1.0f / D, D);
AscendC::ReduceSum<float>(out, out, out, D);
AscendC::PipeBarrier<PIPE_V>();
float var;   AscendC::GetReduceMaxMinCount<float>(var);

// normalize: (x - mean) * rsqrt(var + eps)
float invStd = 1.0f / std::sqrt(var + eps);     // scalar,host-style fp32
AscendC::Muls(out, x2, invStd, D);              // x2 still holds (x - mean)

// Affine: out = out * gamma + beta (gamma/beta)
AscendC::Mul(out, out, gammaLocal, D);
AscendC::Add(out, out, betaLocal, D);
```

**Two deaths of Layer Norm**:
- `var` used fp16 to count → when entering large (x-man) ##2 to gain inf
- `rsqrt(var + eps)` magnify values in var extreme hours (close to zero differential lines) to output nan/inf; eps cannot save, fp32 is recommended for `1e-5f` starting

## 4. RMSNorm

Formula: `y = x / sqrt(mean(x^2) + eps) * gamma`, no decrease in mean this step.

```cpp
AscendC::Cast(x, xIn, ..., D);
AscendC::Mul(x2, x, x, D);                       // x^2
AscendC::Muls(x2, x2, 1.0f / D, D);
AscendC::ReduceSum<float>(out, x2, x2, D);
AscendC::PipeBarrier<PIPE_V>();
float meanSq;  AscendC::GetReduceMaxMinCount<float>(meanSq);

float invRms = 1.0f / std::sqrt(meanSq + eps);
AscendC::Muls(out, x, invRms, D);
AscendC::Mul(out, out, gammaLocal, D);
```

## 5. Batched Multiline: with `WholeReduce*` + `repeatTimes`

To handle line B (B known or controlled) in a kernel call,**do not go to the cut-off view**(for further information [[[ascendc-localtensor-subviews]]) and use the `repeatTimes` parameter of `WholeReduceMax` to intrinsic invert itself:

```cpp
// InLocal Shape =(B *D,) fp32, D elements per row continuous
LocalTensor<float> maxOut;  // shape >= (B,)

AscendC::WholeReduceMax<float, false>(
    /*dst=*/        maxOut,
    /*src=*/        inLocal,
    /*Mask=*/ D, / Every reduce processing D elements
    /*Report Times =*/ B, // Total B
    /*dstRepStride=*/1, / / Every result interval 1 element
    /*srcBlkStride=*/1,
    /*srcRepStride=*/ D / 8/ / src Jump D/8 32B block
);
```

`D` must be able to be severed (or aligned with 32B) by 64, otherwise the padded D is used, and the excess position is used to fill `-INFINITY` with `Duplicate` before reduce(max path) or 0(sum path).

## 6. Pass sprit: two kernel vs. single kernel single lanch

Softmax tried to split two kernels into max writing GMEM, pass-2 reading max reading exp/sum/divide) with the motivation**to reduce UB pressure**. But this road**will hardly win**for reasons:

- Center max to go GMEM → One more DMA + one lanch
- Pass-1 and pass-2 must host-side `aclrtSynchronizeStream` or pass-2 read old data → all NAN
- Only Kernel uses `PipeBarrier<PIPE_V>` to synchronize inside UB, zero additional cost

**Conclusion**: Single Kernel single lanch is always faster unless D is too big to fit UB.

## 7. Failure mode quick checkup

| The phenomenon | Diagnosis | Reform |
|---|---|---|
| `Implementation=N/N` All NAN | Exp Spill → softmax all 0 → 0/ 0 = NN. Checks if leaks max or max scalar did not read | Order of `PipeBarrier<PIPE_V>` + `GetReduceMaxMinCount` for §2 |
| All Inf | exp input unreduced max and reduce sum spills into Inf | Ibid. |
| It's roughly right, it's very different. | {\cHFFFFFF}{\cH00FF00} reduce uses fp16 plus → Catastropic cancellation | fp32 Intermediate calculation |
| Output right, but less than ref 5× | `Div` for `Muls(1/x)`; or two Kernel split | §2 to Muls; §6 single Kernel |
| `errno 507035` UB OOB | Enter/output LocalTensor overlaps within UB | CalcBuf slices must not overlap (for further details [[ascendc-localtensor-subviews]] §4] |

## 8. Don't do anything.

- Do not add up fp16/bf16 to mean /var/sum,**permanently upcast to fp32**.
- Do not read `dst[0].GetValue()` directly after reduce - `LocalTensor::GetValue` does not guarantee a committee on V page, first `PipeBarrier<PIPE_V>` + `GetReduceMaxMinCount` takes the scalar path.
- Do not replace `1/sum` with `Div(out, e, sumTensor, D)` - AscendC `Div` with very slow for "mins"
- Do not save eps - Looks like input is never zero, but there's a zero-side difference on the backscripe or border.
