---
name: ascendc-localtensor-subviews
description: "LocalTensor subviews and index rules: which subviews are legal, which triggers UB crossing borders, and which are valid alternatives to multi-line watching / per-row processing."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "elementwise, reduction, normalize, softmax"
---

# AscendC LocalTensor Subview and Index Rules

In Kernel, it is very common to take a local Tensor subview - to break the UB buffer line after the bat, to split the double buffer in half, and to take the first element from the result of reduce.**The vast majority of UBs crossed the border (VEC input error / UB access out of homes / erno 507035 vector core exception) from a non-variant that has been violated: the deviation of subviews must be known during the writing period at Kernel.**

Ben Skill has to read it before you're going to do multi-watching, double-buffer splits, per-row processing, reduce result i scalar.

## 1. A word rule.

**`tensor[const_offset]` is legal; `tensor[runtime_var]` crosses borders at most vector intrinsic entries.**

```cpp
// ✅ legal: offset is a constant of the compilation period
LocalTensor<float> half2 = calcBuf.Get<float>()[TILE_LENGTH];  // TILE_LENGTH Yes. constexpr
AscendC::Exp(half2, half1, count);

// ❌ illegal: offset at Kernel before running
uint32_t row_off = block_idx * tiling->paddedD;       // runtime
LocalTensor<float> row_view = inBuf.Get<float>()[row_off];   // It's done.
AscendC::ReduceMax(maxBuf, row_view, count, ...);              // runtime UB OOB
```

`operator[]` on LocalTensor accepts runtime expression (C++ level) -**compiled**, but vector intrinsic needs static UB start-up address when decoded descriptor, runtime offset will allow hardware to capture an illegal UB address and trigger errno 507035.

## 2. Really valid subview source

Only deviations from the following sources are secure on vector intrinsic:

| Offset Source | Legal | Annotations |
|---|---|---|
| `constexpr` / `template` template constant | ✅ | `tensor[TILE_LENGTH]`,`tensor[BUFFER_SIZE / 2]` |
| `#define` Macro | ✅ | `tensor[HALF_TILE]` |
| A constant index after a cycle is expanded | ✅ | Expand `#pragma unroll` |
| Const parameters for the `__aicore__ inline` function | ✅ | Template Parameter Import |
| `tiling->fieldName` values read from GMEM | ❌ | runtime Variable |
| `block_idx`,`GetBlockIdx()` | ❌ | runtime Variable |
| `i` of the circular variable `for (i = 0; i < N; ++i)` | ⚠️ | Only when N is constexpr and compiler recycles are secure |

Method of judgement:**This is a runtime variable if it relies on any GM-readed fields, kernel reference, `GetBlockIdx()` or non-extension loop variable.**

## 3. Multi-basket counters and corrects

Soft maximise, playnorm, RMSNornm, and so on, per-row operations often want to "a lanch to handle B-line expenses." Common error:

```cpp
// ❌ Example: Runtime-offset subview leads to UB OOB
__aicore__ inline void Process() {
    auto inLocal = inQueue.DeQue<float>();  // shape = (B, paddedD)
    for (uint32_t r = 0; r < B; ++r) {
        auto row = inLocal[r * paddedD];               // runtime offset
        AscendC::ReduceMax(maxTmp, row, paddedD, ...); // ☠ runtime OOB
        // ...
    }
}
```

There are three legal alternatives:

### Method A: per-row Independent Alloc/Free (simplified, slightly less performance)

One full queue cycle in each row, avoiding a check tensor + subview at root:

```cpp
for (uint32_t r = 0; r < rows_this_core; ++r) {
    auto in = inQueue.AllocTensor<T>();
    DataCopy(in, xGm[r * D], paddedD);
    inQueue.EnQue(in);
    auto x = inQueue.DeQue<T>();
    // Now the starting address for x is UB 0 offset, all intrinsic is secure
    AscendC::ReduceMax(maxBuf, x, D, ...);
    // ...
    inQueue.FreeTensor(x);
}
```

### Method B: Link all rows into a series of consecutive data (only when reduce axis is the last dimension)

If you reduce at last-dim, the data in line B is a continuous `B * D` element on GM, treat the entire paragraph as an `(1, B*D)` LocalTensor processing -**still a reduce call**but use `mask`/ `repeatTimes` to get inrinsic to press a D paragraph itself:

```cpp
// HoleReduceMax for recap: one-time treatment of D-length row
AscendC::WholeReduceMax<float, false>(
    maxOut,          // shape = (B,) of LocalTensor
    inLocal,         // shape = (B*D,) The flat. LocalTensor
    /*mask=*/D,
    /*repeatTimes=*/B,
    /*dstRepStride=*/1, / / every reduce result
    /*srcBlkStride=*/1,
    /*srcRepStride=*/D / 8 / src Jumping D/8 block (32B)
);
```

This path is a position that really accelerates the speed of the watched softmax / playernorm - using `repeatTimes` + stride to allow intrinsic to move over and over again,**without having to write any child view**.

### Method C: Fixed B (frequently but occasionally useful) for the known translation period

Only when B compiles the kernel period constant (template parameters or macros) may `r * paddedD`, after circulation, be able to fold compiler into the constant. This writing is weak,**is not recommended as the preferred**, but can be used as a special path for a single shape.

## 4. UB Internal Aliasing Rules

The input and output operations of vector intrinsic**must not overlap within UB**, otherwise the hardware will not miss the translation period but generate silent data curruption, and the next op to use the result will explode in the form of "accuracy Error" or "NAN".

```cpp
// ❌ Aliasing:c both src and dst
AscendC::Adds(c, c, 0.0f, count);  // Equivalent c += 0Hardware does not guarantee semantics

// ❌ Overlap: dst covered the second half of src
auto a = buf.Get<float>();
auto b = buf.Get<float>()[count - 8];  // Deliberate overlap 8 elements
AscendC::Mul(b, a, c, count);            // ☠ Data crash.

// ✅ in-place must be "specify in-place" with the same address as shape
//    Or dst completely stagger the whole src
```

Special attention is paid to a few LocalTensors of the `calcBuf` hand-held slices,**which must not overlap at all as long as they appear in a single intrinsic call**.

```cpp
// CalcBuf Total Size 2 * Tile_LENGTH* sizeof (float)
auto a = calcBuf.Get<float>();                  // [0,                 TILE_LENGTH)
auto b = calcBuf.Get<float>()[TILE_LENGTH];     // [TILE_LENGTH, 2*TILE_LENGTH)
// No overlap a and b at any count < = Tile_Length
```

## 5. Failure mode quick checkup

| Wrong. | Actual reasons | Reform |
|---|---|---|
| `VEC instruction error: the ub address out of bounds` / errno 507035 | Subview offsets are runtime variables; or src/dst crossing the UB boundary in UB | Change to Method A or B (see §3) |
| `error: no matching function for call to '...'` when compiling | LocalTensor returns non-matched vector intrinsic | Most of the subview returns not `LocalTensor<T>`, but `LocalTensor<T> &&`, which is stored in famous variables before transmission. |
| accuracy All NAN / All Inf, but Kernel does not miss | UB ALIASING QUIET CORRUPTION | Draw an interlocking table with all pieces cut by `[offset]` on CalcBuf to confirm that there is no overlap |
| First run right, change a line and get wrong.| Took a runtime offset view, which happens to be normal for some sape down to zero. | Rewrite with A by §3 |

## 6. Don't do anything.

- Do not introduce the `inLocal[r * stride]` mode for "batch B row" without exception.
- Do not use the `*(__ubuf__ T*)((__ubuf__ uint8_t*)inLocal.GetPhyAddr() + offset)` manual pointer calculation to circumvent the subview limit - the compilation period will pass, but destroy the page sync inference that the subsequent intrinsic will be abnormal.
- Do not reassign repeatedly to `buf[...]` for the same LocalTensor in the `Process()` call - compiler tracks each LocalTensor's initial UB address is tied and re-directing will cause a Pipeline schedule error.
