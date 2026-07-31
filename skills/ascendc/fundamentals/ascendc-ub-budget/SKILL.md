---
name: ascendc-ub-budget
description: "UB Capacity Budget: 910 B/910B3/910B4 by UB sizes, queue /calcBuf/TBuf byte algorithms, BUFFER_NUM with common security combinations for Tile_LENGTH, and diagnostic path for UB OOB."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC UB Capacity Budget

UB (Unified Buffer) is an AI Victor Core on-chip hold.**It is a limited and hard hardware resource**— once `pipe_->InitBuffer` applications exceed physical UB, compiled and run UB OOB (erno 507035 vector core exception).

The vast majority of the accidents I took from BUFER_NUM from 2 to 3 and crashed, or "I want to catch 4 lines." This skill taught you**to clear the byte before writing the code.**

## 1. Each device UB capacity

| device | UB for each Victor Core | Remarks |
|---|---|---|
| 910B (Atlas 800T A2 reasoning card) | **192 KB** | Mononuclear UB |
| 910B3 (training 8 cards) | **192 KB** | Mononuclear UB |
| 910B4 (training 8 cards, tight) | **128 KB** | Smaller |
| 310B / 310P | 256 KB | aiv Unlike aic, only aiv |

**The smallest UB of the target device is programmed.**If the code is to run 910B, too, 910B4 is designed to be the 128 KB cap.

## 2. UB Byte Book Formula

```text
ub_used = sum(InitBuffer Bytes per application)
        + compilerInvisible scratch  (~ 2–4 KBKeep it. headroom)
```

Bytes of each `pipe_->InitBuffer` application:

```text
TQue<POSITION, BUFFER_NUM>:   buffer_bytes_per_alloc × BUFFER_NUM
TBuf<POSITION>:               buffer_bytes  (BUFFER_NUM Forever. = 1, there's no more buffer)
```

Take the example of elementwise unary kernel:

```cpp
pipe_->InitBuffer(inQueue_,  BUFFER_NUM, TILE_LENGTH * sizeof(T));     // VECIN
pipe_->InitBuffer(outQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));     // VECOUT
pipe_->InitBuffer(calcBuf_,  2 * TILE_LENGTH * sizeof(float));         // VECCALC

// Assumptions BUFTER_NUM=2, Tile_Length=4096, T=float(sizeof=4)
ub_used = 2 * (4096 * 4)      // inQueue:   32 KB
        + 2 * (4096 * 4)      // outQueue:  32 KB
        + 2 * 4096 * 4        // calcBuf:   32 KB
        = 96 KB
// 192 KB card with 96 KB headroom, secure.
```

Also Kernel, lift `BUFFER_NUM` from 2 to 3, `TILE_LENGTH` to 8192 (half/bf16 want more):

```cpp
// BUFFER_NUM=3, TILE_LENGTH=8192, T=half (sizeof=2)
ub_used = 3 * (8192 * 2)      // inQueue:   48 KB
        + 3 * (8192 * 2)      // outQueue:  48 KB
        + 2 * 8192 * 4        // calcBuf:   64 KB
        = 160 KB
// KB: 32 KB to Scratch, tight but feasible
// 128 KB card: already OOB
```

## 3. Budget table for the design phase (910B3, 192 KB)

Write a new kernel by estimating which Buffer is needed. Select the following table:

| Purpose | RECOMMENDED BUFER_NUM | Single Buffer bytes | Tile_Length cap |
|---|---|---|---|
| Single-input single output elementwise (fp32) | 2 | TILE × 4 | 8192 |
| Single-input output elementwise (fp16/bf16) | 2 | TILE × 2 | 16384 |
| Single Input Output + a fp32 calcBuf | 2 | TILE × 2 + 2·TILE × 4 | 8192 |
| Double Input Single Output (e. g. add/mul) | 2 | 2 × (TILE × 2) | 12288 |
| Reduce-broadcast (softmax/layernorm) | 1–2 | TILE × 4 + 3·TILE × 4 | 4096 |
| Cube + Vector fused | 1 | Complex | See §6 |

**Empirical law**: control of `ub_used` at 80% of hardware UB (910B3: 154 KB), with 20% reserved for compiler hidden scratch, Stack, tilling case, etc.

## 4. The real cost of more buffer (BUFFER_NUM)

BUFFER_NUM is motivated to overlap with MTE2 (load) and V (vector compute) and theoretically to hide DMA latency. But the price is**UB takes**BUFFER_NUM multiplication**. See kernel's compute-to-memory radio:

| scene | BUFFER_NUM=1 | BUFFER_NUM=2 | BUFFER_NUM=3 |
|---|---|---|---|
| Compute > > DMA (e. g. exp chain long unary) | Waste of UB | **Best** | Marginal gains < 5% |
| Compute ≈ DMA (simply add) | Slow | **Best** | There's almost no return. |
| DMA > Compute (e.g. copy/cast)| Serious bottlenecks | Large improvement | **probably worth it**, but often bound by UB size |

**Do not automatically push BUFFER_NUM to 3**. First use 2 to walk, profling to see if MTE2 really doesn't overlap with V before deciding.

## 5. CalcBuf and TBuf

`calcBuf` / `TBuf` is a user-managed temporary storage area, often cut into float buffer for intermediate results.**Cuts must be offset by a constant of the compilation period**(detailed in [[ascendc-localtensor-subviews]]):

```cpp
// calcBuf size = 3 * TILE_LENGTH * sizeof(float)
pipe_->InitBuffer(calcBuf_, 3 * TILE_LENGTH * sizeof(float));

// In Compute:
auto c = calcBuf_.Get<float>();                       // [0,             TILE_LENGTH)
auto w = calcBuf_.Get<float>()[TILE_LENGTH];          // [TILE_LENGTH,  2*TILE_LENGTH)
auto z = calcBuf_.Get<float>()[2 * TILE_LENGTH];      // [2*TILE_LENGTH,3*TILE_LENGTH)
// Three buffer completely non-overlapping
```

**CalcBuf is a common means to lift BUFER_NUM**: many unary kernel uses 2-3 fload buffer, but can actually reset (compute over it), crushing 3 buffer into 2 bytes of 1 ×tile ×4, thus lifting BUFER_NUM from 2 to 3. This is the direction of swi_glu/ softmax type reduce kernel that can really make perf.

## 6. Cube + Victor Common Kernel Additional Budget

With Cube's kernel, you're gonna have to pull the LooA/L0B/L0C:

| Buffer | Capacity (910B3) |
|---|---|
| L0A | 64 KB |
| L0B | 64 KB |
| L0C | 128 KB |
| L1  | 512 KB |

Vector is still partially away from the UB, but note that when Cube is finished, it will use `SetFlag<HardEvent::M_V>` / `WaitFlag<HardEvent::M_V>` pairs between Vector. Cube writes L0C results**that will not automatically enter the UB**, requiring fixpipe or DataCopy.

## 7. UB Account Template for Optimizing

Optimization of performance does three things at the same time: increase the tile, add queue depth and add calcbuf. The three are not independent and must be evaluated in the same account book.

```text
Enter Queue:    input_count  × BUFFER_NUM × TILE × sizeof(input_dtype)
Queue Output:    output_count × BUFFER_NUM × TILE × sizeof(output_dtype)
fp32 scratch: scratch_f32 × TILE × 4
half scratch: scratch_h16 × TILE × 2
reduce tmp:   reduce_tmp_bytes
Constant/Small watch.:    const_bytes
headroom:     At least. 20% UB
```

Example: a used action, fp16 input output, 3 fp32 scratch:

```text
BUFFER_NUM=2, TILE=4096:
in/out queue = 2 × 2 × 4096 × 2 = 32 KB
fp32 scratch = 3 × 4096 × 4 = 48 KB
total        = 80 KB  // Clear.

BUFFER_NUM=3, TILE=8192:
in/out queue = 2 × 3 × 8192 × 2 = 96 KB
fp32 scratch = 3 × 8192 × 4 = 96 KB
total        = 192 KB // 910B3 None headroomActual high-risk
```

If you want to bring `BUFFER_NUM` from 2 to 3, the common practice is not hard lifting, but compression first:

```cpp
// Discrepancies: Three scratchs at the same time.
auto a = calc.Get<float>();
auto b = calc.Get<float>()[TILE];
auto c = calc.Get<float>()[2 * TILE];

// Good: Make c reuse a space after confirming a result is no longer in use.
auto a = calc.Get<float>();
auto b = calc.Get<float>()[TILE];
auto c = a;
```

per-dtype file should be accounted for independently. fp32 path may not have Cast Scratch, tile can be larger; fp16/bf16 path, although input is smaller, often requires fp32 scratch, not necessarily double.

```cpp
if (dtype == DTYPE_FLOAT) {
  tile = 8192;   // None fp32 cast scratch
} else {
  tile = 6144;   // Enter small, but scratch More
}
```

Multi-line batches also need to be accounted for: `rowsPerTile × rowLength × dtypeSize` must accommodate both input, output, interim statutes and pedding. If a combination results in UB over 80%, priority is given to reducing the number of batches instead of cutting off the necessary accuracy buffer.

## 8. UB OOB field diagnostics

When `errno 507035 vector core exception` + `errorStr: VEC instruction error: the ub address out of bounds` appears:

1. **Precalculated book**: Add all `InitBuffer` bytes of the current Kernel. More than 80% of the UB capacity will essentially collapse.
2. **Check the combination of §3 tables**: Did you fit `BUFFER_NUM=3` with `TILE_LENGTH=8192 + fp16` + multiple calcBuf?
3. **Check sub-view offset**: see [[ascendc-localtensor-subviews]], runtime offset sub-view will be reported in the same errno.
4. **Add print no**: device end `printf` unstable under multiple cores. Replace with the previous elements of suspected LocalTensor `DataCopy` back to GMEM, host end dump.
5. **Diphthalmic location**: Half the note goes to see if it's still falling; step by step back to the intrinsic.

## 9. Don't do anything.

- Don't "keep all the buffer to maximum, run through and compress" -- you can't get back if you can't run, you should start with the smallest Buffer_NUM=1, the smallest Tile.
- Do not tune `aclrtMalloc` in Kernel - UB is distributed staticly, runtime distribution disrupts the pipe sync schedule.
- Do not assume that "edited = UB is not super" - compiler only checks if the sum of `InitBuffer` is superphysical UB (and the path of the pair view and tail is almost undetected), the actual OOB often reports runtime.
