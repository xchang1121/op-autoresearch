# Victor computing efficiency optimization design

## 1. Optimization of objectives

In operator, a pure or Vector-led power bottleneck of performance often comes from three types of UB internal efficiency losses: Scalar calculates manually for each Vector API recap/mask, repeats the high latency reduce command for the returned scene, and successive Vector computational chains write the intermediate results back to GM and then read back to UB. The optimisation is based on the Counter model of the control reduction directive, the low latency command combination replaces a single high-latency return path, UB data straight through consumption,**the reduction of Scalar sales, Vector directive latency and GM round-tripp**.

| Indicators | naive | optimized | Proceeds |
|------|-------|-----------|------|
| Scalar Control Command | Manual calculation per Victor API | Counter Mode One-time Settings, All-Automated | Scalar reduced costs by 30-50 per cent |
| GM load times (n step Victor chain) | 2n moves (per move + move out) | 2 (first move + final move) | Reduction in the number of moves n times |
| Return Command latency | HoleReduceSum single high latency | BlockReduceSum + WhoReduceSum | latency down 20-40% |
| Command Level Parallel | Cyclical Thread Execution | Looping exposure parallels | ILP Upgrade |


## 2. Overview of the structure

### 2.1 Three types of optimized position in Victor PIPE

The three types of optimization work for the Victor execution path in the Uninuclear UB:

- **Counter mode (entry)**: `SetMaskCount(dataSize)` later, Victor API calls over all data, avoiding manual calculations of `repeatTimes` and `tailSize`.
- **UB integration chain (intermediate)**: Multiple steps from `VECCALC` queue left the intermediate result calculated by Victor within UB, without landing GM, less `2n → 2` moves.
- **Low latency Convention (Exports)**: replace a single high latency Convention path with a combination of `BlockReduceSum + WholeReduceSum`.

### 2.2 Data flow comparisons

| Mode | Data stream | GM Number of Moves |
|------|--------|------------|
| No, no, no, no. | Every step Vector moves the intermediate result out of GM, and the next move in. | `2n` times |
| Optimized (formal) | First move to UB (VECALC), intermediate result chain consumption, final move out | `2` times |

## 3. Key Parameter Configuration

```cpp
// Counter Mode Parameters
struct CounterModeConfig {
    uint32_t totalElements;   // Total number of elements, direct to SetMaskCount
    uint32_t oneRepeatSize;   // Single repeat Process the number of elements (e. g. 128/256)
};

// UB integration chain parameters
struct FusedChainConfig {
    uint32_t chainLength;     // Vector Length of the calculation chain (e.g., 3:Exp→Abs→Mul)
    uint32_t veccalcBufferSize; // VECCALC buffer Size (number of elements)
};

// Low latency condensation parameters
struct LowLatencyReduceConfig {
    uint32_t blockSize;       // BlockReduceSum Size of blocks (e. g.) 64/128)
    uint32_t numBlocks;       // Total block count = totalElements / blockSize
};
```

### 3.1 Principles for the selection of parameters

| Parameters | Typical value | Annotations |
|------|--------|------|
| `oneRepeatSize` | 128 (FP16) / 64 (FP32) | Aligns the width with Victor Unit |
| `chainLength` | 2-5 | Limited to UB capacity, excessive length crowds double buffering space |
| `blockSize` | 64 / 128 | BlockReduceSum is a valid block size |

## 4. Core Calculator Cycle

### 4.1 naive version (before optimization)

**Counter Reverse - Manual repeat/mask:**
```cpp
// Manual calculation for each Victor API
uint32_t repeatTimes = dataSize / ONE_REPEAT_SIZE;
uint32_t tailSize = dataSize % ONE_REPEAT_SIZE;

// Main
Add(dst, src1, src2, FULL_MASK, repeatTimes, {1, 1, 1, 8, 8, 8});

// End block (additional API call + mark settings required)
if (tailSize > 0) {
    SetVectorMask(tailMask);
    Add(dst + offset, src1 + offset, src2 + offset, tailMask, 1, {1, 1, 1, 8, 8, 8});
}
```

**UB Integration Reverse - Every step out of GM:**
```cpp
void Process() {
    // Step 1: Exp
    CopyIn();           // GM → UB
    Compute_Exp();      // UB → UB
    CopyOut();          // UB → GM

    // Step 2: Abs
    CopyIn1();          // GM → UB
    Compute_Abs();      // UB → UB
    CopyOut1();         // UB → GM
}
```

**Retrogression - Single HoleReducesum:**
```cpp
// High latency Single Commands
float sum = WholeReduceSum(src, dataSize);  // latencyIt's high. It's obvious when it's big.
```

### 4.2 Optimized version (after optimization)

**Counter Mode - All Call at One:**
```cpp
// Counter Mode: SetMaskCount Last API Call
SetMaskCount(dataSize);
Add(dst, src1, src2, dataSize);  // Automatically handle the main block+The tail, no manual. mask
SetMaskMode(NORMAL);  // Recovery after completion mask Mode
```

**UB Convergence Chain - Middle result undefeated GM:**
```cpp
// Defines the VECCALC queue for intermediate results
TQue<QuePosition::VECCALC, 1> midQueue;
pipe.InitBuffer(midQueue, 1, dataSize * sizeof(T));

// It's only the first move, the final move.
CopyIn(inLocal, inGm, dataSize);  // GM → UB(VECCALC)

// Chain calculation, intermediate result retains UB
LocalTensor<T> mid1 = midQueue.AllocTensor<T>();
Exp(mid1, inLocal, dataSize);           // Exp

LocalTensor<T> mid2 = midQueue.AllocTensor<T>();
Abs(mid2, mid1, dataSize);              // Abs

LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
Mul(outLocal, mid2, scale, dataSize);   // Mul

CopyOut(outGm, outLocal, dataSize);      // UB → GM
```

**Low latency Convention — BlockReducesum + HoleReducesum group:**
```cpp
// Two-stage contract: BlockReduceSum split, and HoleReduceSum aggregate
// Stage 1: BlockReduceSum, each blockSize element produces a part and
for (uint32_t i = 0; i < numBlocks; i++) {
    blockSums[i] = BlockReduceSum(src + i * blockSize, blockSize);
}
PipeBarrier<PIPE_V>();

// Phase 2: HoleReduceSum Summary and
float totalSum = WholeReduceSum(blockSums, numBlocks);
```

## 5. Key change points from naive to vector_efficity

| Modify Item | (before optimization) | vector_efficacy (after optimization) |
|--------|---------------|---------------------------|
| Victor API Caller | Manual calculation | Counter Mode: SetMaskCount + One Call |
| Life cycle of intermediate outcomes | Every step out of the GM, next step in. | Keep UB/VECCALC, chain straight through consumption |
| Return Directive | HoleReduceSum single high latency | BlockReduceSum + WhoReduceSum |
| GM Number of Moves (n-step chain) | 2 n | 2 times |
| Scalar expenses | High (repeat/mask calculation) | Low (Counter Mode Autoprocessing) |

## 6. note/ Constraint

1. **Counter mode relies on hardware support**: only part of Victor API supports Counter mode, and confirmation of API document is required before use.

2. **Counter mode must be restored mask**: After the Counter mode has been used, `SetMaskMode(NORMAL)` must be called back to avoid affecting subsequent Victor API.

3. **The length of the UB integration chain is subject to UB capacity limitations**: the UB budget needs to be calculated accurately when the middle buffer may crowd into double buffering space when the chainLength is too long or dtype is wider.

4. **BlockReducesum requires temporary buffer**: an additional block Sums buffer is required for the low latency contract combination to ensure that UB has sufficient space.

5. **Cycle is open for interaction with Hardware Loop**: The expanded cycle still needs to meet Hardware Loop conditions (`uint16_t` iterative variable, start 0, step 1) to avoid the introduction of if/else.

6. **End block processing**: The Counter mode automatically handles tail block processing, but there is still a need to retain the tail block processing logic as fallback under the Normal mode.

## 7. Implement common issue and Solutions

| Problem | Gene. | Solutions |
|------|------|---------|
| After the Counter mode, API results are abnormal. | Mask not restored to NORMAL | Call `SetMaskMode(NORMAL)` after each Counter mode |
| UB integration chain UB spill | longLength too long or dtype wide | reducing chainLength or file_size;precise calculation of UB budget |
| BlackReducesum. It didn't work out right. | BlockSize Unmatched | BlockSize needs to align Victor Unit width (e. g. 64/128) |
| It's down after the cycle's spread. | ICache miss due to over-expanding factor | Expand factor ≤4, default recommended 2x |

## 8. Selective decision-making and self-check list

### 8.1 Selective decision-making

```
if (profiling Show Scalar High time ratio):
    → Enable Counter Mode
elif (profiling Show MTE2/MTE3 High percentage of intermediate removal and multiple steps Vector Chain):
    → Enable UB Integration chain(s)VECCALC Straight through)
elif (profiling Show returnlatencyHigh):
    → Enable LowlatencyGroup of Consorts (%2)BlockReduceSum + WholeReduceSum)
else:
    → Standard Vector Just make it happen.
```

### 8.2 Self-check List

- [ ] Restore `SetMaskMode(NORMAL)` at the right location after the Counter mode is used
- [ ] VECCALC Buffer size of the UB integration chain is accurately calculated and does not spill
- [ ] BlockReduceSum alignment of blockSize to Victor Unit width
- [ ] Looping end block correct processing (`repTimes % unrollFactor`)
- [ ] Memory budget ≤ 32 RegTensor
- [ ] Validation pass: achieves the same results compared to the naive
