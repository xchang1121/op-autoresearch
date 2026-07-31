# Softmax state Buffer cross-cycle resident optimisation design

## 1. Optimization of objectives

In scenarios such as Flash Attention / Sparse Flash Attention, online softmax needs to accumulate `softmaxMax`, `softmaxSum`, `softmaxExp` in multiple cycles in the direction of S2. give time to redistribute/ release these UB Buffer per S2 cycle and introduce unnecessary `PipeBarrier` and `InitBuffer` expenses.

This optimization will re-use double buffering through the `loop % preLoadNum` index in three states buffer**resident in UB**after one-time distribution, avoiding double-allocation costs per cycle S2 while supporting reading and writing parallel running water.

**Source operators**: `ai_infra_sparse_flash_attention_gqa`, `ai_infra_fused_infer_attention_sink`

| Indicators | naive | optimized | Proceeds |
|------|-------|-----------|------|
| InitBuffer Call in S2 Cycle | 3 times per round | 0 times | Elimination of duplicate distribution costs |
| PipeBarrier Number | Number of rounds per round | Significant reduction | Reduce the number of events waiting to synchronize |
| UB Space utilization | Low (resulting debris from frequent distribution) | High (pre-allocation continuous layout) | More controllable memory budget |
| Stream Overlay | Low (single-buffer, has to wait until MTE3 is finished to start the next round) | High (double buffering supports reading and writing parallels) | Supporting pingpong running water |

> operator applies: `softmax` (with variations of `softmax`, `log_softmax`, `softmax_cross_entropy` etc.) and `flash_attention`, `sparse_flash_attention` embedded online softmax.

## 2. Overview of the structure

### 2.1 Storage tiers and data flows

```
GM (Global Memory)
  │
  │ MTE3 (workspace ←→ GM)
  ▼
UB Buffer [softmaxMax_ping | softmaxMax_pong | softmaxSum_ping | softmaxSum_pong | softmaxExp_ping | softmaxExp_pong]
  │
  │ Vector PIPE (SoftmaxFlashV2 / online softmax)
  ▼
UB Output → MTE3 → GM
```

### 2.2 Permanent representation in Buffer layout

```
UB Address Space:
[softmaxMaxBuff: SOFTMAX_TMP_BUFFER_SIZE * preLoadNum]
[softmaxSumBuff: SOFTMAX_TMP_BUFFER_SIZE * preLoadNum]
[softmaxExpBuff: SOFTMAX_TMP_BUFFER_SIZE * preLoadNum]

Indexing:
outIdx = loop % preLoadNum
softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T)
```

### 2.3 double buffering Rationale

- **Permanent**: `InitBuffer` is called only once during the initialization phase of operator and three state buffer permanent UB is not released.
- **double buffering Index**: ping/ pong pairs of buffer are used in `loop % preLoadNum` round. When `preLoadNum = 2` is used, Victor PIPE processes round N data while MTE3 can write the last round of results to N+1 round buffer to read and write in parallel.

### 2.4 Event Synchronization Model

| Event type | Meaning | Purpose |
|---------|------|------|
| `V_MTE3` | Victor complete → to allow MTE3 overwrite | Status buffer release control |
| `MTE3_V` | MTE3 complete → to allow Victor to read | Status Buffer Data Ready |

> Note: The specific type of event needs to be adjusted to the actual PIPE configuration, with the core principle being "Vector calculates" and "MTE3 handles" decoupling via double buffering.

## 3. Key Parameter Configuration

```cpp
// Host side TilingData (or constInfo)
struct SoftmaxConstInfo {
    uint32_t preLoadNum;           // double bufferingDepth, usually taken 2
    uint32_t softmaxTmpBufferSize; // Single Status buffer Size, usually = 2K(2048 bytes)
};

// UB Buffer definition for Kernel side
TBuf<QuePosition::VECCALC> softmaxMaxBuff;
TBuf<QuePosition::VECCALC> softmaxSumBuff;
TBuf<QuePosition::VECCALC> softmaxExpBuff;

// InitBuffer one-time distribution (constructive function or Init phase)
pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxSumBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxExpBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
```

### 3.1 PreLoadNum Selection Principle

| preLoadNum | UB Occupation | Stream Overlay | Apply scene |
|-----------|---------|-----------|---------|
| 1 | 3 × 2K = 6KB | None (string) | UB is extremely nervous, only in state, no overlapping. |
| **2** | 3 × 2K × 2 = **12KB** | Pingpong double buffering | **Default recommendation**, reading and writing in parallel |
| 3 | 18KB | Three streams of water. | S2 Extreme cycle, MTE3 latency high. |
| 4 | 24KB | Four streams of water. | UB Sufficient and MTE3 considered when bottlenecks |

### 3.2 SOFTMAX_TMP_BUFFER_SIZE Calculations

```
SOFTMAX_TMP_BUFFER_SIZE = row_length * sizeof(COMPUTE_T)

Typical value(s)row_length = 512, COMPUTE_T = float):
SOFTMAX_TMP_BUFFER_SIZE = 512 * 4 = 2048 bytes
```

## 4. Core Calculator Cycle

### 4.1 naive version (before optimization)

```cpp
for (uint32_t s2Loop = 0; s2Loop < s2LoopNum; s2Loop++) {
    // Every round is redistributed -- it's expensive.
    TBuf<QuePosition::VECCALC> softmaxMaxBuff;
    TBuf<QuePosition::VECCALC> softmaxSumBuff;
    TBuf<QuePosition::VECCALC> softmaxExpBuff;
    pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_TMP_BUFFER_SIZE);
    pipe->InitBuffer(softmaxSumBuff, SOFTMAX_TMP_BUFFER_SIZE);
    pipe->InitBuffer(softmaxExpBuff, SOFTMAX_TMP_BUFFER_SIZE);

    LocalTensor<COMPUTE_T> softmaxMaxUb = softmaxMaxBuff.Get<COMPUTE_T>();
    LocalTensor<COMPUTE_T> softmaxSumUb = softmaxSumBuff.Get<COMPUTE_T>();
    LocalTensor<COMPUTE_T> softmaxExpUb = softmaxExpBuff.Get<COMPUTE_T>();

    // Calculatescore = Q_i * K_j^T
    // ..matmul results at mmresub...

    // Softmax Calculator
    SoftmaxFlashV2<...>(mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb,
                        softmaxExpUb, inSumTensor, inMaxTensor, ...);

    // Buffer is automatically released at the end of each round, but the next round is redistributed
}
```

### 4.2 Optimized version (after optimization)

```cpp
// Phase 1: One-time allocation for the Init phase (executed only once)
pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxExpBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxSumBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);

// Retrieving full-standing UB tensor
LocalTensor<COMPUTE_T> softmaxMaxUb = softmaxMaxBuff.Get<COMPUTE_T>();
LocalTensor<COMPUTE_T> softmaxExpUb = softmaxExpBuff.Get<COMPUTE_T>();
LocalTensor<COMPUTE_T> softmaxSumUb = softmaxSumBuff.Get<COMPUTE_T>();

for (uint32_t s2Loop = 0; s2Loop < s2LoopNum; s2Loop++) {
    // double buffering Index
    uint32_t outIdx = s2Loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T);

    // Calculatescore = Q_i * K_j^T
    // ..matmul results at mmresub...

    // Softmax calculates, using a permanent buffer + offset index
    SoftmaxFlashV2<...>(mmResUb, softmaxSumUb[softmaxOutOffset],
        softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, ...);

    // Optional: MTE3 returns this round to GM/workspace, overlapping with the next Vector calculation
    // WaitFlag<V_MTE3>(outIdx); // Waiting for last round of MTE3 to complete
    // MTE3 Removal...
    // SetFlag<MTE3_V>(outIdx); / /notify next round of data ready
}
```

### 4.3 double buffering current water indication (preLoadNum = 2)

```
Timeline →
─────────────────────────────────────────────────────
S2 Loop 0:  [Vector: softmax on ping]  [MTE3: write ping result]
S2 Loop 1:                                [Vector: softmax on pong]  [MTE3: write pong result]
S2 Loop 2:                                                          [Vector: softmax on ping]  ...
```

> When Victor handles Loop N (pong Buffer), MTE3 can write the results of Loop N-1 (ping Buffer) to GM/workspace, eliminating MTE3 waiting.

## 5. Key change points from live to state_resulter

| Modify Item | (before optimization) | State_reident (after optimization) |
|--------|---------------|------------------------|
| Buffer Assign Positions | S2 cycle, each `InitBuffer` | operator Init phase,**one-time distribution** |
| Buffer life cycle | Releases at the end of each round, redistribution of the next round | **Permanent in UB**, end of cycle not released |
| Buffer Number | Single copies (3 copies) | `3 × preLoadNum` copies to support double buffering |
| Buffer Index | Fixed offset 0 | `loop % preLoadNum` Round Index |
| Chile | `buff.Get<T>()` | `buff.Get<T>()[outIdx * size / sizeof(T)]` |
| PipeBarrier Number | Multiple times assigned/released per round | Init phase only once, not in cycle |
| MTE3 Overlap with Victor | None (string) | Yes (double buffering decomposition, optional event sync) |
| UB Budget certainty | Low (distribution fragmentation) | High (total pre-allocation = 3 ×2K ×pre LoadNum) |

## 6. note/ Constraint

1. **UB Spacebar**: Permanent occupation `3 × SOFTMAX_TMP_BUFFER_SIZE × preLoadNum`. In the case of `preLoadNum = 2`, `SOFTMAX_TMP_BUFFER_SIZE = 2048`, the total occupancy of 12KB is 12KB. There is a need to ensure that the total budget of the UB (usually 256KB) has sufficient space for other buffer (e.g. input file, output file, intermediate calculation).

2. **preloadNum cap**: subject to UB volume limits. If `preLoadNum` is too large to cause UB spills, the compilation/fiction phase is reported to be incorrect. It is recommended that the following formulas be verified:
   ```
   totalResident = 3 × SOFTMAX_TMP_BUFFER_SIZE × preLoadNum
   totalResident + otherBuffers ≤ UB_SIZE
   ```

3. **Statusbuffer and Softmax APICompatibility**:`SoftmaxFlashV2`Waiting for advancedAPIRequest for incoming`softmaxMaxUb`,`softmaxSumUb`,`softmaxExpUb` as `LocalTensor`type. Sub-use offset indextensorNeed to ensure type andshapeMatchAPIRequest.

4. **Initialization Zero**: Before the first use of a permanent buffer, it is recommended that the initial value (e.g. max = -inf, sum = 0) be written through `Duplicate` or `DataCopy` to avoid interference with the cumulative logic of residual data online softmax.

5. **Relation to workspace**: in the Flash Attention scenario, online softmax's intermediate m, l, O usually requires GM workspace across S2 file. The permanent solution is**UB internal buffer distribution**without changing the logic of GM workspace.

## 7. Implement common issue and Solutions

### Question 1: UB RAM Spill

**Specific**: Compiled or simulated error, prompting UB distribution failure or spill.

**Reason: `preLoadNum` settings are too large, or `SOFTMAX_TMP_BUFFER_SIZE` computational errors (e.g. failure to match `sizeof(COMPUTE_T)`) result in the total occupancy of the resident buffer exceeding UB capacity.

**Solution**:
- Lower `preLoadNum` (e. g. 3→2 or 2→1)
- Verify whether `SOFTMAX_TMP_BUFFER_SIZE` is aligned with 32B or 64B (ascend hardware alignment requirement)
- Print UB total occupancy using `printf` or debug tool:
  ```cpp
  // Debug Printing
  printf("UB resident size: %u bytes\n",
         3 * SOFTMAX_TMP_BUFFER_SIZE * preLoadNum);
  ```

### Problem 2: softmax result error (value drift)

**Symptom**: accuracy verification failed, softmax output was inconsistent with reference implementation.

**Reason: double buffering index error, e.g. `softmaxOutOffset` calculation is not divided by `sizeof(COMPUTE_T)`, leading to a pointer bias to the wrong position; or `preLoadNum` is 1 and the `% preLoadNum` index is still used (the result is always 0, logically correct but not optimized).

**Solution**:
- Verify offset calculation:
  ```cpp
  // Correct: Distortion by element
  uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T);
  // Error: Direct bytes to LocalTensor
  // uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE; // ❌
  ```
- Buffer:
  ```cpp
  Duplicate(softmaxMaxBuff.Get<COMPUTE_T>(), FLOAT_NEG_INF, 3 * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) * preLoadNum);
  ```

### Question 3: preloadNum = 2 but no performance enhancement

**phenomena**: cycle numbers are close to the naive version and double buffering is not valid.

**Reason: Only buffer is resident, but no synchronisation of events between Victor and MTE3 achieves a true flow overlap. `SoftmaxFlashV2` calls and closes back to MTE3 and is still a serial execution.

**Solution**: Insert `SetFlag` / `WaitFlag` event pair, write MTE3 back to the next round of Victor calculation decomposition:
```cpp
// The right water synchronisation pattern.
for (uint32_t s2Loop = 0; s2Loop < s2LoopNum; s2Loop++) {
    uint32_t curIdx = s2Loop % preLoadNum;
    uint32_t prevIdx = (s2Loop + preLoadNum - 1) % preLoadNum;

    // Waiting for the last round of MTE3 to write back (release current buffer)
    WaitFlag<MTE3_V>(curIdx);

    // Victor Calculator
    SoftmaxFlashV2<...>(...);

    // Notify MTE3 to write back to the current buffer
    SetFlag<V_MTE3>(curIdx);

    // MTE3 Interstep back to the previous round (overlapsing with the next Vector calculation)
    // Note: In the actual code, MTE3 wrote responses before the next round of WaterFlag after SetFlag
}
```

### Question 4: Cumulative logical conflict with m/sum online softmax

**System**: In a multitile (S2 loop) scenario, softmax results are inconsistent across the tile.

**Reason: Online softmax requests to keep running max and running sum across the file. If each S2 cycle uses an independent buffer (double buffering), but does not correctly pass the previous state to the next round (e.g. through GM workspace), it will break the state.

**Solution**:
- Make a clear distinction between "UB internal state buffer" (this optimization is used for single-wheel softmax calculations) and "cross-file cumulative" (transmitted through GM workspace, online softmax algorithm itself).
- This optimization does not change the logic of the cross-tile status transfer online softmax, only optimises the UB Buffer distribution policy in a single file.

### Summary of issues

| # | Problem | Gene. | Solutions | Impact |
|---|------|------|---------|------|
| 1 | UB Spill | preLoadNum Too Large or size not aligned | Reduce preLoadNum, Verify Alignment | Failed to compile/ imitate |
| 2 | accuracy failed | Offset calculation error or buffer not initialized | Validate infset formulae, start cycle zero | Result error |
| 3 | Performance not enhanced | Missing synchronisation, still serial | Insert SetFlag/WaitFlag | Performance has not improved. |
| 4 | Inconsistencies across the file | Convey UB Optimization and Online Softmax Status Transfer | Keep GM workspace cumulative logic unchanged | accuracy failed |

## 8. Checklist of measured performances, overlaps and self-checks

### 8.1 Additional relationships with other optimizations

| Optimization | Collapse Feasibility | Annotations |
|------|-----------|------|
| **pingpong double buffering** | ✅ highly compatible | This optimization is the specific application of pingpong on softmax state buffer, `preLoadNum` for pingpong depth |
| **online softmax algorithm** | ✅ necessary prefix | Status-based optimization presupposes the use of online softmax, without both |
| **FP32 Intermediate calculation** | ✅ compatibility | `COMPUTE_T` for the status buffer can be set as float, co-examining with FP32 numerical stability |
| **mte2_preload** | ⚠ ️ Partial Compatibility | Make sure that the pre-buffer does not conflict with the permanent-based Buffer if there is an MTE2 preset before softmax |
| **swat / streamk** | ❌ Not applicable | These two optimizations are for the MatMul CUBE core, which does not interact directly with the softmax Vector core |

### 8.2 Selective decision-making

```
if (operatorOrganisation online softmax && S2 Number of cycles > 1):
    → Enable state_resident Optimization
    → preLoadNum = 2(Default,UB You can try it when you've got enough. 3)
else:
    → No such optimization is required (one-wheel) softmax No recycling gain)
```

### 8.3 Self-check List

- [ ] `InitBuffer` is called only once during the initialization phase of operator, not in the S2 cycle
- [ ] `preLoadNum` takes the value of 1/2/3/4 and meets the UB budget constraints
- [ ] `softmaxOutOffset` correctly calculated: `outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T)`
- [ ] First cycle pre-buffer initialized (max = -inf, sum = 0)
- [ ] double buffering Index with Event Synchronization (SetFlag/WaitFlag) to achieve true flow of water
- [ ] accuracy Validation Passed (error within 1e-5)
- [ ] cycle number comparison: optimized / naive < 0.9 (at least 10% up, usually 15-25%)
