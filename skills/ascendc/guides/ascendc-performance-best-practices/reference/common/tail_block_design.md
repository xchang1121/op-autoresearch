# Tail Block / Data Design Optimization

## 1. Optimization of objectives

The input of data in operator such as Control, Poping, Gather, etc. often cannot be separated by core numbers or tile size, producing tail block (tail block). Naive achieves a fixed tile processing that leads to cross-border access, data competition, or load imbalance.

This optimization ensures a near-perfect load balance and proper border handling in all scenarios through double-track calculations, vector tailings processing, multi-dimensional Tiling tailings separation, etc.

| Indicators | naive | optimized | Proceeds |
|------|-------|-----------|------|
| End block processing | It's a simple cut. It could cross the border. | Double-track nuclear separation | Avoiding data competition and cross-border access |
| End block calculation | scalar Looping | GatherMask vector processing | One-time processing of 64 elements, increasing 4-8 times |
| Multiple load | Tail nucleotage or overload | Dynamic CostLimit + Tolerance Mechanism | Improved performance of non-completed scenes 10-30% |
| Memory Alignment | Unconsidered Alignment | 32B/64B Alignment + DataCopyPad | Avoid loss of non-reciprocal access performance |


## 2. Overview of the structure

### 2.1 Storage tiers and data flows

Tail Block processed data streams: The input data was moved from MTE2 to UB/L1 by MTE2 and divided into normal blocks (normBlock) and tails (tailBlock). Regular nuclear processing: 100% of `defaultUbFactor`'s data, tail processing of the remaining data. The tailings are filled with GatherMask vector or DataCopyPad, resulting in MTE3 writing back to GM, multi-nuclear competition scenes using AtomicAdd to avoid data competition.

### 2.2 Two-track computing model

The host side divides the data into normal core (normCore) and tail core (tailCore):
- **Regular nuclei**: volume of data processed multiple times `defaultUbFactor`
- **tail core**: processing of surplus data volume, separate calculation of number and size of tailing block cycles

KernelSide by side`blockIdx_`To determine whether they are normal or tailings, choose the corresponding cycling parameters.

### 2.3 Event Synchronization Model

| Event type | Meaning | Purpose |
|---------|------|------|
| `MTE2_V` | MTE2 Move complete → allows Victor to read | Main data file readiness |
| `V_MTE3` | Victor complete → to allow MTE3 to write back | End block calculation completed, writeable GM |
| `PIPE_V` | Victor Pipe Barrier | Data dependency between vector commands |

## 3. Key Parameter Configuration

```cpp
// Host side TilingData
struct TailBlockTiling {
    uint32_t normBlockLoop;           // Number of normal nuclear cycles
    uint32_t normBlockTailLoopSize;   // Normal nuclear final cycle size
    uint32_t tailBlockLoop;           // Number of tailings cycle
    uint32_t tailBlockTailLoopSize;   // Final Cyclical Size
    uint32_t defaultValueUsedCoreNum; // Normal nuclei
    uint32_t defaultUbFactor;         // Default UB Process Particle Degree
};
```

### 3.1 Tile Size Selection Principle

| Parameters | Typical value | Annotations |
|------|--------|------|
| `defaultUbFactor` | 64 / 128 / 256 | vector treatment particle size to align Vector unit width |
| `alignCoef` | 32B / 64B | data type-related alignment coefficient (FP16 = 32B, FP32 = 64B) |

**Ascend binding**:

- Global Memoory Visit requires 32B alignment
- Victor command optimal access particle size of 32B (FP16) or 64B (FP32)
- Cube Matrix Multiplication requires 16x16 or 32x32 alignment

```cpp
// Alignment
uint32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);  // 32B / sizeof(T)
uint32_t alignedSize = (size + numPerBlock - 1) / numPerBlock * numPerBlock;
```

### 3.2 Memory budget

End block processing requires additional pattern buffer (for GatherMask) and index buffer:
- `tmpPattern` Buffer: about 64B (uint32_t/uint16_tpattern)
- `indexBuf` Buffer: 6 Index × tile sizes (for use in daaptive popling)

## 4. Core Calculator Cycle

### 4.1 naive version (before optimization)

```cpp
// Fixed file processing, no tail block special processing
for (uint32_t i = 0; i < this->innerLoops; i++) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
// Problem: The last file may cross the border; the tail load is uneven
```

### 4.2 Optimized version (after optimization): Two-track tail block processing

```cpp
// Host side: calculation of nuclear parameters norm/tail
normBlockLoop_ = Ops::Base::CeilDiv(normCoreHandleDefaultValues_, defaultUbFactor_);
normBlockTailLoopSize_ = normCoreHandleDefaultValues_ - defaultUbFactor_ * (normBlockLoop_ - 1);
tailBlockLoop_ = Ops::Base::CeilDiv(tailCoreHandleDefaultValues, defaultUbFactor_);
tailBlockTailLoopSize_ = tailCoreHandleDefaultValues - defaultUbFactor_ * (tailBlockLoop_ - 1);

// Kernel side: Select parameters according to blockIdx
loop_ = tilingData_.normBlockLoop;
tailLoopSize_ = tilingData_.normBlockTailLoopSize;
if (blockIdx_ == tilingData_.defaultValueUsedCoreNum - 1) {
    loop_ = tilingData_.tailBlockLoop;
    tailLoopSize_ = tilingData_.tailBlockTailLoopSize;
}

// Main cycle
for (uint32_t i = 0; i < loop_; i++) {
    bool isLastLoop = (i == loop_ - 1);
    uint32_t currentTileSize = isLastLoop ? tailLoopSize_ : defaultUbFactor_;

    // Use DataCopyPad to process non-reciprocated tailings
    if (isLastLoop && currentTileSize != defaultUbFactor_) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentTileSize * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM[...], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM[...], currentTileSize);
    }

    Compute(dataLocal, currentTileSize);
    CopyOut(dataLocal, currentTileSize);
}
```

### 4.3 GatherMask vector tailings processing

```cpp
// Reorder data with GatherMask for the end block of the last output point
if constexpr (std::is_same_v<T, float>) {
    LocalTensor<uint32_t> bufPattern = tmpPattern.Get<uint32_t>();
    int32_t preLeftShift = numPerBlock + lastLeftShift;
    bufPattern.SetValue(0, (1u << preLeftShift) - (1u << lastLeftShift));
    GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset],
               bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
}
// Avoid tomic operation costs, correct tail block data layout
```

### 4.4 Atom Operating Processing of Non-System Data

```cpp
// Use AtomicAdd to avoid data competition for data that are not 32B alignment
uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
uint64_t mask[2] = {mask0, 0};
Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
SetAtomicAdd<T>();
DataCopy(outputGlobal[offset], outputLocal, cTailAlign);
SetAtomicNone();
```

## 5. Key change points from live to tail_block

| Modify Item | (before optimization) | tail_block (optimized) |
|--------|---------------|---------------------|
| Nuclear Type | Same amount of data for all nuclear processes | Two-track system: normCore + tailCore |
| End block processing | Simple cut-off or cross-border visits | GatherMask vector / DataCopyPad Alignment |
| Data competition | Unprotected. Possible conflict. | AtomicAdd/ Boundary Skip |
| Load Balance | Fixed distribution | Dynamic CostLimit + Tolerance Mechanism |
| Memory Alignment | Not considered | 32B/64B Alignment + padding |
| Indexing | runtime division/mixing | IndexBuffer is expected to be reused |

## 6. note/ Constraint

1. **Pattler type of GateMask**. Float type is uint32_tpatern, half/bfloat16_t uint16_tpatern.

2. **AtomicAdd performance costs**. While ensuring correctness, atomic operations have a qualitative loss, preferential use of GatherMask to avoid competition.

3. **The tail block size must be > 0**. If the tail block size is 0, the nuclear should return directly to avoid empty circulation.

4. **End block of multi-dimensional Tiling**. Each dimension (Channel, Spatial, Batch) is independent in calculating the end block parameters.

5. **Overlap testing**. The potential overlap of Kernel in Adaptive Pooling requires the detection and setting of `isOverLap` markers, using atomic add to ensure correctness.

6. **The correctness of the data to be fully filled**. The SkipPad policy needs to ensure that zero of the padding area is not miscalculated in subsequent calculations (e.g. variance).

7. **32B Alignment is a mandatory requirement**. Unmatched GM visits may result in hardware anomalies or reduced performance.

## 7. Implement common issue and Solutions

### Q1: The amount of tailings processed is much smaller than the normal core, resulting in the tailings being completed soon but waiting for the normal core
**A**: Use tolerance mechanism to allow a slight overload to reduce debris. `costLimit` dynamic calculated as "the total remaining cost / the remaining nuclear number".

### Q2: Pattern calculator error for GatherMask leads to data reordering anomalies
**A**: Ensure that `preLeftShift` and `lastLeftShift` are correctly calculated and that the pattern value is `(1u << preLeftShift) - (1u << lastLeftShift)`.

### Q3: DataCopyPad 's pedding value subsequent calculation
**A**:Use`SkipPadSubMean`Waiting for strategy to skippaddingarea, or used before computing`Duplicate`FillpaddingArea is neutral (if)0 or MASK_VALUE).

### Q4: Uncertainty of results due to the atoms at the end of the multi-nuclear scenario
**A**: Using the quartile mode, reverse processing and detection of border index conflicts, skipping the conflict index is processed by the previous nuclear.

## 8. Example of scene

### 8.1 Examples of each operator family

**Elementwise**`[1000, 128]`, fileRows=256, tails 232, line 1:

```cpp
uint32_t loopCount = (rows + tileRows - 1) / tileRows;
for (uint32_t loop = 0; loop < loopCount; loop++) {
    uint32_t curRows = (loop == loopCount - 1)
                           ? (rows - loop * tileRows) : tileRows;
    uint32_t curLen = curRows * cols;
    DataCopy(xLocal, xGm[loop * tileRows * cols], curLen);
    Adds(yLocal, xLocal, 1.0f, curLen);
    DataCopy(yGm[loop * tileRows * cols], yLocal, curLen);
}
```

**MatMul**`M=1023`, blockM=128, tail 127.

```cpp
// Host: Two-track subnuclei
uint32_t mPerCore = CeilDiv(M, coreNum);
// Kernel: Define M range based on blockIdx
uint32_t localM = (blockIdx < coreNum - 1) ? mPerCore : M - blockIdx * mPerCore;
uint32_t loopCount = (localM + blockM - 1) / blockM;
for (uint32_t loop = 0; loop < loopCount; loop++) {
    uint32_t curM = (loop == loopCount - 1) ? (localM - loop * blockM) : blockM;
}
```

**FlashAttention**`seqLen=1025`, fileSeq=1024, line 1, policy 1+3:

```cpp
uint32_t kvLoopCount = (kvSeqLen + kvTileSize - 1) / kvTileSize;
for (uint32_t kvLoop = 0; kvLoop < kvLoopCount; kvLoop++) {
    uint32_t curKvLen = (kvLoop == kvLoopCount - 1)
                            ? (kvSeqLen - kvLoop * kvTileSize) : kvTileSize;
    if (curKvLen % 16 != 0) {  // Cube Input needs alignment
        DataCopyPad(kLocal, kGm[...], copyParams, padParams);
    }
    if (isCausal && qLoop == kvLoopCount - 1) {
        ApplyCausalMaskPartial(..., curKvLen);  // End block causal mask
    }
}
```

---

## 9. Selective decision-making and self-check list

### 9.1 Selective decision-making

```
if (operatorInclude iterative loops && The amount of data can't be used. tile Size Division):
    → Enable tail_block Optimization
    → Calculate normBlockLoop / tailBlockLoop
    → End block use DataCopyPad or GatherMask
    → Multi-nuclear scenario use AtomicAdd or certainty model
else:
    → Standard tile Just take care of it.
```

### 9.2 Self-check List

- [ ] Host Side correctly calculates / tailBlockLoop / normBlockTailLoopSize / tailBlockTailLopSize
- [ ] Kernel side with blockIdx_right selection of norm/tail parameters
- [ ] Back directly when tail block size is 0
- [ ] DataCopyPad Arguments Correct (blockCount, blockLen, srcStTriide, dstStride)
- [ ] GatherMask Pattern type matches data type (faat→uint32_t, half→uint16_t)
- [ ] 32B/64B Alignment constraints satisfied
- [ ] AtomicAdd in SetAtomicAdd / SetAtomicNone pair
- [ ] Correct reverse processing and border detection under certainty mode
- [ ] accuracy Validation: achieves comparison with naive, error < 1e-5(FP32) or < 1e-3(FP16)
