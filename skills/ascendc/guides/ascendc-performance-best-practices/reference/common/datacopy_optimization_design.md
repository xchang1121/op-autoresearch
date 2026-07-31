# DataCopy Optimization Optimization

## 1. Optimization of objectives

DataCopy is the most basic and frequent operation in Ascend C operator. Naive achieves the most simple, block-by-block movement that does not take full advantage of the bandwidth potential of the hardware DMA engine, resulting in:

- **Non-reciprocal access penalty**: DMA efficiency declined significantly when data were not aligned to device ' s optimal particle level. 32B ' s bandwidth aligned, for example, as measured in Ascend 910B, is approximately 70% of 512B alignment, and the best matching values of device are subject to validation.
- **Small loads of multiple loads**: each iterative cycle is independent of DataCopy, DMA start-up cost accumulation.
- **Format conversion extra costs**: ND→NZ conversion requires an independent TransData operator, plus one L1 reading and writing round.
- **Ineffective non-continuous memory access**: Scatter/Gather scenario element-by-fact access cannot take advantage of SIMD parallelity.
- **Memory access mode is not optimized**: step-by-step access leads to a large number of small transfers (< 256B), and DMA set-up costs dominate.

This optimization maximizes the efficiency of data removal by automatically aligning DataCopyPad, conversion of ND2NZ integration formats, Scatter/Gather vectorification, batch DataCopy consolidation, and GM alignment to device optimal particle size.

| Indicators | naive | optimized | Proceeds |
|------|-------|-----------|------|
| Alignment Access bandwidth | Unmatched by optimal particle size: ~70% | Press device optimal particle size | bandwidth up to 30% (Ascend 910B measured) |
| DMA startup times | LoopCount | 1 (volume consolidation) | Significant reduction in command launch costs. |
| ND→NZ Conversion | Independent TransData operator | Integration completed during removal | Save one L1 reading and writing round trip. |
| Scatter/Gather | GetValue/SetValue | vectorization Gather/Scatter Command | 5-20 times the performance |
| Extracts from non-continuous columns | Crop after moving all data | blockLen+srcStride | Reduction of invalid data handling |

> The operator family applies: the `conversion` family has all variants involved in data handling, format conversion, memory layout reordering, such as `transpose`, `concat`, `split`, `gather`, `scatter`, `nd2nz`, etc.

## 2. Overview of the structure

### 2.1 Storage tiers and data flows

DataCopyOptimizing overwrite data fromGM to UB/L1  All the way through  DataCopyPadAccomplish automatic alignment of non-matched data;byNd2NzParamsCombining during removalND→NZformatting conversions;throughDataCopyExtParams of blockLen+srcStrideUnsequencing withdrawals achieved; reduced through bulk consolidationDMAnumber of start-ups;throughGather/ScatterOffset table achievedvectorDisconnected visits;by pressingdeviceBest particle size.GMAlignment (e.g.Ascend 910B as 512B) MaximizebandwidthUtilization factor.

### 2.2 Optimizing the matrix of strategies

| scene | Optimizing Policy | Core API/ Parameters |
|------|---------|----------------|
| Non-recognised data handling | DataCopyPad Autofill | `DataCopyPadExtParams{isPad, leftPadding, rightPadding, paddingValue}` |
| ND→NZ format conversion | Integration on removal | `Nd2NzParams{nValue, dValue, srcDValue, dstNzC0Stride, dstNzNStride}` |
| Extracts from non-continuous columns | blockLen + srcStride | `DataCopyExtParams{blockCount, blockLen, srcStride, dstStride}` |
| Batch Output Merge | Cyclical build-up, collating back in the cycle | Accumulation to UB, once after the cycle, DataCopy |
| vector Gather | Expected Offset Table + Gather Command | `Gather(dst, src, offsetTable, 0, count)` |
| vector Scatter | Index Count + DataCopyPad | `inputStride0_ / inputStride1_` Uncontinuing Address |
| Maximise GM bandwidth | Align with the optimal particle size of device (e. g. 512B) | `AlignUp(offset, 512)` |

### 2.3 DMA efficiency threshold

The following threshold values are based on the Ascend 910B empirical experience and vary in the size of the device DMA controller, bus width and cache rows, with specific values to be confirmed by reference to the corresponding hardware manual or measurements.

| Transfer Size | Efficiency | Annotations |
|---------|------|------|
| < 32 bytes | Extremely Low | Alignment costs are dominant. |
| 32-256 bytes | Bad | DMA setup costs are significant |
| 256-4096 bytes | Medium | Most of the scenes are acceptable. |
| > 4096 bytes | Okay. | Exploited bus bandwidth |
| > 65536 bytes | Excellent. | Close to the peak. |

## 3. Key Parameter Configuration

```cpp
// DataCopyExtParams Structure (Multi-dimensional Bulk Transfer)
struct DataCopyExtParams {
    uint16_t blockCount;   // Number of blocks
    uint32_t blockLen;     // Number of bytes per block
    uint32_t srcStride;    // Spacing between source address blocks (bytes)
    uint32_t dstStride;    // End address block step (bytes)
    uint32_t reserved;     // Reservations
};

// DataCopyPadExtParams Structure (autofill)
template <typename T>
struct DataCopyPadExtParams {
    bool isPad;            // Whether to enable fill
    uint8_t leftPadding;   // Left Fill Bytes
    uint8_t rightPadding;  // Right Fill bytes (maximum) 255)
    T paddingValue;        // Fill Value
};

// Nd2NzParams Structure (ND→NZ conversion)
struct Nd2NzParams {
    uint32_t nValue;       // N Dimensions
    uint32_t dValue;       // D Dimensions
    uint32_t srcDValue;    // Source D Step long.
    uint32_t dstNzC0Stride; // Purpose NZ C0 Dimension length (need to align:fp16 as 16,fp8 as 32)
    uint32_t dstNzNStride; // Purpose NZ N Step long.
};
```

### 3.1 Aligning parameters

| data type | 32B Alignment Elements | 512B alignment elements | Annotations |
|---------|--------------|----------------|------|
| FP32 | 8 | 128 | A multiple of 8 for address/ length |
| FP16/BF16 | 16 | 256 | A multiple of 16 addresses/lengths |
| INT8 | 32 | 512 | Address/length multiple of 32 |

```cpp
// 32B Alignment
uint32_t align32 = (size + 31) / 32 * 32;  // or CeilAlign(size, 32)
// Align with device optimal particle size (e. g. Ascend 910B to 512B)
uint32_t align512 = (offset + 511) / 512 * 512;
```

### 3.2 Padding Limit

- `rightPadding` is `uint8_t` with a maximum of Z255 bytes
- `blockLen` in bytes, with attention to data type size conversion
- After zero data are added to subsequent calculations, it is necessary to ensure that zero does not affect the correctness of algorithms (e.g. ReduceSum scenes to make zero safe, but ReduceMax may be affected)

## 4. Core Calculator Cycle

### 4.1 naive version (before optimization)

```cpp
// Stage 1: Simple DataCopy, unmatched
AscendC::DataCopy(xLocal, xGm[offset], this->tileSize);

// Stage 2: ND→NZ requires independent TransData operator
AscendC::DataCopy(l1Tensor, gmTensor, size);
TransData(l1Tensor, nzTensor, ...);  // Extraoperator

// Phase 3: Independent DataCopy in the cycle
for (int i = 0; i < loopCount; i++) {
    Compute();
    DataCopy(gm[offset], ub, size);  // Every time. 16 An element
}

// Stage 4: element by element Scatter writes
for (int64_t i = 0; i < loadCount; i++) {
    IndexT idx0 = indLocal.GetValue(i * INDICES_LAST_DIM);
    IndexT idx1 = indLocal.GetValue(i * INDICES_LAST_DIM + 1);
    int64_t gmOffset = idx0 * inputStride0_ + idx1 * inputStride1_;
    // Move out line by line, one line at a time
    DataCopy(inputGm_[gmOffset], updLocal[i * updateRowElements_], updateDimSize_);
}

// Stage 5: Element by element Gather Read
for (uint32_t j = 0; j < idxGatherDim; j++) {
    int32_t gatherIdx = idxLocal.GetValue(idxRowOffset + j);
    float value = xLocal.GetValue(xRowOffset + gatherIdx);
    yLocal.SetValue(yRowOffset + j, value);
}

// Stage 6: Uncontinuing column extraction — cropping after removal of all data
DataCopy(fullLocal, gmSrc, fullSize);  // Remove All Data
for (int i = 0; i < rows; i++) {
    ExtractColumn(local[i], fullLocal[i], colStart, colLen);  // UB Upper Crop
}
```

### 4.2 Optimized version (after optimization)

```cpp
// Synchronization point.
uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
dataCopyParams.blockLen = info.s2dealNum;
dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
dataCopyParams.dstStride = 0;
DataCopyPadExtParams<bool> padParams{true, 0,
    static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 0};
DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);

// === Variant B: ND→NZMerge Format Conversions===
template<typename INPUT_T>
__aicore__ inline void CopyToL1Nd2Nz(const LocalTensor<INPUT_T> &l1Tensor,
    const GlobalTensor<INPUT_T> &gmTensor,
    uint32_t nValue, uint32_t dValue, uint32_t srcDValue) {
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.nValue = nValue;
    gm2L1Nd2NzParams.dValue = dValue;
    gm2L1Nd2NzParams.srcDValue = srcDValue;
    gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;  // fp16 Alignment 16
    gm2L1Nd2NzParams.dstNzNStride = 1;
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}

// Synchronization point.
LocalTensor<int32_t> nInt32Out = outputQue2.template AllocTensor<int32_t>();
for (uint32_t i = 0; i < loopCount; i++) {
    DealBmm1ResBaseBlock(info, nInt32Out, ...);  // Additional data copy
}
outputQue2.EnQue(nInt32Out);
outputQue2.DeQue<int32_t>();
uint32_t dealRowCount = (loopCount - 1) * gSplitSize + tailSplitSize;
DataCopy(nUpdateGm[...], nInt32Out, dealRowCount);  // One-time writeback.

// = Variant D: Expected offset table + Garther extract = = =
// Init Phase Projected (executed only once)
for (uint32_t i = 0; i < V1_BASE_T; i++) {
    for (uint32_t j = 0; j < N_; j++)
        preOffsetBuf_.SetValue(offset1++, curOffset * sizeof(P));
    curOffset += N_;
    for (uint32_t j = 0; j < N_; j++)
        postOffsetBuf_.SetValue(offset2++, curOffset * sizeof(P));
    curOffset += nSquare;
}
// Follow-up through Gather extraction
Gather(hPreBuff_, matmulRes_, preOffsetBuf_, 0, lenT * N_);
Gather(hPostBuff_, matmulRes_, postOffsetBuf_, 0, lenT * N_);

// Synchronization point.
DataCopyExtParams copyParams{
    static_cast<uint16_t>(ubFactor),
    static_cast<uint32_t>(RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
    static_cast<uint32_t>(ROPE_LENGTH * sizeof(KV_DTYPE)),
    0, 0};
DataCopyPad(xLocal, kvGm[kvGlobalMemoryOffset], copyParams, padParams);

// === Variant F: GMPressdeviceBest Particle AlignmentbandwidthOptimization===
// The data below are ascend 910B results, and the optimal alignment of particles may differ from device.
uint32_t offset = AlignUp(rawOffset, 512);  // 910B Let's go. 512B The optimal particle size.
DataCopy(ubTensor, gmTensor[offset], dataSize);
// bandwidth's comparison is measured (GM →UB, Ascend 910B):
// Optimal 512B alignment: approximately 100% bandwidth efficiency
// 256B Alignment: ~90% bandwidth Efficiency
// 32B Alignment: ~70% bandwidth Efficiency (worst case)
```

## 5. Key change points from naive to datacopy_optimisation

| Modify Item | (before optimization) | Datacopy_optimization |
|--------|---------------|-------------------------------|
| Non-matched removal | Simple DataCopy | DataCopyPad Autofill Alignment |
| ND→NZ Conversion | Independent TransData operator | Integration on removal Nd2NzParams |
| Cycle Output | Every time it's independent, DataCopy. | Cyclical build-up, collating back in the cycle |
| Scatter Write | SetValue | vectorDataCopyPad |
| Gather Read | GetValue | Expected Offset Table + Gather Command |
| Extracts from non-continuous columns | Handle All Post Crops | blockLen+srcStride |
| GM Address Alignment | No special treatment | Maximize bandwidth by device ' s Best Particle |
| DMA transfer size | Multiple small transfers (< 256B) | Merge to Large Transfer (>4096B) |

## 6. note/ Constraint

1. **DataCopyPad rightPading limit**: `rightPadding` is `uint8_t` with a maximum supplement of Z255 bytes. Beyond this limit, manual segment processing is required.

2. **Security of data added to zero**: data added to zero need to be included in subsequent calculations to ensure that it does not affect the correctness of algorithms. For example, ReduceSum is not affected by zero security (0 does not affect the sum required), but ReduceMax may be affected (0 may change the maximum value).

3. **ND2NZ alignment requirement**: `dstNzC0Stride` needs to be aligned with data type - fp16 is 16 element and fp8 is 32 element. Different data type is not the same as Chiki number, and the conditions are required to be compiled.

4. **DataCopyParams stride is uint16_t**: max. 65535; over-limit to `DataCopyExtParams`.

5. **Scatter writes line by line MTE3 inefficiency**: only one line at a time is suitable for the less number of lines update.

6. **Gather Offset Table UB Occupancy**: The Offset Table occupies UB space (number of elements ×4 bytes), and the number of sub-tensors is significantly high. Offset must be bytes off and Gather requires continuous source data in UB.

7. **GM aligning with device maximum particle size**: Kernel input (including Workspace/Tiling) addresses are normally guaranteed, and developers need to be concerned about whether offsets maintain the device ' s best aligned particle size (e.g. Ascend 910B is 512B, other device needs to be consulted or confirmed).

8. **DMA efficiency threshold**: DMA set-up costs are significant at transfer size < 256B; > 4096B bus bandwidth is fully utilized. Avoid excessive tilling resulting in each DMA transfer being too small. Specific thresholds vary depending on the properties of the device DMA controller, reference is required to the corresponding hardware manual.

9. **Column extracts srcStride and blockLen must satisfy 32B alignment**: each sub-field extracted requires multiple move calls.

10. **UB Space Management for DataCopy**: Additional UB space is required to accumulate in the cycle, ensuring total occupancy is less than UB capacity. Additional judgement logic may be required for the first iterative period.

## 7. common issue and Solutions

### Q1: What about the DataCopyPad pedding value?

```cpp
// General scenario: zero
DataCopyPadExtParams<T> padParams{true, 0, rightPadding, 0};

// scene that requires a specific filling value (e. g. mask fill)
DataCopyPadExtParams<T> padParams{true, 0, rightPadding, MASK_VALUE};

// Do not enable filling (data aligned)
DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
```

### Q2: ND2NZ is different from data type's paired base number when converted?

| data type | C0 Qiquis | Formula |
|---------|-----------|---------|
| FP16 | 16 Elements | `(nValue + 15) >> 4 << 4` |
| FP8 | 32 Elements | `(nValue + 31) >> 5 << 5` |

### Q3: How does DataCopy deal with tailings when they are merged?

```cpp
uint32_t dealRowCount = (loopCount - 1) * gSplitSize + tailSplitSize;
DataCopy(nUpdateGm[...], nInt32Out, dealRowCount);
```

The size of the tail block `tailSplitSize` is usually smaller than the standard block `gSplitSize`, which is to be calculated and passed on to the host side Tiling.

### Q4: How does the Garther Offset Table project?

The offset table is calculated only once at the `Init` stage and is reused later:
```cpp
// Init Phase
for (uint32_t i = 0; i < V1_BASE_T; i++) {
    for (uint32_t j = 0; j < N_; j++)
        preOffsetBuf_.SetValue(offset1++, curOffset * sizeof(P));
    curOffset += N_;
    // ...
}
// Profess phase (each iterative)
Gather(hPreBuff_, matmulRes_, preOffsetBuf_, 0, lenT * N_);
```

### Q5: How to diagnose DMA efficiency?

Profiling focuses on the following indicators:
- `aiv_mte2_time` / `aiv_mte3_time` abnormally high → check alignment and transfer size
- bandwidth has low utilization → to check for large small transmissions (< 256B)
- A large number of DMA operations → check if they can be combined into batch transfers

## 8. Selective decision-making and self-check list

### 8.1 Selective decision-making

```
if (operatorRelated to data handling or format conversion):
    → Enable datacopy_optimization

    if (Data not aligned 32B):
        → Use DataCopyPad Autofill

    if (Yes. ND→NZ Format Conversion):
        → Use Nd2NzParams Merge conversion on removal

    if (Multiple times in the cycle DataCopy Write back):
        → Merge to Batch DataCopy,Circle Unised Back

    if (Need indexed itemstensor):
        → Projected Offset Table + Gather Command

    if (Non-continuous column extraction required):
        → Use blockLen + srcStride One extraction.

    if (Multi-nuclear parallel visits GM):
        → Make sure you press it.deviceOptimal particle alignment, staggered access or line splits if necessary

    if (Transfer Size < 256B):
        → Consider combining transfers or adjustments tiling Policy
else:
    → Standard DataCopy That's fine.
```

### 8.2 Self-check List

- [ ] All data handlers meet 32B alignment, using DataCopyPad when not aligned
- [ ] GM address offset to keep device optimally aligned particle size (e. g. 512B) to maximize bandwidth. Different device needs to be verified
- [ ] ND→NZ conversion using `Nd2NzParams`, `dstNzC0Stride` to data type
- [ ] DataCopy in the cycle multiple times has been merged into a batch output
- [ ] Garther Offset Table is projected at the Init stage, re-used at the Procss phase
- [ ] Scatter evaluate MTE3 efficiency when writing line by line and consider optimization when lines are numbered
- [ ] `rightPadding` ≤ 255 bytes for DataCopyPad
- [ ] Zero data does not affect the correctness of algorithms.
- [ ] Stride ≤ 65535 for `DataCopyParams`, ultra-restrictive use of `DataCopyExtParams`
- [ ] Transfer size > 256B (example as Ascend 910B) avoids a large number of small DMA transfers, which may differ from the device threshold
- [ ] UB Space budget is adequate and bulk is built up within capacity
- [ ] accuracy Validation: achieves data consistency in comparison with naive 100%
