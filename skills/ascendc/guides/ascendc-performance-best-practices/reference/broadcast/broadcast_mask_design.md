# Broadcast & Mask Operations Optimizing Design

## 1. Optimization of objectives

Broadcast (radio) and Mask (mask) are memory access models for the medium-high frequency (HF) presence of operator. Naive achieves the usual element-by-element scalar processing or runtime conditionality judgement, resulting in:

- **scalar-vector hybrid operation**: scalar input (e.g., learning rate lr) is repeatedly involved in scalar's calculations in the cycle and cannot be used for the entire bandwidth unit.
- **runtime broadcast mode judgement**: BA/AB broadcast mode is processed through runtime conditions branch in Kernel to introduce branch predictions of failure costs.
- **Low accuracy/ Inefficient Mask means**: If operator prototype requires fload mask (0.0/-inf) to occupy 4B/Element; change to bool mask (1B/Element) can reduce by 75% bandwidth. If the prototype is fixed as float, the `faat mask Cast ' should be reused on Kernel side as bool ', the subsequent calculation will still have an RAM efficiency gain.
- **GM Multi-nuclear Address Conflict**: MTE2/MTE3 time was exceptionally high when multi-nuclear parallel visits were made to the same 512B area.

This optimization maximizes the performance and memory efficiency of the broadcast/mask scenario by means of high-level API (SelectWith BytesMask, SoftMax), compilation broadcast mode distribution, scalar Duplicate broadcast, GM address circumvention.

| Indicators | naive | optimized | Proceeds |
|------|-------|-----------|------|
| Mask Memory Occupancy | Float (4B/Element) | Bool (1B/Element), prototype supported or post-Cast reuse in Kernel | bandwidth down 75% |
| scalar Operations | scalar-vector Mixing (Muls) | vector-vector Unified (Duplicate+Mul) | Vector Unit Full bandwidth |
| Radio mode judgement | runtime if-else | Compiled `if constexpr` | Zruntime branch expenses |
| Mask Semantics | Multiplication Mask (accuracy loss) | SelectWithBytesMask (exact replacement) | Clear semantics, accuracy without damage |
| Multiple nuclear GM access | Consequence with 512B regional conflicts | Misplaced access/line splits | MTE2/MTE3 significant reduction in time |

> operator applies to all variants of the `broadcast` family that include mask input, broadcast dimension, condition selection, e.g. `scaled_masked_softmax`, `masked_scatter_with_position`, `apply_adagrad_d`, etc.

## 2. Overview of the structure

### 2.1 Storage tiers and data flows

The data stream of the Broadcast/Mask scene: src and mask moved from MTE2 to UB in UB through high-grade API (Select WithBytesMask, SoftMax) calculations, which were written back by MTE3. Core optimizations include scalar Duplicate radio, BA/AB compilation mode distribution, Maskofset offset management, and GM address conflict circumvention.

### 2.2 Broadcast Mode Classification

| Mode | Description | Typical scene. | Indexing |
|------|------|---------|---------|
| **Scalar Broadcast** | scalar input broadcast as vector | Learning Rate lr, scale et al. scalar | `Duplicate` Filled vector Operations |
| **BA model** | Mask, we're broadcasting in the last few dimensions. | Mask shape [1, S] | `maskIdx = i % xInner` |
| **AB model** | Mask, broadcast in the first few dimensions. | Mask shape [B, 1] | `maskIdx = rowIdx` |
| **Batch Broadcast** | Mask is broadcasting at the bat. | causal mask [1, 1, S, S] | `maskMode |= BROADCAST_BATCH` |
| **Channel Broadcast** | We've got a radio broadcast in Channel. | channel-wise mask | `maskMode |= BROADCAST_CHANNEL` |

### 2.3 Event Synchronization Model

| Event type | Meaning | Purpose |
|---------|------|------|
| `MTE2_V` | MTE2 Move complete → allows Victor to read | src/mask file data ready |
| `V_V` | Victor complete → to allow Victor to continue | SelectWithBytesMask Internal Dependence |
| `V_MTE3` | Victor complete → to allow MTE3 to write back | Calculate finished, writeable GM |

## 3. Key Parameter Configuration

```cpp
// Host side TilingData
struct BroadcastMaskTiling {
    uint32_t patternType;     // PATTERN_AB = 0, PATTERN_BA = 1
    uint64_t maskMode;        // bit0: batch broadcast, bit1: channel broadcast
    uint32_t padLineNum;      // Line width after alignment
    uint32_t alignedMaskWidth; // After alignment mask Width
    uint32_t xInner;          // BA Internal dimensions of the mode
    uint32_t xOuter;          // BA The outer dimensions of the mode
    SoftMaxTiling softmaxTilingData;  // SoftMax High API tiling
};

// Kernel side Maskofset structure
struct MaskOffset {
    uint64_t batchOffset = 0;
    uint64_t channelOffset = 0;
    uint64_t lineOffset = 0;
    __aicore__ inline void NextChannel(uint64_t channelNum) {
        channelOffset = (channelOffset + 1) % channelNum;
        if (channelOffset == 0) batchOffset++;
        lineOffset = 0;
    }
    __aicore__ inline uint64_t GetOffset(uint64_t realBatch, uint64_t realChannel, uint64_t realLine) {
        return batchOffset * realBatch + channelOffset * realChannel + lineOffset * realLine;
    }
};
```

### 3.1 Constraint constraints

| data type | Align Particle Degrees | Annotations |
|---------|---------|------|
| FP32 | 8 Element (32B) | Width needs to be multiple of 8 |
| FP16/BF16 | 16 Element (32B) | Width needs a multiple of 16 |
| bool (mask) | 32 Element (32B) | Mask needs a multiple of 32 width |

```cpp
uint64_t alignedXBlock = AlignedBytes / xDtypeSize;   // 32 / sizeof(T)
uint64_t xPaddingNum = (width % alignedXBlock) ? (alignedXBlock - width % alignedXBlock) : 0;
uint64_t alignedMaskBlock = AlignedBytes / BOOL_SIZE; // 32 / 1 = 32
uint64_t maskPaddingNum = (width % alignedMaskBlock) ? (alignedMaskBlock - width % alignedMaskBlock) : 0;
```

## 4. Core Calculator Cycle

### 4.1 naive version (before optimization)

```cpp
// Phase1:scalar-vectorMixed Operationsscalar lrdirect participation)
lrScalar = lrGm.GetValue(0);
AscendC::Muls(lrMulGradLocal, gradLocal, this->lrScalar, this->tileSize);

// Stage 2: runtime broadcast mode judgement (2D simple processing)
uint32_t maskRow = (this->maskDim0 == 1) ? 0 : row;
uint32_t maskCol = (this->maskDim1 == 1) ? 0 : col;
uint32_t maskIdx = maskRow * this->maskDim1 + maskCol;

// Phase 3: Float Mask Multiplication (accuracy loss, lack of semantics)
AscendC::Muls(scaledLocal, xLocal, scale, tileLength);
AscendC::Add(scaledLocal, scaledLocal, maskLocal, tileLength);  // mask Yes. float Type

// Stage 4: Manual softmax
AscendC::ReduceMax(maxLocal, scaledLocal, sharedLocal, tileLength);
float rowMax = maxLocal.GetValue(0);
AscendC::Duplicate(maxLocal, rowMax, tileLength);
AscendC::Sub(expLocal, scaledLocal, maxLocal, tileLength);
AscendC::Exp(expLocal, expLocal, tileLength);
AscendC::ReduceSum(sumLocal, expLocal, sharedLocal, tileLength);
float rowSum = sumLocal.GetValue(0);
float invSum = 1.0f / rowSum;
AscendC::Muls(outLocal, expLocal, invSum, tileLength);

// Phase 5: MNA access (conflict swathes)
for (int i = 0; i < loopOneCore; i++) {
    DataCopy(dst, src[i * blockSize], blockSize);  // All nuclear visits to the same region
}
```

### 4.2 Optimized version (after optimization)

```cpp
// === Variant A: scalar DuplicateRadio===
// scalar lr first Duplicate for vector, then vector-vector multiplication with grad to avoid the scalar-vector hybrid operation
float lrScalar = lrGm.GetValue(0);
AscendC::Duplicate(lrLocal, lrScalar, tileSize);
AscendC::Mul(dstLocal, gradLocal, lrLocal, tileSize);

// Synchronization point.
if constexpr (PATTERN_TYPE == PATTERN_AB) {
    if (maskGm[rowidx] == true) { /* AB Mode */ }
} else {
    if (maskGm[i % xInner] == true) { /* BA Mode */ }
}

// === Variant C/D/F: SelectWithBytesMaskExactmaskApply===
LocalTensor<uint8_t> maskTmpBuf = this->sharedBuffer.template Get<uint8_t>();
SelectWithBytesMaskShapeInfo shapeInfo;
shapeInfo.firstAxis = this->lineNum;
shapeInfo.srcLastAxis = this->paddedHeadDim_;
shapeInfo.maskLastAxis = this->paddedHeadDim_;
SelectWithBytesMask(tmpOutLocal, tmpOutLocal, MASK_VALUE, maskLocal, maskTmpBuf, shapeInfo);

// Synchronization point.
SoftMaxTiling softmaxTilingData = tilingData.softmaxTilingData;
SoftMaxShapeInfo softmaxShapeInfoData = {
    static_cast<uint32_t>(lines),
    static_cast<uint32_t>(tilingData.padLineNum),
    static_cast<uint32_t>(lines),
    static_cast<uint32_t>(tilingData.width),
};
SoftMax<float, false, false>(dstTensor, srcTensor, sharedBuffer, softmaxTilingData, softmaxShapeInfoData);

// Synchronization point.
struct MaskOffset {
    uint64_t batchOffset = 0;
    uint64_t channelOffset = 0;
    uint64_t lineOffset = 0;
    __aicore__ inline void NextChannel(uint64_t channelNum) {
        channelOffset = (channelOffset + 1) % channelNum;
        if (channelOffset == 0) batchOffset++;
        lineOffset = 0;
    }
    __aicore__ inline uint64_t GetOffset(uint64_t realBatch, uint64_t realChannel, uint64_t realLine) {
        return batchOffset * realBatch + channelOffset * realChannel + lineOffset * realLine;
    }
};

// Synchronization point.
// Misappointment Access
for (int i = 0; i < loopOneCore; i++) {
    int newProgress = (i + GetBlockIdx()) % loopOneCore;
    DataCopy(dst, src[newProgress * blockSize], blockSize);
}
```

## 5. Key change points from live to Broadcast_mask

| Modify Item | (before optimization) | Broadcast_mask (optimized) |
|--------|---------------|------------------------|
| scalar Operations | scalar-vector Mixing (Muls) | Duplicate Broadcast unified vector after vector |
| Radio mode judgement | runtime if-else | Compiled `if constexpr` zero-cost branch |
| data type | Float (4B/Element) | Bool (1B/Element), prototype supported or post-Cast reuse in Kernel |
| Mask application | Multiplication Mask (accuracy loss) | SelectWithBytesMask (exact replacement) |
| Softmax Achieved | Manual ReduceMax+Exp+ReduceSum | SoftMax High Level API (10-20%) |
| Mask Offset Calculator | Simple 2D Index | Maskofset Structure Supportbatch/channel broadcast |
| Multiple nuclear GM access | Serial with Area | Staggered visits or linets to avoid 512B conflicts |

## 6. note/ Constraint

1. **SelectWithBytesMask**: when the corresponding position is true, dst takes value; otherwise src. Mask=true should be replaced with MASK_VAL (e. g. -1000.0), which will become near 0 in softmax.

2. **Broadcast mode recognition completed at the Host end**: `CanBroadcastBAOrAB` functions recognize broadcast mode on the Host side and pass it to Kernel by tiling data `PATTERN_TYPE`. Use template parameters in Kernel to complete the translation period branch.

3. **Alignment bound**: All data handling and Vector calculations must meet 32B alignment. FP32 requires 8 element alignment, FP16/BF16 requires 16 element alignment, Bool Mask requires 32 element alignment. DataCopyPad is automatically filled when not aligned.

4. **Soft Max API Temporary Buffer**: Additional UB space is needed to store temporary data using the Soft Max High Level API. The size required can be checked through `GetSoftMaxMaxTmpSize`, and the shared Buffer is finely used with SelectWith BytesMask.

5. **GM address conflict circumvention**:
   - The conflict was particularly severe when the data was wide ≤ 512B
   - Error access needs to be aligned with `SyncAll` All-nucleic sync
   - Line parts instead of column parts can naturally avoid conflict, but may lead to an uneven end-line load

6. **maskMode field definition**: bit0 for catch radio, bit1 for channel radio. Set watch radio when watch!=maskBatch; set channel when channel!=maskchannel.

7. **accuracy balances performance**: FP32 is used internally to calculate numerical stability even if the input/output is FP16.

## 7. common issue and Solutions

### Q1: What's the difference between SelectWithBytesMask and floatmask (Add)?

This means that you can use `Add(scaledLocal, scaledLocal, maskLocal, tileLength)`, and you need to ask the mask to be a float type with a value of 0/-inf:
- Memory occupancy high (4B/Element)
- Lack of semantic clarity (conditional selection through additions)
- accuracy may be affected by multiplication mask

If the operator prototype is fixed as a float mask, it should be reused on the side of Kernel after using the bool type mask (1B/Element), semantic (conditional selection) and memory occupancy is reduced by 75%.

### Q2: How do we handle the Mask broadcast across the bat/channel?

Manage complex mask offsets using `MaskOffset` structures:
```cpp
MaskOffset offset;
offset.GetOffset(realBatch, realChannel, realLine);  // Calculating the current position mask Offset
offset.NextChannel(channelNum);  // Switch to Next channel
```

The `CopyMaskIn` function handles multiple boundary situations: when the current and end of the bat is the same, it only needs to be processed within a channel; instead of having to cross the watch at the same time.

### Q3: How can the BA/AB mode be downgraded when recognition failed?

If the Hostend cannot be recognized as BA or AB mode (e.g., more complex broadcast mode), it should be downgraded to a generic element-by-component index calculation, or the Mask should be extended in advance at the Hostend to complete shape.

### Q4: How does the GM address conflict be diagnosed?

Profiling, when `aiv_mte2_time` or `aiv_mte3_time` are abnormally high, check:
- Multiple access to the same 512B area
- Whether the width of the data line is ≤ 512B
- Optimization through line split or misplaced access

## 8. Selective decision-making and self-check list

### 8.1 Selective decision-making

```
if (operatorOrganisation mask Enter or broadcast dimensions):
    → Enable broadcast_mask Optimization
    → mask Use bool Type (%1)1B/elements; if prototype is float,Kernel Side Cast as bool Reuse Later
    → scalarInput Passed Duplicate Broadcast asvector
    → Host End Identification BA/AB Broadcast mode, compilation and distribution
    → mask Apply Use SelectWithBytesMask High API
    → softmax Calculate Usage SoftMax High API
    → Multi-nuclear scene examination. GM Address conflict, staggered access or line split if necessary
else:
    → StandardvectorIt's good to run.
```

### 8.2 Self-check List

- [ ] Mask data type is bool (1B/ Element) or Kernel has re-used float Cast as bool
- [ ] scalar input for vector by `Duplicate` broadcast
- [ ] BA/AB broadcast mode recognized at the Hostend, `if constexpr` compile branch in Kernel
- [ ] Mask Apply with `SelectWithBytesMask`, non-multiplier mask or Add
- [ ] Soft Max calculates high-level API, non-manual ReduceMax+Exp+ReduceSum
- [ ] All data handlers meet 32B alignment, using DataCopyPad when not aligned
- [ ] Multi-nuclear scene check for GM address conflicts. Enable error access or line cut while wide ≤ 512B
- [ ] Maskofset correctly manages the watch/channel broadcast offset
- [ ] SoftMax temporary Buffer shares detailed Buffer reuse with selectWithBytesMask
- [ ] accuracy Validation: achieves comparison with naive, error < 1e-5(FP32) or < 1e-3(FP16)
