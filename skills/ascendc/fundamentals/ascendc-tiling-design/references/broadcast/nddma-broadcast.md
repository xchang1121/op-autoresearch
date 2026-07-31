# Broadcast - NDDMA Broadcast Branch (DAV_3510)

> **Applied scene**: multidimensional after-axis, DAV_3510 chip.**Only for GM→UB migration phase**, automatic broadcast via NDDMA hardware stride = 0 configuration, data reach UB as complete post-broadcast file. Not applicable to UB internal broadcasting (UB internal request for [dynamic UB Broadcast] (dynamic-ub-broadcast.md)).
>
> **DAV_2201 does not support NDDMA**, use [UB Broadcast static interface] (ub-broadcast.md).

---

## I. SPECIFIC STATE

| Features | Annotations |
|------|------|
| **Chip request** | DAV_3510(Ascend 950) |
| **Post-axis dimensions** | > 1 D |
| **Broadcasting mode** | NDDMA hardware auto copying data according to stride=0 during GM→UB removal |
| **Distinction from UB BRC** | `Broadcast()` API call not required for UB to move in and finish broadcast |
| **NDDMA Maximum Dimension** | 5D. More than 5D requires an outer cycle + multiple NDDMA calls |

---

## II. Core API: DataCopy + MultiCopyParams

Nature of NDDMA broadcasting: Configure multi-dimensional moded copy, setting the srcStride of the broadcast axis to 0, where hardware is automatically copied.

### 2.1 MultiCopyParams Structure

```cpp
// NDDMA Maximum Support 5D
constexpr int64_t NDDMA_MAX_DIMS = 5;

AscendC::MultiCopyLoopInfo<NDDMA_MAX_DIMS> loopInfo;
// LoopInfo.loopSize[i] — Number of cycles of i-dimensional
// LoopInfo.loopSrcStride[i] — GM source leap (stride=0 → this dimension automatically)
// LoopInfo.loopDstStTride[i] — target jump for i-dimensional UB

AscendC::MultiCopyParams<T, NDDMA_MAX_DIMS> params = {loopInfo, constValue};
```

### 2.2 Motivation

```cpp
static constexpr AscendC::MultiCopyConfig config = {false, 0, 0, false};
AscendC::DataCopy<T, NDDMA_MAX_DIMS, config>(localTensor, globalTensor[gmOffset], params);
```

### 2.3 Broadcast effects of stride = 0

```
Example:: x=[1,3,8] It's broadcast. out=[4,3,8]
  inputStrides  = [0, 8, 1]    ← Axis 0 stride=0It needs to be broadcast.
  outputStrides = [24, 8, 1]

NDDMA Configure:
  loopSize      = [4, 3, 8]
  loopSrcStride = [0, 8, 1]    ← srcStride[0]=0Hardware on the axis. 0 Repeat the same data
  loopDstStride = [24, 8, 1]

Effect: hardware broadcasts `[1, 3, 8]` four times to fill `[4, 3, 8]`.
```

---

## III. Two models

### 3.1 WitoutLoop (schMode=1): the remaining axis after UB split ≤ 5

The remaining dimension is within the 5-dimensional limit of NDDMA and a `DataCopy` call is complete.

```cpp
// Configure MultiCopyParams: Map ubSplitAxis and subsequent axes to the 5D of NDDMA
MultiCopyParams<T, 5> params = BroadcastSetNddmaConfigWithoutLoop<T>(
    outputDims, outputStrides, inputStrides, shapeLen, ubSplitSize, ubSplitAxis);

// One move in to finish the broadcast.
DataCopy<T, 5, config>(localTensor, globalTensor[gmOffset], params);
```

**Optimization**: If a broadcast input `inputStrides[ubSplitAxis] == outputStrides[ubSplitAxis]` (i.e. the input does not need to be broadcast at the UB split axis), degradation is normal `DataCopyPad` and NDDMA costs are avoided.

### 3.2 With Loop (schMode=2): the remaining axis after UB split > 5

The remaining dimension exceeds the 5-dimensional limit of NDDMA. The innermost 5-dimensional limit is given to NDDMA, and the outer axle circulates through Kernel.

```cpp
// Configure NDDMA handles the inner 5D
MultiCopyParams<T, 5> params = BroadcastSetNddmaConfigWithLoop<T>(
    outputDims, outputStrides, inputStrides, shapeLen, ubSplitAxis);

// The outer circle runs through the remaining axes
int64_t nddmaProduct = BroadcastFuseAxes(outputDims, ubSplitAxis + 1, shapeLen - 5) * ubSplitSize;
int64_t nddmaIndices[3] = {0};

for (int64_t i = 0; i < nddmaProduct; i++) {
    if (i != 0) {
        BroadcastUpdateNddmaAxesIndices(nddmaIndices, outputDims, ubSplitAxis, ...);
    }
    int64_t nddmaGmOffset = BroadcastGetNddmaOffset(nddmaIndices, inputStrides, ...);
    int64_t nddmaUbOffset = BroadcastGetNddmaOffset(nddmaIndices, outputStrides, ...);

    DataCopy<T, 5, config>(localTensor[nddmaUbOffset],
                           globalTensor[gmOffset + nddmaGmOffset], params);
}
```

### 3.3 FuseAxis Optimization (WithLoop + CopyBrcSize ≤ 4)

When CopyBrc node ≤4 and ≥3, try to merge the adjacent axes in the same broadcast mode and reduce the number of NDDMA calls:

```cpp
// Scanning outward from the inner axis, with the same broadcasting mode (both stide=0 or both stide>0) combined
while (count > ubSplitAxis) {
    curFlag = inputStrides[count] == 0 ? 0 : 1;
    if (curFlag != oriFlag) {
        // Different modes → New dimensions
        outputDims2[newCount] = outputDims[count];
    } else {
        // Same mode → merge
        outputDims2[newCount] *= outputDims[count];
    }
}
```

---

## IV. Tiling Parameter Calculation

Same as the UB Broadcast branch (shared `DoBrodcastTiling`), except for SchMode:

```cpp
// Decision
int64_t axisInsideUB = shapeLen - ubSplitAxis;
if (axisInsideUB <= 5) {
    schMode = 1;   // WithoutLoop
} else {
    schMode = 2;   // WithLoop
}
```

---

## Kernel Implementation Process

```cpp
__aicore__ inline void Process()
{
    int64_t ubLoopNum = (GetBlockIdx() == GetBlockNum() - 1)
                        ? blockTail : blockFormer;

    int64_t axesIndices[8] = {0};
    BroadcastGetAxesIndices(axesIndices, blockFormer * GetBlockIdx(),
        outputDims, ubSplitAxis, dimProductBeforeUbInner);

    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx++) {
        if (ubLoopIdx != 0) {
            BroadcastUpdateAxesIndices(axesIndices, outputDims, ubSplitAxis, ubOuter);
        }

        int64_t ubSplitSize = (axesIndices[ubSplitAxis] == ubOuter - 1)
                              ? ubTail : ubFormer;

        // 1. Broadcast input: NDDMA migration (stride = 0-axis hardware automatic reproduction)
        //    Data reached UB complete after broadcast file
        BroadcastNddmaWithoutLoop/WithLoop(globalTensor, localTensor, ...);

        // 2. General input: DataCopyPad linear migration
        DataCopyPad(inputLocal, inputGm[gmOffset], {1, inputLength * sizeof(T), 0, 0});

        // 3. Calculation
        Add(outputLocal, input0Local, input1Local, tileLength);

        // 4. Removal
        DataCopyPad(outputGm[outOffset], outputLocal, {1, tileLength * sizeof(T), 0, 0});
    }
}
```

---

## VI. Comparison with UB Broadcast

| Dimensions | UB Broadcast (DAV_2201) | NDDMA Broadcast (DAV_3510) |
|------|---------------------|----------------------|
| Time to broadcast. | After moving in, UB inline calls Broadcast API | When moved in, hardware is automatically completed |
| UB Occupation | Need src + dst space | Only dst space (removation is the result) |
| API | `Broadcast<T, dim, axis>()` | `DataCopy<T, 5, config>()` |
| Dimension limits | Static interface 1D/2D; Dynamic interface rank 1-9 | NDDMA maximum 5D, exceeding required outer circulation |
| Performance | Additional Vector Command Costs | Hardware complete, no additional command. |
| tmpBuffer | Needs (Broadcast API internal) | I don't need it. |

---

## VII. Constraints

| Constraints | Annotations |
|------|------|
| **Chip** | DAV_3510 (Asend,950) only, DAV_2201 not supported |
| **NDDMA Maximum Dimension** | Five. Over the five-dimensional outer circle |
| **Tride=0 meaning** | InputStrides' axis = 0, hardware repeats unpropulsed address |
| **Degraded without broadcasting** | InputStrides = outputStrides degraded to DataCopyPad |
