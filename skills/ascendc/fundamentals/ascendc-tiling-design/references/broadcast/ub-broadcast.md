# Broadcast - UB Broadcast static interface (DAV_2201)

> **Applied scene**: Multi-dimensional after axle, there are inputs that need to be broadcast. The `Broadcast()` static interface is used to expand the → calculation by moving to raw data → UB. Only rank=1/2, axis=0/1 is supported.
>
> **DAV_3510**It is suggested to use dynamic interfaces (rank 1-9, no alignment limit), as detailed in [dynamic-ub-broadcast.md] (dynamic-ub-broadcast.md).

---

## I. SPECIFIC STATE

| Features | Annotations |
|------|------|
| **Post-axis dimensions** | > 1 D |
| **Broadcasting mode** | Move to unbroadcast data, UB in call `Broadcast()` API extension |
| **Data continuity** | Enter consecutively, but each input different (radio axle dim=1) |
| **Calculated** | vector equal size to output file |

---

## II. Decision-making on broadcasting

```
⚠️ Align judgment with original shape Dimension Value × sizeof(T)No, it's not. DataCopyPad Move UB Back-to-back!
   Example:srcShape[1]=37, 37×4=148B → No, it's not. 32B Multiplier → Unsatisfied constraints
   Even after `DataCopyPad`, the UB row width is 160B and still violates the constraint.

Broadcast Inputsstride=0 Axis:
  │
  ├─ axis=-1((M,1)→(M,N))?
  │   ├─ Satisfied Broadcast Static interface binding ()srcShape[0] × sizeof(T) Yes. 32B How many?
  │   │   └─ YES → Broadcast Static interface(s)§(iv)
  │   │
  │   └─ NO → M > 2?
  │       ├─ YES → DataCopyPad dummy Fill + Copy + GatherMask(§3.1)
  │       └─ NO(M≤2) → Line-by-line Duplicate Expand
  │
  ├─ axis=-2((1,N)→(M,N))?
  │   ├─ Satisfied Broadcast Static interface binding ()srcShape[1] × sizeof(T) Yes. 32B How many?
  │   │   └─ YES → Broadcast Static interface(s)§(iv)
  │   │
  │   └─ NO → Copy Line Copy + GatherMask(§3.2)
  │            This is not a good idea.DataCopyPad After moving in, UB already 32B Alignment
  │
  └─ Other → Broadcast Static interface(s)§(iv)
```

---

## III. Efficacy of handling command broadcasting

Broadcast API was replaced by DataCopyPad + Copy + GatherMask, and tmpBuffer was used to carry running water to complete the broadcast.

### 3.1 axis =-1 Broadcast: DataCopyPad dummy fill

**Applicable scene**: most kernel dim=1 broadcast to dim=N, e.g. `(13, 1)` → `(13, 37)`

**Rationale**: DataCopyPad fills the dummy data with**lead element value**of the source data block when `blockLen` is under 32B alignment. `blockLen = sizeof(T)` (one element) fills 32B (float: 1→8).

**Example**(dtype=float):

```
Step 1: DataCopyPad Move in (%1)blockLen=4B, blockCount=13)
  GM [0] → UB: [0,0,0,0,0,0,0,0]   ← 32B,8 Same one. float
  GM [1] → UB: [1,1,1,1,1,1,1,1]
  ...
  GM [12]→ UB: [12,12,...,12]
  Got it. UB Let's go. (13, 8) Data

Step 2: Copy Expand the width of the target(s)srcStride=0 Repeat the same. 8 Elements)
  (13, 8) → (13, 40)              ← 40 = ceil(37/8)*8

Step 3: GatherMask Crop to a valid width
  (13, 40) → (13, 37)             ← Get rid of the tail. 3 A spare element.
```

### 3.2 axis=2 Broadcast: Copy Row Copy

**Applicable scene**: 2nd-dimensional dim = 1 broadcast to dim = M, e.g. `(1, 37)` → `(13, 37)`

**Rationale**: Copy the line to the number of the target lines using Copy, then crop the GatherMask.

**Example**(dtype=float):

```
Step 1: DataCopyPad Move in 1 LineblockLen=37*4=148B, blockCount=1)
  GM [0..36] → UB: [0,1,...,36, ?,?,?]   ← 160B(32B I'm not sure I can do that.40 individual float

Step 2: Copy Copy to Target Lines (%1)srcStride=0 Repeat the same line)
  (1, 40) → (13, 40)

Step 3: GatherMask Crop each line to a valid width
  (13, 40) → (13, 37)
```

### Involving API

| API | Purpose |
|-----|------|
| `DataCopyPad` | GM→UB, axis=1 fill with dummy; axis=2 move to single line (automatic 32B alignment) |
| `Copy` | UB Inner Removal, srcStride=0 Repeat reading to achieve row/column extensions |
| `GatherMask` | Press mask to select a valid element and crop to the actual width |

### Advantages

- No Broadcast API, tmpBuffer
- The handling instructions are broadcast, the running water can be transported in parallel with the calculating of the running water.

---

## IV. Broadcast static interface

DAV_2201/DAV_3510 is available, but DAV_3510 recommends that priority be given to the dynamic interface (see the initial link). dim and axis support only 1D/2D, axis=0 or 1.

```cpp
// dim: tensor dimension (1 or 2)
// axis: broadcast dimensions (0 or 1)
Broadcast<T, dim, axis>(dstLocal, srcLocal, dstShape, srcShape, tmpBuffer);
// or framework automatically apply for a temporary space version (tmpBuffer without manual management)
Broadcast<T, dim, axis>(dstLocal, srcLocal, dstShape, srcShape);

// Example: [M,1] → [M, K](broadcasting along axis=1)
uint32_t dstShape[] = {M, K};
uint32_t srcShape[] = {M, 1};
Broadcast<float, 2, 1>(dstLocal, srcLocal, dstShape, srcShape, tmpBuffer);
```

### Constraints

| Constraints | Annotations |
|------|------|
| **Dimensions** | 1D and 2D only |
| **axis** | Only 0 and 1 |
| **dim=2, axis=0** | SrcShape [1] must be 32B aligned |
| **dim=2, axis=1** | SrcShape [0] must be 32B aligned |
| **Address overlap** | Src and dst cannot overlap |
| **dtype(DAV_2201)** | int8_t, uint8_t, half, float |

### tmpBuffer Size

```cpp
// Host Side Retrieving
uint32_t maxTmpSize, minTmpSize;
GetBroadCastMaxMinTmpSize(platform, srcShape, dstShape, sizeof(T), false, maxTmpSize, minTmpSize);
// Kennel side reserved maxTmpSize bytes
```

---

## V. Tiling Parameter Calculation

### 5.1 UB Capacity Calculation

```cpp
// BufferNum = all survival nodes
// maxDtypeBits = maximum dtype bit width
maxElemNum = (ubSize - extraSize) * 8 / (bufferNum * maxDtypeBits);
maxElemNum = floor_align(maxElemNum, 256 * 8 / minDtypeBits);  // 256B repeat Alignment
```

### 5.2 UB split

```cpp
// Aggressive output from the inner axis to the outside dims, find indestructible axes
uint64_t curProduct = 1;
for (i = shapeLen - 1; i >= 0; i--) {
    curProduct *= outputDims[i];
    if (curProduct > maxElemNum) {
        ubSplitAxis = i;
        curProduct /= outputDims[i];
        break;
    }
}
ubFormer = maxElemNum / curProduct;
ubOuter = ceil(outputDims[ubSplitAxis] / ubFormer);
ubTail = outputDims[ubSplitAxis] - (ubOuter - 1) * ubFormer;
```

### 5.3 Polynuclear Cuts

```cpp
fusedProduct = ubOuter;
for (i = 0; i < ubSplitAxis; i++) {
    fusedProduct *= outputDims[i];
}
blockFormer = ceil(fusedProduct / coreNum);
blockNum = ceil(fusedProduct / blockFormer);
blockTail = fusedProduct - (blockNum - 1) * blockFormer;
```

---

## VI. KERNEL IMPLEMENTS

### 6.1 Data flows

```
For each of them. tile (by ubSplitAxis Severation):

  General input (no broadcast):
    GM → DataCopyPad → UB [tileSize]

  Broadcast Inputsstride=0 Axis):
    GM → DataCopyPad → UB [srcShape, Not broadcast]
      ↓
    Broadcast(dst, src, dstShape, srcShape) → UB [dstShape, Broadcasted]

  Calculate:
    Add/Mul/Sub(output, input0, input1, eleNum)

  Move out.:
    UB → DataCopyPad → GM
```

### 6.2 MDI indexing

Kernel needs to maintain a multi-dimensional index `axesIndices[8]` to calculate the GM offset for each file:

```cpp
// Initialization: expand blockFormer * blockIdx to a multi-dimensional index
void BroadcastGetAxesIndices(int64_t axesIndices[], int64_t flatIdx,
    const int64_t outputDims[], int64_t ubSplitAxis, int64_t dimProduct)
{
    for (int64_t i = 0; i < ubSplitAxis; i++) {
        dimProduct /= outputDims[i];
        axesIndices[i] = flatIdx / dimProduct;
        flatIdx %= dimProduct;
    }
    axesIndices[ubSplitAxis] = flatIdx;  // ubSplitAxis Index in %1
}

// ubLoop every time later
void BroadcastUpdateAxesIndices(int64_t axesIndices[], const int64_t outputDims[],
    int64_t ubSplitAxis, int64_t ubOuter)
{
    axesIndices[ubSplitAxis]++;
    if (axesIndices[ubSplitAxis] >= ubOuter) {
        axesIndices[ubSplitAxis] = 0;
        // Outward bits
        for (int64_t i = ubSplitAxis - 1; i >= 0; i--) {
            axesIndices[i]++;
            if (axesIndices[i] < outputDims[i]) break;
            axesIndices[i] = 0;
        }
    }
}
```

### 6.3 GM Offset Calculation

```cpp
// GM offset for normal input
int64_t gmOffset = 0;
for (int64_t i = 0; i < ubSplitAxis; i++) {
    gmOffset += axesIndices[i] * inputStrides[i];
}
gmOffset += axesIndices[ubSplitAxis] * ubFormer * inputStrides[ubSplitAxis];

// Radio input: stride=0 axis non-contribution deviation
// (Axes of broadcasting have been set to zero during the axis phase, and the formula is naturally skipping)
```

### 6.4 Core implementation cycle

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

        // 1. Moving into regular input
        int64_t gmOffset0 = BroadcastGetGmOffset(axesIndices, input0Strides, ...);
        DataCopyPad(input0Local, input0Gm[gmOffset0],
            {1, inputLength0 * sizeof(T), 0, 0});

        // 2. Moving into broadcast input (original shape)
        int64_t gmOffset1 = BroadcastGetGmOffset(axesIndices, input1Strides, ...);
        DataCopyPad(srcBrcLocal, input1Gm[gmOffset1],
            {1, srcBrcLength * sizeof(T), 0, 0});

        // 3. UB Internal Broadcasting
        uint32_t dstShape[] = {ubSplitSize, innerDim};
        uint32_t srcShape[] = {1, innerDim};  // Radio axis dim=1
        Broadcast<T, 2, 0>(dstBrcLocal, srcBrcLocal, dstShape, srcShape, tmpBuffer);

        // 4. Calculation
        Add(outputLocal, input0Local, dstBrcLocal, ubSplitSize * innerDim);

        // 5. Removal
        int64_t outOffset = BroadcastGetGmOffset(axesIndices, outputStrides, ...);
        DataCopyPad(outputGm[outOffset], outputLocal,
            {1, ubSplitSize * innerDim * sizeof(T), 0, 0});
    }
}
```

---

## VII. Buffer Planning

| Buffer | Size | Purpose |
|--------|------|------|
| General input × N₁ | tileSize × sizeof(T) | Inputs without broadcast |
| Broadcast input source × N₂ | srcTileSize × sizeof(T) | Raw data before broadcast |
| Broadcast Input Expand × N₂ | tileSize × sizeof(T) | Post-broadcast data |
| Output | tileSize × sizeof(T) | Calculate results |
| tmpBuffer | GetBroadCastMaxMinTmpSize | Broadcast API temporary space |

UB ≈ (N₁s + 2× N₂ + 1) × fileSize × maxDtypeBytes +tmpBufferSize

---

## VIII. common issue

| Problem | Reason | Solutions |
|------|------|---------|
| Broadcast made a mistake. | dstShape/srcShape | dstShape is output file Shape, srcShape is input original |
| dim=2, axis=0 | srcShape [1] Not 32B alignment | Use dynamic interface or pad to align |
| accuracy error | The ubSplitAxis division resulted in incomplete broadcast coverage | Make sure the radio axes aren't cut apart by UB. |
| The multi-dimensional index crossed the border. | AxesIndices Logic Error | Check UpdateAxesIndices Boundaries |
