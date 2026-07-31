# UB Buffer zone management guide

TBuf/TQue Selection, Double Buffer pipeline Parallel, Batch Removal Mode.

---

## Contents

1. [Tbuf vs TQue Selection](#tbuf-vs-tque-Selection)
2. [TQue Detailed](#tque - Detailed)
3. [TBuf Detailed](#tbuf-Definitioned)
4. [Double Buffer pipelineParallel](#double-buffer-pipelineParallel)
5. [Bulk removal + line-by-line calculation mode](#volume removal - line-by-line calculation mode)

---

## TBuf vs TQue Selection

| scene | Type of recommendation | Annotations |
|-----|---------|------|
| MTE2/MTE3 Move buffer zone | `TQue<VECIN/VECOUT>` | Need to parallel Victor and need EnQue/ DeQue |
| Pure Victor calculates the buffer zone | `TBuf<VECCALC>` | Not involving MTE handling, fetching with `Get<T>()` |
| Double Buffer | `TQue` + `InitBuffer(que, 2, size)` | Set num = 2 in InitBuffer |

---

## TQue Detail

### Template parameters

```cpp
template <TPosition pos, int32_t depth, auto mask = 0> class TQue;
```

| Parameters | Annotations |
|------|------|
| `pos` | Queue logical position: `VECIN`, `VECOUT`, `A1`, `A2`, `B1`, `B2`, `CO1`, `CO2` |
| `depth` | Queue Depth, indicating the number of consecutive EnQue/ DeQue |
| `mask` | Data format conversion (ND↔NZ) or compilation optimization parameters |

### Key description of depth parameters

| depth value | Apply scene | Annotations |
|---------|---------|------|
| `depth=1` | **Default recommended**, non-Tensor in situ | compiler has a special optimization and better performance. |
| `depth=0` | **Tensor In situ Operation** | Required Settings |
| `depth=2` | 2 consecutive EnQue scenes | Independent of num parameters for InitBuffer |

**Note: `depth` has nothing to do with Double Buffer. Double Buffer is controlled by `num` parameters for `InitBuffer`.

```cpp
// ✅ Non-continuous entry (ordinary scenario): decth=1
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;
pipe->InitBuffer(que, 1, size);
auto tensor = que.AllocTensor<T>();
que.EnQue(tensor);
tensor = que.DeQue<T>();
que.FreeTensor(tensor);
```

### Double Buffer Configuration

**Double Buffer is set in the `num` parameter for `InitBuffer`, not related to the template parameter `depth`.**

| InitBuffer Arguments | Role | Annotations |
|----------------|------|------|
| `InitBuffer(que, num, size)` | `num` Control Double Buffer | `num=1` = single Buffer, `num=2` = open Double Buffer |
| Template Parameters `depth` | Queue Depth | Indicates the number of consecutive EnQue |

```cpp
// ✅ Open Double Buffer: set num=2 in InitBuffer
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;  // Templates depth=1 That's fine.
pipe->InitBuffer(que, 2, size);  // num=2 Open Double Buffer

// ✅ Close Double Buffer
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;
pipe->InitBuffer(que, 1, size);  // num=1 Single Buffer
```

### TQue Buffer Quantity Limit

| Product series | Number of events IDs | Maximum TQue number |
|---------|-------------|---------------|
| Atlas Training Series | 4 | 4 |
| Atlas Logic Series AI Core | 8 | 8 |
| Atlas Logic Series, Victor Core | 8 | 8 |
| Atlas A2/A3 series | 8 | 8 |

Note:
- Double Buffer (num=1): Up to 8 TQue
- Open Double Buffer (num=2): Each TQue occupies 2 buffer with a maximum of 4 TQue

```cpp
// Only 4 TQue when you open Double Buffer
pipe->InitBuffer(que0, 2, size);  // ✅
pipe->InitBuffer(que1, 2, size);  // ✅
pipe->InitBuffer(que2, 2, size);  // ✅
pipe->InitBuffer(que3, 2, size);  // ✅
pipe->InitBuffer(que4, 2, size);  // ❌ Beyond the limit
```

### TQue Use correctly

```cpp
// TQue: Need for queue management (MTE handling-related)
// Template depth=1 is sufficient, Double Buffer sets in the InitBuffer num parameter
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
pipe->InitBuffer(inQueueX, 2, bufferSize);  // num=2 Open Double Buffer

AscendC::LocalTensor<half> x = inQueueX.AllocTensor<half>();
AscendC::DataCopyPad(x, xGm, {1, size * sizeof(half), 0, 0}, {false, 0, 0, 0});
inQueueX.EnQue(x);
// ...
AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
inQueueX.FreeTensor(xLocal);
```

---

## TBuf Detail

### Features

| Features | Annotations |
|------|------|
| Memory uses | Could not execute EnQue/DeQue |
| Memory Allocation | Each time InitBuffer assigns only one memory |
| Tensor Release | There's no need to release it manually. |

```cpp
// TBuf: Pure calculation of buffer zone
AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
pipe->InitBuffer(workBuf, bufferSize);

// ✅ Use Get<T>() to get Tensor without release
AscendC::LocalTensor<float> work = workBuf.Get<float>();
// ...calculating logic...
// No need, FreeTensor.
```

---

## Duble Buffer pipeline

### Core knowledge

**Double Buffer is not "calculated with two pieces of memory" but "to move in/out with two blocks of memory so that MTE2/MTE3 is calculated in parallel with Victor".**

Essential:**memory handling in parallel with calculation, covering latency**.

### Hardware principles

- **MTE2**: movers, GM → UB
- **Vector**: Processors, calculations
- **MTE3**: movers, UB → GM

### Timeline Contrast

**No Double Buffer (serial)**:
```
Row 0: [MTE2][Vector][MTE3]
Row 1:                      [MTE2][Vector][MTE3]
```

**There is Double Buffer (parallel)**:
```
Row 0: [MTE2-B0][Vector-B0][MTE3-B0]
Row 1:          [MTE2-B1][Vector-B1][MTE3-B1]
                  ↑ MTE2andVectorParallel!
```

### Realization of principles

| Buffer Type | InitBuffer num | Annotations |
|------------|----------------|------|
| `TQue<VECIN>` (MTE2 Removal) | **2** | Num=2 open Double Buffer, parallel to Victor |
| `TQue<VECOUT>` (MTE3 Removal) | **2** | Num=2 open Double Buffer, parallel to Victor |
| `TBuf<VECCALC>` (pure calculation) | - | TBuf does not involve MTE handling |

### Use correctly

```cpp
// Init: num=2 Start Double Buffer
pipe->InitBuffer(inQueueX,  2, tileSize * sizeof(T));
pipe->InitBuffer(outQueueY, 2, tileSize * sizeof(T));
pipe->InitBuffer(workBuf, workSize * sizeof(T));

// 2. Process: Single-cycle structure, TQue automatic rotation
for (int i = 0; i < totalTiles; i++) {
    CopyIn(i);   // MTE2 Step forwarding
    Compute(i);  // Vector Calculate
    CopyOut(i);  // MTE3 Move out of here.
}

// 3. CopyIn
void CopyIn(int i) {
    LocalTensor<T> x = inQueueX.AllocTensor<T>();
    DataCopyPad(x, xGm[i * tileSize], {1, (uint32_t)(tileSize * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    inQueueX.EnQue(x);
}

// 4. Compute
void Compute(int i) {
    LocalTensor<T> x = inQueueX.DeQue<T>();
    LocalTensor<T> y = outQueueY.AllocTensor<T>();
    Add(y, x, constTensor, tileSize);
    outQueueY.EnQue(y);
    inQueueX.FreeTensor(x);
}

// 5. CopyOut
void CopyOut(int i) {
    LocalTensor<T> y = outQueueY.DeQue<T>();
    DataCopyPad(yGm[i * tileSize], y, {1, (uint32_t)(tileSize * sizeof(T)), 0, 0});
    outQueueY.FreeTensor(y);
}
```

### Why does it have to go in parallel?

| Operation | Features |
|------|------|
| `DataCopy` | Step DMA, back immediately. |
| `EnQue` | Unblocked. Mark ready. |
| `DeQue` | Blocking, waiting. |

### Common error zone

| Error | Get it right. |
|------|---------|
| Manually split into two sets of Ping/Pong codes | Single Cycle + `InitBuffer(que, 2, size)` AutoManaging |
| depth Template Parameter Control | Double Buffer is controlled by `num` parameters for `InitBuffer` |
| The bigger the bigger the better. | Template depth is usually set to 1 with the highest value for money |
| All buffer needs num = 2. | Double Buffer is needed only for MTE porters |

---

## Batch load + line-by-line mode

### Apply scene

When processing multiline data, batch handling reduces the number of MTE2/MTE3 calls, making full use of bandwidth.

### Mode Structure

```
CopyInBatch(NOkay.) → Line-by-line calculation(NNumbers) → CopyOutBatch(NOkay.)
```

### Code Template

```cpp
__aicore__ inline void ProcessBatch()
{
    uint32_t totalRowsToProcess = endRow - startRow;
    if (totalRowsToProcess == 0) return;

    for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
        uint32_t startLocalRow = tile * tileRows;

        // Border check: prevent uint32_t spill
        if (startLocalRow >= totalRowsToProcess) break;

        uint32_t remaining = totalRowsToProcess - startLocalRow;
        uint32_t rowsThisTile = (remaining < tileRows) ? remaining : tileRows;

        CopyInBatch(startLocalRow, rowsThisTile);
        ComputeBatch(rowsThisTile);
        CopyOutBatch(startLocalRow, rowsThisTile);
    }
}
```

### Host Side Tiling Calculator

```cpp
// A2/A3 UB = 192KB
constexpr uint64_t UB_SIZE = 192 * 1024;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;  // DataCopyPad blockCount Limits

// bytesPerTileRow: double buffer (in*2 + out*2)
uint32_t bytesPerTileRow = paddedColsT * typeSizeBytes * 4;

// tileRows
uint32_t tileRows = (UB_SIZE - overheadBytes) / bytesPerTileRow;
tileRows = std::max(1u, std::min(tileRows, MAX_BLOCK_COUNT));
```

### note

1. **tieRows limit**: `blockCount` maximum 4095 for DataCopyPad
2. **Final processing**: early exit at `startLocalRow >= totalRowsToProcess`
3. **stride calculation**: UB side stride is 32 bytes, GM side is bytes

---

## Multisage shared L1 / L0 Buffer constants

### Apply scene

Mix Kernel (`__mix__(N, M)`) or more stage operator, the same pair of L1 / L0 Buffer is often rotated by multiple Compute status-sharing (e.g. two consecutive Mmad calculations of GMM operator share the same pair of L1 input buffer).

### It has to be consistent constants.

The following constants are used in each stage function,**which must be consistent with the actual allocation bytes for InitBuffer**:

- Number of single slot elements (`slotElems`)
- Single slot bytes (`slotBytes`)
- per-slot profile/ slot offset base

### Typical pedal.

```cpp
// InitBuffer when allocated:
buf.InitBuffer(matAL1_, 64 * 1024 * PRELOAD_NUM);   // Every slot 64KB

// CenterStage1 (correct):
const uint32_t slotElems = 64 * 1024 / sizeof(DATA_T);   // and InitBuffer Unanimously
auto a = matAL1_.Get<DATA_T>()[loopSlot * slotElems];

// CommandStage2 (Error!
const uint32_t slotElems = 128 * 1024 / sizeof(DATA_T);  // ❌ Error 128KB
auto b = matAL1_.Get<DATA_T>()[loopSlot * slotElems];    // task=0 Offset=0 Mooching,task=1+ Read cross-border dirty data
```

### Symptom

- Task task = 0 output normal (diversion = 0, even if the constant is not crossed, read in the legal distribution area)
- Task task =1+ output NAN / inf (diversion to dirty data at end of buffer)
- cyclical error of the "Alternative PASS / Odd Case FAIL" or "First task PASS / Follow-up Task All Exploding"

### Project constraints

All per-slot constants refer to a single header or single constexpr definition, all stages refer to the same definition:

```cpp
// constants.h
constexpr uint32_t L1_BUF_A_SLOT_BYTES = 64 * 1024;
constexpr uint32_t L1_BUF_B_SLOT_BYTES = 64 * 1024;
constexpr uint32_t L1_BUF_A_SLOT_ELEMS = L1_BUF_A_SLOT_BYTES / sizeof(DATA_T);
```

Avoids declaring `const uint32_t slotElems = ...;` separately in each stage function.