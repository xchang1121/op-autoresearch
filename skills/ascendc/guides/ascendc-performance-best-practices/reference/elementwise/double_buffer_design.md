# Double Buffering / double buffering pipeline Optimizing Design

## 1. Optimization of objectives

in memory-boundElements by Elementsoperator,single buffer (InitBuffer num=1) Caused byMTE2(data moving in)Vector(calculated)MTE3The three stages of the serial execution, the computing unit is free of much effort. This is optimized.double buffering(InitBuffer num=2) or multi-stage buffer strategy for data removal and calculationpipelineIn parallel,**Hide Memory Accesslatency**.

| Indicators | naive | optimized | Proceeds |
|------|-------|-----------|------|
| MTE2/VECTOR Overlay | 0% (string) | ~80-90% (pipeline parallel) | throughput Upgrade 20-50% |
| Utilization of computing units | Low (mass idle) | High (continuable data available) | Maximize hardware utilization |
| UB Memory Occupancy | `tile_size × num_tensors` | `2 × tile_size × num_tensors` | Double it, need budget precision. |
| Code Complexity | Low | Medium (sync events to manage) | Pre-cycle + Event Sync |

> ApplicationoperatorGroup:`elementwise`(`sin`, `cos`, `abs`, `exp`, `foreach_add`Waiting for everything.memory-boundElement by Elementsoperator) and others`omni-ops`General scene.

## 2. Overview of the structure

### 2.1 Storage tiers and data flows

> `double_buffering.md` Pipeline Timeline Comparison:

**Single buffer:**
```
Single buffer:
  MTE2: [LOAD0]          [LOAD1]          [LOAD2]
  VEC:         [COMP0]          [COMP1]          [COMP2]
  MTE3:               [STORE0]        [STORE1]        [STORE2]
  Time: =========================>
```

**double buffering:**
```
Double buffer:
  MTE2: [LOAD0][LOAD1][LOAD2][LOAD3]...
  VEC:         [COMP0][COMP1][COMP2]...
  MTE3:               [STORE0][STORE1]...
  Time: ============>  (significantly shorter)
```

### 2.3 Event Synchronization Model

| Event type | Meaning | Purpose |
|---------|------|------|
| `HardEvent::MTE2_V` | MTE2 migration completed → Victor readable | Data readiness notifications |
| `HardEvent::V_MTE3` | Victor complete → MTE3 writeback | Calculate completion notification |
| `PipeBarrier<PIPE_V>` | Victor PIPE Synchronization | Data Dependence Between Commands |

### 2.4 Buffer strategy extension

| Policy | Buffer Number | Apply scene | Water flow phase |
|------|------------|---------|---------|
| Double Buffer | 2 | Universal memory-baund | Move in/calculate/ remove tier 3 overlaps |
| Triple Buffer | 3 | AIC/AIV Mixed Core (Cube+Vector) | Remove /Cube/Vector Level 3 simultaneously |
| 2×2 Matrix Buffer | 4 | Matmul cut M+K two dimensions | M/K double-dimensional simultaneously flowing water |
| TQueBind | 1 (VECIN/VECOUT shared) | Read-reform-write in-place | 50% savings |
| Custom PingPong | 2+ (handwritten flag) | FlashAttention Multiple Storage | L1/L0 Multi-level PingPong |

## 3. Key Parameter Configuration

```cpp
// Host side TilingData
struct DoubleBufferTiling {
    uint32_t tileSize;        // Single tile Volume of data (number of elements)
    uint32_t bufferNum;       // Number of buffers:2(double buffering)or 3(three buffers)
    uint32_t ubFactor;        // UB Part of the factor.bufferNum Half when we double it.
};
```

### 3.1 Tile Size Selection Principle

| Parameters | Typical value | Annotations |
|------|--------|------|
| `tileSize` | 2048 / 4096 / 12288 | 32B alignment; halved double buffering |
| `bufferNum` | 2 (general)/ 3 (CV integration) | double buffering standard, three buffers for Cube+Vector |
| `ubFactor` | Original value / `bufferNum` | ub_factor has to be halved when the num of InitBuffer doubles |

**UB Memory Budget Validation:**
```
Required UB = tile_size × sizeof(dtype) × buffer_num × num_tensors
Available UB = 192KB (Ascend 910B3)

Example: FP16, 2 tensors (in+out), double buffer
  = 12288 × 2 × 2 × 2 = 98304 bytes = 96KB [PASS]
```

### 3.2 Self-adaptation double buffering decision-making

When UB is large enough to accommodate all data once at once,**double buffering**is not required (to reduce the costs of synchronization); otherwise double buffering is enabled to hide latency.

```cpp
// Host side adaptation decision-making
bool useDb = true;
if (maxRow > rowPerHeadCore) {  // UB Large enough.
    useDb = false;                // Do Not Usedouble buffering
}
SetTilingKey(context, xDtype, useDb);
```

## 4. Core Calculator Cycle

### 4.1 naive version (before optimization)

```cpp
// Single buffer, line execution
TQue<QuePosition::VECIN, 1> inQueue;
TQue<QuePosition::VECOUT, 1> outQueue;
pipe.InitBuffer(inQueue, 1, tileSize * sizeof(T));
pipe.InitBuffer(outQueue, 1, tileSize * sizeof(T));

for (uint32_t i = 0; i < tileNum; i++) {
    // Stage 1: MTE2 Move!
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopy(inLocal, inGm[i * tileSize], tileSize);
    inQueue.EnQue(inLocal);

    // Stage 2: Victor calculate (MTE2 employment!)
    LocalTensor<T> computeLocal = inQueue.DeQue<T>();
    Compute(computeLocal, tileSize);

    // Stage 3: MTE3 Move out (Vector idle!)
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
    DataCopy(outGm[i * tileSize], computeLocal, tileSize);
    outQueue.EnQue(outLocal);
    outQueue.FreeTensor(outLocal);
    inQueue.FreeTensor(computeLocal);
}
```

### 4.2 Optimized version (after optimization): Standard double buffering

```cpp
// double buffering + Pre-restructuring Cycle
// Note: TQue template parameter depth is queued depth (continuous EnQue number), not related to double buffer.
//       Depth is recommended as 1 (compiler with special optimization) in a non-local operating scenario.
//       Double Buffer is only enabled by the num parameter of InitBuffer (num=2).
TQue<QuePosition::VECIN,  1> inQueue;   // depth = 1(Recommended value)
TQue<QuePosition::VECOUT, 1> outQueue;  // depth = 1(Recommended value)
pipe.InitBuffer(inQueue,  2, tileSize * sizeof(T));  // num = 2 Open double buffer
pipe.InitBuffer(outQueue, 2, tileSize * sizeof(T));  // num = 2 Open double buffer

// Prefetch: Load the first file
CopyIn(0);

// Main cycle: Calculating i, pre-empting i+1, writing back i-1
// Compatibility of MTE2 / Victor / MTE3 Level 3 hardware pipeline
for (uint32_t i = 0; i < tileNum - 1; i++) {
    CopyIn(i + 1);      // MTE2: Preset Next tile(with current round) Vector Parallel)
    Compute(i);         // Vector: Calculating Current tile(with previous round) MTE3 Parallel)
    if (i > 0) {
        CopyOut(i - 1); // MTE3: Write back the previous result (with next round) MTE2 Parallel)
    }
}

// End: processing of the last two files (ends unwritten in cycle)
Compute(tileNum - 1);
for (uint32_t i = (tileNum > 1) ? tileNum - 2 : 0; i < tileNum; i++) {
    CopyOut(i);
}
```

### 4.3 Optimized version: Event Synchronization double buffering (precision control)

```cpp
// Use HardEvent to create a more refined pipeline sync
// Note: Following is the use of TBuf to assign two physical buffers, manually synchronized by anevent
TBuf<TPosition::VECCALC> pingBuf, pongBuf;
pipe.InitBuffer(pingBuf, tileSize * sizeof(T));
pipe.InitBuffer(pongBuf, tileSize * sizeof(T));

event_t pingId = EVENT_ID6;
event_t pongId = EVENT_ID7;

for (uint32_t idx = 0; idx < tileNum; idx++) {
    auto pipeId = (idx % 2 == 0) ? pingId : pongId;
    LocalTensor<T> local = (idx % 2 == 0) ? pingBuf.Get<T>() : pongBuf.Get<T>();

    // Waiting for the previous round of MTE3 to write back (avoid overwrite unwritten data)
    WaitFlag<HardEvent::MTE3_MTE2>(pipeId);
    CopyIn(local, idx * tileSize, tileSize);  // MTE2 Move in
    SetFlag<HardEvent::MTE2_V>(pipeId);       // Announcements Vector Data Ready

    WaitFlag<HardEvent::MTE2_V>(pipeId);      // Waiting for data to be ready
    Compute(local, tileSize);                  // Vector Calculate
    SetFlag<HardEvent::V_MTE3>(pipeId);       // Announcements MTE3 Calculate finished
}
```

### 4.4 Optimized version: three buffers (Cube+Vector Integration)

```cpp
// / Cube / Victor Level 3 currents
template<BufferType bufferType, SyncType syncType>
class BuffersPolicy3buff {
    Buffer<bufferType, syncType> a_, b_, c_;
    uint32_t flag1_ = 0, flag1_vec1_ = 0, flag1_bmm2_ = 0;
public:
    Buffer<bufferType, syncType>& Get() {
        if (flag1_ == 0) { flag1_ = 1; return a_; }
        else if (flag1_ == 1) { flag1_ = 2; return b_; }
        else { flag1_ = 0; return c_; }
    }
    Buffer<bufferType, syncType>& GetVec() {
        if (flag1_vec1_ == 0) { flag1_vec1_ = 1; return a_; }
        else if (flag1_vec1_ == 1) { flag1_vec1_ = 2; return b_; }
        else { flag1_vec1_ = 0; return c_; }
    }
    Buffer<bufferType, syncType>& GetCube() {
        if (flag1_bmm2_ == 0) { flag1_bmm2_ = 1; return a_; }
        else if (flag1_bmm2_ == 1) { flag1_bmm2_ = 2; return b_; }
        else { flag1_bmm2_ = 0; return c_; }
    }
};
```

## 5. Key change point from naive to doule_buffer

| Modify Item | (before optimization) | Double_buffer (optimized) |
|--------|---------------|----------------------|
| InitBuffer Memory Blocks | `num = 1` | `num = 2` (or 3/4) |
| Loop Structure | `for(i) { CopyIn(i); Compute(i); CopyOut(i); }` | Prefetch +`for(i) { CopyIn(i+1); Compute(i); CopyOut(i-1); }` |
| UB Usage | `tile_size × num_tensors` | `2 × tile_size × num_tensors` |
| Synchronisation Method | Implicit (EnQue/DeQue) or none | HardEvent Visible Sync + PipeBarrier |
| Tiling Link | Fixed file_size | tile_size Halve (maintenance of total UB budget) |
| Apply scene | All scenes (but low performance) | Memoory-bund, big data scene |

## 6. note/ Constraint

1. **Not enough to change the num of InitBuffer only. The cycle must be recreated to pre-take mode**.
   - Error: Change `num=1→2` to `InitBuffer(..., num, ...)` only, recycles the same → without performance enhancement.
   - Correct: Pre-element (prefetch) cycles must be achieved.

2. **Tiling: ub_factor has to be halved when the num of InitBuffer doubles.**double buffering doubles the queue memory, and without adjusting ub_factor, total UB excesses can cause translation failure or runtime to cross the border.

3. **Synchronization command costs**: HardEvent `SetFlag/WaitFlag` has a slight cost, which is used only by the necessary path. Excessive synchronization offsets the flow of water.

4. **Tail file processing**: tile_size halved and the number of tiles doubled, ensuring that the tail file processing logic is correctly updated.

5. **Boundary applicable to different buffer strategies**:
   - Purely Victor operator → Double Buffer (2 Buffer)
   - Cube+Vector Integration → Triple Buffer (3 Buffer)
   - Matmul Double-Drive → 2 × 2 Mattrix Buffer (4 buffer)
   - Read-reform-write in-place → TQueBind (shared physics Butffer)

6. **Self-adaptation strategy**: UB is large enough (to accommodate all data at a time), instead of double buffering, to reduce synchronised costs.

7. **Limited number of QUE Buffer on the same TPostion**(reference official document).
- Atlas A2 training series products (e.g. 910B1): QUE Buffer does not exceed**8**on the same TPostion.
- Atlas Training/Dictionary Series: QUE Buffer does not exceed**4**on the same TPostion.
- `QuePosition::VECIN` and `QuePosition::VECOUT` averaged the bottom map to `TPosition::VECCALC` (UB), so `inQueue(2) + outQueue(2)` together occupied**$4**. While the ceiling of 910B1 is not reached, attention is to be paid to the expansion. If more buffer is needed, it should be merged into a block of buffer by deflecting**instead of continuing to add a separate queue.

8. **Event ID management**: Custom PingPong requires manual management of events ID arrays to avoid relapse conflicts.

## 7. Implement common issue and Solutions

| Problem | Gene. | Solutions |
|------|------|---------|
| Unable to enhance after double buffering | Cycle is not reformed as pre-feed mode | `CopyIn(i+1); Compute(i); CopyOut(i-1)` flow structure has to be achieved |
| UB Spill Compiler Failed | tile_size not halved | ub_factor must simultaneously halve the num: 1→2 of InitBuffer |
| Data competition/ silent error | Sync missing or event ID reuse | Use independent event ID for each SetFlag/ WaitFlag pair |
| End file processing anomaly | tile doubles the number and ends the tile logic is not updated | Update host side tilling calculation: tileNum =ceil (total / newTileSize) |
| Three buffer debugging difficulties | 3 sets of stand-alone flag complex management | Harmonize the Get/GetVec/GetCube interface with the BuffersOffice3buff |

## 8. Selective decision-making and self-check list

### 8.1 Selective decision-making

```
if (operatorYes. memory-bound && MTE2 active >> VECTOR active):
    if (Pure Vector operator):
        → Enable Double Buffer(InitBuffer num=2)
        → tile_size Halve it, keep it. UB Budget
    elif (Cube+Vector Integration):
        → Enable Triple Buffer(InitBuffer num=3)
        → Removal/Cube/Vector Level 3 currents
    elif (UB It's big enough to accommodate all data at once.):
        → Do Not Usedouble buffering(reduced synchronized costs)
else:
    → A single buffer is sufficient (compute-bound scenedouble buffering(No income)
```

### 8.2 Self-check List

- [ ] Reconstruct cycle to pre-take mode: `CopyIn(0)` + `for(i) { CopyIn(i+1); Compute(i); CopyOut(i-1); }`
- [ ] `InitBuffer num: 1→2` when `ub_factor` synchronised halved
- [ ] Use `HardEvent` or `PipeBarrier` to ensure data dependence is correct
- [ ] The number of tail tiles has doubled as the number of tail tiles has been halved, the tail tile logic has been updated
- [ ] UB Memory Budget Validation: `tile_size × sizeof(dtype) × buffer_num × num_tensors < UB_capacity`
- [ ] Event ID No Reuse Conflict (SetFlag/ WaitFlag Independent)
