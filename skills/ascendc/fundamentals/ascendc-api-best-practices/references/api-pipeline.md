# Guide to the pipeline Synchronization Mechanism

MTE 's core mechanism for synchronization with Victor.

---

## Contents

1. [core issues](#core issues)
2. [Settling](#Settling)
3. [By two scenarios](#By two scenarios)
4. [full pipeline template](#full pipeline template)
5. [Debug techniques](# debug techniques)

---

## Core issues

**DataCopy/DataCopyPad is a step apart DMA operation, doing Vector calculations directly on data after removal and possibly reading incomplete data!**

### Hardware architecture

```
GM → MTE2 (A walk.) → UB → Vector (Sync) → MTE3 (A walk.) → GM
```

### Problem scene

```cpp
// ❌ error: use data directly after DataCopyPad
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);  // Probably read the data on the uncompleted removal!
```

**phenomena**: random, error in output data

---

## Solutions

### Option I: EnQue/DeQue queue synchronization (recommended)

**Rationale**: EnQue/DeQue mechanism provides automatic hardware synchronization points.

```cpp
// ✅ Correct: Sync with EnQue/ DeQue
// Step 1: CopyIn - MTE2 Removal
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
inQueueX.EnQue(xLocal);                    // Tags"Ready"

// Step 2: Compute - Victor Calculator
AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // Block and wait. MTE2 Completed
AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
AscendC::Adds<float>(yLocal, xIn, 1.0f, count);
outQueueY.EnQue(yLocal);
inQueueX.FreeTensor(xIn);

// Step 3: Copyout - MTE3 Removal
AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // Block and wait. Vector Completed
AscendC::DataCopyPad(yGm[gmOffset], yOut, copyOutParams);
outQueueY.FreeTensor(yOut);
```

**Key points**:
- `EnQue(xLocal)` Tag Buffer Data Ready
- `DeQue<float>()` block pending data readiness
- After DeQue returns, the data must have been moved.

### Option 2: PipeBarrier Manual Synchronization

```cpp
// ✅ Available: Sync with PipeBarrier
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
AscendC::PipeBarrier<PIPE_ALL>();          // Wait. MTE2 Completed

AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);

AscendC::DataCopyPad(yGm[gmOffset], yLocal, copyOutParams);
AscendC::PipeBarrier<PIPE_ALL>();          // Wait. MTE3 Completed
```

**Disadvantages**: performance costs (full pipeline pause) not recommended for high performance scenarios

---

## Comparison of the two scenarios

| Features | EnQue/DeQue | PipeBarrier |
|-----|-------------|-------------|
| Synchronize Particle Degrees | Buffer Level | All pipeline |
| Performance | High (support parallel) | Low (serial waiting) |
| Code Complexity | Queue management required | Simple and straightforward. |
| Level of recommendation | ⭐⭐⭐⭐⭐ | ⭐ ⭐. |

### The dual role of EnQue/DeQue

1. **Queue management**: Double Buffer scenario managed many buffer rotations
2. **Hardware Synchronization**: provides synchronization points between MTE ↔ Vector

```cpp
// EnQue/DeQue, not just the queue, but more importantly the sync mechanism.
inQueueX.EnQue(xLocal);    // 1. Tag Data Ready  2. Notification hardware can wait.
xLocal = inQueueX.DeQue(); // 1. Blocking and waiting.  2. Access Available buffer
```

---

## Full pipeline template

```cpp
__aicore__ inline void ProcessTile(uint32_t tileIdx)
{
    // Synchronization point.
    // MTE2: GM → UB (Alternate)
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopyPad(xLocal, xGm[tileIdx * tileSize], copyParams, padParams);
    inQueueX.EnQue(xLocal);              // Synchronization point: mark ready

    // Synchronization point.
    // Victor: UB Calculating (Sync, awaiting MTE2)
    AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // Sync Point: Wait MTE2
    AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    AscendC::Adds<float>(yLocal, xIn, 1.0f, tileSize);
    outQueueY.EnQue(yLocal);             // Synchronization point: mark ready
    inQueueX.FreeTensor(xIn);

    // Synchronization point.
    // MTE3: UB → GM (sliding, waiting for Victor)
    AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // Sync Point: Wait Vector
    AscendC::DataCopyPad(yGm[tileIdx * tileSize], yOut, copyParams);
    outQueueY.FreeTensor(yOut);
}
```

### Scheduling of pipeline

```
Time →

Tile 0:  [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                      ↑ DeQueWait.    ↑ DeQueWait.
Tile 1:          [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                  ↑ Parallel!    ↑ DeQueWait.    ↑ DeQueWait.

Key:DeQue Block pending the completion of the last phase of the walk.
```

---

## Debug techniques

### Check for missing EnQue/ DeQue

```cpp
// ❌ error: direct after AllocTensor
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
Compute(x);  // Wrong. Probably read the data on the uncompleted removal.

// ✅ Correct: DeQue to calculate later
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();  // Waiting for removal to complete
Compute(xIn);
```

### Temporary plus PipeBarrier debugging

```cpp
DataCopy(x, gm, size);
PipeBarrier<PIPE_ALL>();  // Temporary plus, if the result correctly indicates the problem of synchronization
Compute(x);
```

**If Pipe Barrier can solve the problem, it means synchronization.**→ Rehabilitation Program: change to EnQue/ DeQue Mechanism

### Common error zone

| Error | Get it right. |
|-----|---------|
| Data is available after AllocTensor | AllocTensor only assigns memory without waiting for removal. |
| DataCopy. It's synchronized. | DataCopy is a step away. DMA, return immediately. |
| No, EnQue/DeQue can work. | Must sync with EnQue/ DeQue or PipeBarrier |
| Pipe Barrier, good performance. | Pipe Barrier, full pipeline pause, performance poor. |
