# Group Reduce (cross-border return)

## 3. Group Reduce (cross-border return)

**Applicable scenario**: R is too big for one nucleus to go through; and A is too small to make full use of multinucleus

### 3.1 Two-stage implementation model

```
Phase 1(nuclear independence):
  ┌──────┐  ┌──────┐  ┌──────┐
  │Core 0│  │Core 1│  │Core 2│
  │R[0:K]│  │R[K:2K]│ │R[2K:N]│
  └──┬───┘  └──┬───┘  └──┬───┘
     │          │          │
     ↓          ↓          ↓
  workspace[0] workspace[1] workspace[2]
     ↓          ↓          ↓
  ┌────────────────────────────┐
  │        SyncAll()           │
  └────────────────────────────┘
     ↓
Phase 2(Consolidated nuclear):
  read workspace[0..coreNum]
  merge all partials → final output
```

### 3.2 Phaase 1 Implementation Templates

```cpp
void GroupReducePhase1() {
    int myRStart = rGroupIdx * rPerGroup;
    int myREnd = min(myRStart + rPerGroup, totalR);

    // Initialize partial
    Duplicate(partialBuf, initValue, outSize);  // 0 for sum, -inf for max

    for (int r = myRStart; r < myREnd; r += cutRSize) {
        int curR = min(cutRSize, myREnd - r);
        CopyIn(xLocal, r, curR);
        // Partial return
        ReduceOp(partialBuf, partialBuf, xLocal, curR);
    }

    // Write partial to workspace
    int wsOffset = blockIdx * SLOT_STRIDE;  // 64B Keep your eyes open. bank conflict
    DataCopyPad(workspaceGm[wsOffset], partialBuf, {1, outSize * sizeof(float), 0, 0});
}
```

### 3.3 Phase 2 Implementation Templates

```cpp
void GroupReducePhase2() {
    SyncAll();  // Wait till all the cores are complete. Phase1

    // Merge All Partials
    Duplicate(finalBuf, initValue, outSize);

    for (int g = 0; g < groupR; g++) {
        int wsOffset = (aBlockIdx * groupR + g) * SLOT_STRIDE;
        CopyIn(partialLocal, workspaceGm[wsOffset], outSize);
        ReduceOp(finalBuf, finalBuf, partialLocal, outSize);
    }

    // Write Final Results
    CopyOut(yGm[myAStart], finalBuf, outSize);
}
```

### 3.4 Welford Group Reduce (specialized for statistical reporting)

For reduce_var, Phase 1 output is (partial_mean, partial_M2, partial_count),
Phase 2 merges with the Welford formula:

```cpp
void WelfordGroupReducePhase2() {
    SyncAll();

    // Read first group as initial value
    float totalMean = workspace_mean[0];
    float totalM2 = workspace_M2[0];
    int totalCount = workspace_count[0];

    // Group by Group
    for (int g = 1; g < groupR; g++) {
        float gMean = workspace_mean[g];
        float gM2 = workspace_M2[g];
        int gCount = workspace_count[g];

        float delta = gMean - totalMean;
        int newCount = totalCount + gCount;
        totalMean += delta * gCount / newCount;
        totalM2 += gM2 + delta * delta * totalCount * gCount / newCount;
        totalCount = newCount;
    }

    float var = totalM2 / (totalCount - correction);
}
```

