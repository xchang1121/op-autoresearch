# UB/TBuf Resident Re-entry and Bank Conflict Circumvention Optimization Design

## 1. Optimization of objectives

In Victor operator, the small size parameters (weight, gamma, scale) or the cross-generational status (cumulator, push result) that repeat each tile/loop from the GM will result in significant excess MTE2 costs. This optimally eliminates double loads by placing such data in UB/Tbuf, reading and writing an initialization outside the cycle,**reading and writing a circular internal film. Also, avoiding UB Bank conflict**by optimizing the distribution of addresses and calculating logical adjustments.

| Indicators | naive | optimized | Proceeds |
|------|-------|-----------|------|
| Number of Parameters moved | 1 per file/loop | Process life cycle only 1 time | Reduction in the number of moves N times (N=tie/ Loop) |
| Temporary Buffer Total | `sum(stage_buffers)` | `max(concurrent_buffers)` | 30-60 per cent reduction in UB occupancy |
| Victor single Repeat cycle | Multishoot (bank conflicts, different from chips) | 1-2 shoot (no conflict) | Multiply the speed of calculation |
| Code Complexity | Low | Medium (needs life cycle management) | Based on buffer + partitions |

## 2. Overview of the structure

### 2.1 Permanent Re-entry vs-minutes

| Policy Type | Data life cycle | Re-use | Typical scene. |
|---------|------------|---------|---------|
| **Removed** | Cross the whole Proces or multilayer cycle | Loop out, read/write only in cycle | Small argument |
| **Restart** | Multistage serials in a singleProcess | Completion of the previous phase, and coverage of the latter phase | Rms Norm → RoPE, Soft Max → Reprocessing |

### 2.2 UB Bank Structure and conflict rationale

Take Ascend 910B, for example: UB (192KB) = 48 bank × 4KB = 16 bank group × 3 bank/group. The number of banks for different chips may differ and the corresponding hardware manual needs to be consulted.

#### 2.2.1 The hardware rationale of the Bank conflict

The Victor computing unit sends multiple DataBlock requests in parallel over a Repeat cycle. Ideally, these requests are distributed and executed on different bank/bank group, and one Repeat cycle completes all visits. When multiple requests fall to**the same bank or bank group**because of the address map, hardware arbitration logic collide them, leading to monoRepeat being degraded from 1-2 to multiple shots (specific numbers vary depending on the number of chip and conflict DataBlock).

> **Note: The bank structure and conflict conditions of the different chips vary, and the corresponding hardware files must be consulted. The following is illustrated by the NPU version 220x (Atlas A2/A3 series) and Atlas 350.

#### 2.2.2 UB Bank structure of different chips

| Chip. | UB Size | Bank Number | Bank Group | Bank Number per Group | Per Bank Size | Size per Line |
|------|---------|---------|---------------|-------------|-------------|---------|
| DAV_2201 | 192 KB | 48 | 16 | 3 | 4 KB (128 Line) | 32 B |
| DAV_3510 | 256 KB | 16 | 8 | 2 | 16 KB (512 rows) | 32 B |

**Bank Group composition**: group each bank number matches `bank_id % num_groups`. Example: 220x:
- group 0: bank 0, 16, 32
- group 1: bank 1, 17, 33
- ...
- group 15: bank 15, 31, 47

**Differences in conflict conditions**(Key differences):

| Type of conflict | NPU 220x | Atlas 350 |
|---------|----------|-----------|
| Conflict in reading and writing | Same**bank** | Same**bank** |
| Writing Conflict | Same**bank group** | Same**bank group** |
| Read Conflict | Same**bank group** | Same**bank**(two reading operations), or**more than two**read operations same bank group |

> The bank group in Atlas 350 has two reading and writing groups, so**two reading operations will not conflict with the same bank group when they visit different bank**, but there will be clashes with arbitrary multiple simultaneous visits within the same bank group at 220x.

#### 2.2.3 DataBlock and Bank Map

The data processed by the Vector command were cut into fixed sizes**DataBlock**, with 32B lengths per line. One Vector command processed up to**8 DataBlock**(block0~block7). Each DataBlock maps to a bank based on the initial address and then belongs to a bank group.

**Core law of mapping**(guided by official examples, the formula is based on the chip manual):
- The adjacent 32B row on the address usually maps the adjacent bank (rounded by the number of bankes).
- Thus,**the address difference between DataBlock (decided by `blk_stride`) directly determines whether they fall into the same bank group**.

Take 220x, for example:
- `blk_stride = 16`: Neighbored DataBlock address difference = 16 × 32B = 512B. Bank number difference = 16 (assuming a 32B line of inquiry), 16 % 16 = 0, i.e.**All DataBlock fell into the same bank group**,8 resulting in a repecat.
- `blk_stride = 8`: Bank number difference = 8, block0 and block2, bank difference = 16,16 %16 = 0, fell into the same group, 4 finished one Repaat.

#### 2.2.4 Conflict judgement methods

**Step 1: Identify and send access to data sources**

A Victor command (e. g. `Add(dst, src0, src1)`) will read `src0`, `src1` and write `dst` at the same time in Repeat. DataBlock of these three operations is the source of cross-referenced conflicts.

**Step 2: Analysis of address spacing patterns**

- **Between multiple operations**: The corresponding DataBlock is cyclically mapped to the same bank/bank group if the initial buffer addresses of `dst`, `src0`, `src1` are distributed continuously within UB and the interval is exactly several times the integer of bank size.
- **Inner single operations**: If `blk_stride` makes the difference between bank and bank between DataBlock equals the number of groups (220x: 16,350: 8) or multiples thereof, multiple DataBlocks return to the same bank group.

**Step 3: Confirmation through official tools or experiments**

- **msProfAnalysis of the share of resource conflicts**Officially availablemsProfTools can capture data on the share of resource conflicts and directly locate thembankConflict.operatorDevelop a tool document.
- **Profiling Validation**: If Victor calculates much longer than the theoretical value and MTE2 is not a bottleneck, there is a high probability of a bank conflict.
- **Experimental validation**: add 256B padding or adjust `blk_stride` and reprofiling to observe whether the number of cycles has decreased.

#### 2.2.5 Common conflict scenarios and circumvention principles

The following is a typical example of an official document.

**scene 1: read-write conflict -- src is in the same bank as dst**

```cpp
// Assume x start address at bank0 and y start address at bank0 (integer multiple of bank size)
Add(dst, src, src2, ...);  // src Read it. bank0,dst Write bank0, reading and writing conflicts
```

**Collision**: Ensure that src and dst start addresses are not the same bank. For 220x, usually add padding to middle buffer to stagger an address with at least one bank.

**Scene 2: Writing Conflict - dst's multiple DataBlocks fall into the same bank group**

```cpp
// 220x blk_stride=16:8 DataBlock all fall into the same bank group,8 finished
Adds(dst, src, scalar, MASK, 1, {1, 16, 1, 16});

// 220x blk_stride=8:block0 and block2 fell into the same bank group, 4 finished
Adds(dst, src, scalar, MASK, 1, {1, 8, 1, 8});
```

**Refrain**: change to `blk_stride = 1` (continuous reading) to control address increments across Repeat through `dst_gap/src_gap`.

**scene 3: Read Conflict - double src falls into the same bank group (220x), or the same bank (350)**

```cpp
// 220x:x and y start address difference of several times bank size, DataBlock 0 read the same bank group
Add(zLocal, xLocal, yLocal, ...);
```

**Hiding**: Add 256B padding after `xBuf`, breaking the address cycle, dispersing DataBlock from `x` and `y` to different bank group. For 350, it is also necessary to ensure that two srcs are not in the same bank.

**Screen 4: Distribution of multiple buffer, address cyclical overlap**

```cpp
// 220x original realization: x/y/z distribution on a continuous basis, starting address difference 0x4000 (16KB)
// x: bank0, y: bank0 (16KB = 4×bank size), z: bank0
// Reads the same group, x/ y and z read and write the same bank at the same time as x and y
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));
pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```

**Evasion (220x official recommendation)**:
```cpp
// x Multiple application 256B to avoid reading the same bank group at the same time as x and y within Repeat
// y apply for more space to ensure that z won't fall into the same bank as x/y
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float) + 256);
pipe.InitBuffer(inQueueY, 1, 64 * 1024 - (4096 * sizeof(float) + 256));
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```

### 2.3 Data flows - permanent reuse

Permanent reuse: Small Size Parameters (weight, gamma, scale) move to UB/TBuf at MTE2 at the Proces entrance, then cycle read-only throughout the life cycle, without access to GM at all times.

### 2.4 Data Flow - Reuse Time (Zone Reuse)

Time reuse: Single block TBuf is divided into multiple zones by stage. The previous phase (e.g. RmsNorum) calculates the expiration of data using zone0/1/2, and the later phase (e.g. RoPE) directly reuses the zone0/1/2 in the same physical space without any additional allocation of buffer. The total of UB temporary buffer is reduced from `sum(stage_buffers)` to `max(concurrent_buffers)`.

## 3. Key Parameter Configuration

```cpp
// Permanent-Repeated Parameters
struct ResidentBufferConfig {
    uint32_t paramSize;       // Number of permanent parameters (e. g.) hidden_size)
    uint32_t computePrecision; // Calculateaccuracy:FP16=2, FP32=4 bytes
};

// Reuse time arguments
struct ZoneReuseConfig {
    uint32_t zone0Size;       // Phase 1 Temporary buffer Size
    uint32_t zone1Size;       // Phase 2 Temporary buffer Size
    uint32_t zone2Size;       // Phase 3 Temporary buffer Size
};

// Bank Conflict Circumvention Parameters
struct BankConflictConfig {
    uint32_t dataSize;        // Data Size (bytes)
    uint32_t paddingSize;     // Padding Size (usually) 256B)
};
```

### 3.1 Principles for the selection of parameters

| Parameters | Typical value | Annotations |
|------|--------|------|
| `paramSize` | 64 / 128 / 256 / 512 | Head_dim or hirden_size |
| `paddingSize` | 256 | Stagger neighbouring buffer bank group |
| `zoneOffset` | `rows * headSize` | Split by volume of data |

## 4. Core Calculator Cycle

### 4.1 naive version (before optimization)

**Parameters/tile duplicates:**
```cpp
for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
    for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
        // Every time a loop moves from GM weight
        LocalTensor<half> weightLocal = inQueueW.AllocTensor<half>();
        DataCopyPad(weightLocal, weightGm, copyParams);
        Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);
        Compute(xLocalFp32, weightFp32, ...);
        inQueueW.FreeTensor(weightLocal);
    }
}
```

**Reverted from time to time - independent distribution:**
```cpp
// Independent Buffer, Total = sum
pipe.InitBuffer(rmsBuf0, 1, zoneSize);
pipe.InitBuffer(rmsBuf1, 1, zoneSize);
pipe.InitBuffer(rmsBuf2, 1, zoneSize);
pipe.InitBuffer(ropeCosBuf, 1, zoneSize);
pipe.InitBuffer(ropeSinBuf, 1, zoneSize);
// Total = 5 × zeroSize
```

**Bank Conflict Reverse - Continuous Distribution (220x):**
```cpp
// Question: x/y/z is allocated continuously, with an address difference of DataSize.
// If dataSize is an integer multiple of bank size (4KB), the corresponding DataBlock cyclical map of x/y/z
// The same bank group. Add commands read x, y, and write z at the same time, creating serious reading/writing/read conflicts.
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));   // addr = 0x0
pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));   // addr = 0x4000
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));  // addr = 0x8000

// Question: 220x blk_stride=16,8 DataBlock all fall into the same bank group (bank difference = 16),
// 16 % 16 = 0), 8 finished a Repaat.
Adds(dst, src, scalar, MASK, 1, {1, 16, 1, 16});  // Entire conflict!

// Question: 220x blk_stride=8, block0 and block2 fell into the same group (bank difference = 16), 4 beat.
Adds(dst, src, scalar, MASK, 1, {1, 8, 1, 8});    // Part of the conflict!
```

### 4.2 Optimized version (after optimization)

**Remunerated — Profess entrance moved in, cycle read-only:**
```cpp
__aicore__ inline void Process() {
    // Stage 0: one move to + Cast, permanent in UB
    LocalTensor<float> weightFp32 = this->inQueueW.AllocTensor<float>();
    DataCopyPad(weightLocal, weightGm, copyParams, padParams);
    Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);

    // Stage 1 ~N: read-only permanent within cycle
    for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
            Compute(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32);
        }
    }
    this->inQueueW.FreeTensor(weightFp32);
}
```

**Retrieval — Zone Division:**
```cpp
// Single Tbuf split into multiple zone
int64_t xLocalFp32Offset = 0;
int64_t xSquareLocalOffset = rows * headSize;
int64_t xSumLocalOffset = rows * headSize * 2;

LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];

// Phase 1: Use Zone0/1/2
RmsNorm(xLocalFp32, xSquareLocal, xSumLocal, ...);

// Phase 2: Reuse Zone0/1/2 for RoPE
LocalTensor<float> ropeCosLocal = wsLocal[xLocalFp32Offset];
LocalTensor<float> ropeSinLocal = wsLocal[xSquareLocalOffset];
RoPE(ropeCosLocal, ropeSinLocal, ...);
```

**Bank Conflict Circumvention — Padding Stagger (220x official recommendation):**
```cpp
// Rationale: x Multiple application 256B to stagger bank group for x and y
// y and complete 64KB (16 bank) border to ensure that z is not with x/y bank.
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float) + 256);
pipe.InitBuffer(inQueueY, 1, 64 * 1024 - (4096 * sizeof(float) + 256));
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```

**Bank Conflict Circumvention - Continuous reading and writing:**
```cpp
// Motion: blk_stride=1 gives 8 DataBlocks in the same number of operations a sequence map to the adjacent bank.
// Avoid falling into the same bank group. dst_gap/src_gap controls address increments across Repeat.
// 220x mask=128 (8 DataBlock) read without conflict consecutively; jump scripts spread dst.
UnaryRepeatParams params;
params.dstBlkStride = 8;
params.srcBlkStride = 1;
Adds(dstLocal, srcLocal, 0, 128, 2, params);  // Keep reading, skipping.
```

## 5. Key change points from live to ub_result

| Modify Item | (before optimization) | ub_reident (after optimization) |
|--------|---------------|---------------------|
| Parameter Removal | Repeat MTE2 per file/ loop | One move in at the Procs entrance, permanent at UB |
| Temporary Buffer Total | `sum(stage_buffers)` (separate distribution) | `max(concurrent_buffers)` (division reuse) |
| UB Address Allocation | Distribution, no padding | 256B paddy stagger bank group |
| Vector stride | Jump to Script (blk_stride=16) | Continuously read and write (blk_stride=1) |
| Data life cycle | Independent of each stage | Reuse the same physical space in the serial phase |

## 6. note/ Constraint

1. **Permanent-based buffer main data compression file space**: permanent-based buffer will take over UB for a long time and will need to ensure that the remaining space will still accommodate the main data file, double buffering and temporary computing space.

2. **TBuf/VECCALC is not protected by Queue Synchronization**: Need to use Pipe Barrier manually or clear stages of the boundary to ensure consistency.

3. **Time-to-time reuse requires a strict chain of borders**: strict serial boundaries must be present at each stage, and a miscalculation of the life cycle would lead to subsequent phases covering the data still in use.

4. **Bank Conflict Circumvention requires a distinction between chip types**:
   - NPU 220x (A2/A3): 192KB = 48 bank × 4KB = 16 group × 3 bank, for reading/writing/reading conflict conditions, see 2.2.2.
   - Atlas 350: 256KB = 16 bank × 16KB = 8 group × 2 bank, two reading operations of the same group different bank.
   - Specific specifications and recommended configurations must be consulted in the corresponding version of Ascend C operator Development best practice.

5. **Padding adds UB occupation**: 256B padding although small, multi-buffer accumulated to be included in the total UB budget.

6. **Recalculated for subsequent phases of overlapping water flow: if subsequent optimization changes the serial phase to overlapping water flow, the original Zone use programme may fail.

## 7. Implement common issue and Solutions

| Problem | Gene. | Solutions |
|------|------|---------|
| UB Spills after permanent buffer | Based on buffer + master file + double buffering overbudget | Decrease size of file_size or permanent buffer; precise calculation of UB budget |
| Time-overriding data | Stage boundary error | Ensure that the previous phase is fully covered after completion; use Pipe Barrier to clarify the boundary |
| Victor's still low. | Bank Unsolved Conflict | Use msProf to collect resource conflicts as a percentage confirmed; search by 2.2.5 symmetrical scene (read-write/write-read-read conflicts); confirm that DataBlock of padding after each buffer is not the same bank/ bank group; check if `blk_stride` resulted in 8 DataBlock returns to the same group |
| Weak accuracy | FP16 excretion | Permanently using FP32 accuracy |

## 8. Selective decision-making and self-check list

### 8.1 Selective decision-making

```
if (operatorInclude small size parameters weight/gamma/scale And cross. loop No change.):
    → Enables permanent copying:Process Inbound, read-only in cycle.
elif (operatorInclude multi-stage serial calculation, temporary buffer Life cycle does not overlap):
    → Enable time reuse: single block TBuf Partition, inter-stage coverage
elif (Profiling Show Vector The calculation time is abnormally high.):
    → Enable bank Conflict circumvents:padding staggered or staggered
else:
    → Standard allocation is fine.
```

### 8.2 Self-check List

- [ ] Resident Buffer Size + Main file + double buffering ≤ UB capacity
- [ ] Time-lapse phases with strict serial borders (Pipe Barrier)
- [ ] Bank Conflict Circumvention: Adjacent Buffer interval ≥ 256B or not the same bank group
- [ ] blk_stride of Victor API to avoid causing multiple DataBlock to fall into the same bank group
- [ ] Use FP32 accuracy for pressurizer/push status
- [ ] Validation pass: achieves the same results compared to the naive