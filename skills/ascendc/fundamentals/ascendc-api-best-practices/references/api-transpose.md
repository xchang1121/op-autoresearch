# Transpose API best practice

This document focuses on the API combinations, hard bounds and anti-modules common in small-channel transpose.

> **Current coverage**: This document currently covers only**small channels**; large corridors and common transpose scenarios are not included for the time being and may be supplemented as necessary.

***

## 1. Core computational links

### 1.1 Small channel with `TransDataTo5HD + Gather`

The rationale:

- Step1. TransDataTo5HD must always enter 16 lines each (and fill in valid data for less than 16 lines, e.g., line 0, otherwise unknown anomalies will occur), command operations will convert 16 rows \[16, N] to 16 rows \[, 16], recapat completed \[16, 16], repeat (N + 15) / 16 times after receiving \[N, 16]
- Step2. When the number of lines [e.g. C] before transfer is less than 16, \[n, 16], \[n, 16], \[n, C], ffset needs to be constructed in the kernel in advance from the front TransDataTo5HD by Gather operation;

### 1.2 `TransDataTo5HD` conversion input

```cpp
constexpr uint32_t EPB16 = 16;
uint32_t repeats = tileNA / 16;

LocalTensor<half> srcList[16];
LocalTensor<half> dstList[16];
for (uint32_t i = 0; i < 16; ++i) {
    // Source address offsets according to input line size, total tileNA needs to be converted
    srcList[i] = halfLocal[(i < channelCount) ? (i * tileNA) : 0];
    // Destination addresses are offset according to 16 elements and repeat is converted at a time [16,16] Requires consecutive writing (tileNA + 15) / 16 [16,16]
    dstList[i] = vnLocal[EPB16 * i];
}

// The target repeat transfer output [16,16], multiple repeats need to be rewritten, so write 0byte for the first time, 16*blocksize (32B) = 128B for the second time, set 16 blocksize (32B) for offset
uint16_t dstRS = (repeats == 1) ? 0 : 16;
// Source repeatstride Each repeat consumes 16 elements according to the direction of the line entered, multiple repeats read continuously according to the direction of the line, setting 1 blockside (32B) offset
uint16_t srcRS = (repeats == 1) ? 0 : 1;
TransDataTo5HDParams params(false, false, static_cast<uint8_t>(repeats), dstRS, srcRS);
TransDataTo5HD<half>(dstList, srcList, params);
```

**Specific example (C=3, tileNA=32): understand input output data layout**

> Remember to enter the matrix elements as`a[r][c]`,among them rIndex as Channel0..C-1),cIndex as Column0..tileNA-1).

**Enter `halfLocal`**(`[C, tileNA]` = `[3, 32]` in line order flat storage):

```
halfLocal flat Address:   0 ...  31 |  32 ...  63 |  64 ...  95
Meaning(Matrix Line):     Row0[0..31] | Row1[0..31] | Row2[0..31]
Element Value:           a[0][0..31] | a[1][0..31] | a[2][0..31]
```

```
Row 0:  [a00, a01, a02, ..., a0_15 | a0_16, a0_17, ..., a0_31]
Row 1:  [a10, a11, a12, ..., a1_15 | a1_16, a1_17, ..., a1_31]
Row 2:  [a20, a21, a22, ..., a2_15 | a2_16, a2_17, ..., a2_31]
```

**srcList / dstList build**

```
srcList[0] = &halfLocal[ 0]  → Row 0
srcList[1] = &halfLocal[32]  → Row 1
srcList[2] = &halfLocal[64]  → Row 2
srcList[3..15] = &halfLocal[0]   ← Filling rows (shortfall)16Use Row0 Fill, otherwise abnormal)

dstList[i] = &vnLocal[16 * i]    (i = 0..15)
```

**Implementation process**(repeats = 32/16 = 2):

| Repeat | Read source column range | srcList[i] Offset | Convert Blocks | Write vnLocal range |
|--------|------------|----------------|---------|-----------------|
| 0 | Source column [0.15] | +0 Element | `[16,16]` | Element [0.255](16 ×16) |
| 1 | Source column [16.31] | +16 Element | `[16,16]` | Element [256.511](16 ×16) |

**Repeat 0 Output**(dstRS=16, srcRS=1):

```
vnLocal[  0..15 ]:  [a[0][0],  a[1][0],  a[2][0],  *, *, ..., *]
vnLocal[ 16..31 ]:  [a[0][1],  a[1][1],  a[2][1],  *, *, ..., *]
vnLocal[ 32..47 ]:  [a[0][2],  a[1][2],  a[2][2],  *, *, ..., *]
...
vnLocal[240..255]:  [a[0][15], a[1][15], a[2][15], *, *, ..., *]
```

**Repeat 1 output**:

```
vnLocal[256..271]:  [a[0][16], a[1][16], a[2][16], *, *, ..., *]
vnLocal[272..287]:  [a[0][17], a[1][17], a[2][17], *, *, ..., *]
...
vnLocal[496..511]:  [a[0][31], a[1][31], a[2][31], *, *, ..., *]
```

**Final `vnLocal` data layout**(Equivalent to `[tileNA, 16]` matrix, stored in line order):

```
          col0       col1       col2       col3..15   ← 16Columns, front onlyC=3Column validity
Row 0 :  a[0][0]    a[1][0]    a[2][0]    ******
Row 1 :  a[0][1]    a[1][1]    a[2][1]    ******
 ...       ...        ...        ...       ******
Row15 :  a[0][15]   a[1][15]   a[2][15]   ******
Row16 :  a[0][16]   a[1][16]   a[2][16]   ******    ← repeat=1 Here we go.
Row17 :  a[0][17]   a[1][17]   a[2][17]   ******
 ...       ...        ...        ...       ******
Row31 :  a[0][31]   a[1][31]   a[2][31]   ******
```

> **Order**: `vnLocal[r] [c] = original input a[c] [r]`(c < C), i.e., rounding. `*` is shown as a filling value, followed by Gather discard.

**Gather extracts a valid channel**and obtains the final conversion result `[tileNA, C]` = `[32, 3]`:

```
Row 0 :  a[0][0]   a[1][0]   a[2][0]
Row 1 :  a[0][1]   a[1][1]   a[2][1]
 ...
Row31 :  a[0][31]  a[1][31]  a[2][31]
```

The output of `TransDataTo5HD` is valid for each 16-half block only the previous `channelCount` position; the remaining position is padding. The subsequent `Gather` must be used to retrieve the valid value.

### 1.3 `Gather` Ripping Active Channels

```cpp
auto halfOut = halfLocal;
Gather(halfOut, vnLocal, offsetBuff, 0, validCount);
Cast(outLocal, halfOut, RoundMode::CAST_ROUND, validCount);
```

If the previous in-place round is completed in the FP32 phase, the `Gather` / `Cast` here tends to be processed by matching count, and eventually `half -> uint8` can also use `CAST_NONE` directly; the range of valid output is still determined by the current file `curN * channelCount`.

Here's the `offsetBuff`, which is the expected byte table of the device end (only once), which is managed using Tbuff, which corresponds to the logic of generation:

### 1.4 OffsetBuff Generation: Scalar → Victor Command Optimization

****Question: Universal realization is done on an element-by-element basis in SetValue table, tileNA × C by Scalar. When tileNA = 2048, C = 3 Scalar is written 6144 times, Scalar is up to 90 per cent in small-scale settings and becomes a performance bottleneck.

**Key observation**: offset table has a cyclical structure — a group of 16 p values with a constant inter-group margin of 16 × 16 × sizeof (half) = 512 bytes:

```
Group 0: offset[p*3+0] = (p*16+0)*2,  offset[p*3+1] = (p*16+1)*2,  offset[p*3+2] = (p*16+2)*2   (p=0..15)
Group 1: and Group 0 It's exactly the same. It's just every element. +512
Group 2: and Group 0 It's exactly the same. It's just every element. +1024
...
```

**Optimization method**: Scalar Generation Base Mode + Adds vector Command Batch Extension

```
__aicore__ inline void InitOffsetTable()
{
    auto offsetI32 = offsetBuf.Get<int32_t>();
    uint32_t baseCount = 16 * C;
    // Step 1: Scalar SetValue Generation Base Mode (only 16×C elements)
    for (uint32_t p = 0; p < 16; ++p) {
        for (uint32_t c = 0; c < C; ++c) {
            offsetI32.SetValue(p * C + c, (p * 16 + c) * sizeof(half));
        }
    }
    // Step 2: Adds vector command extension follow-up group (vector operation per group)
    uint32_t totalGroups = tileNA / 16;
    for (uint32_t g = 1; g < totalGroups; ++g) {
        AscendC::Adds(offsetI32[g * baseCount], offsetI32[0],
                      static_cast<int32_t>(g * 16 * 16 * sizeof(half)), baseCount);
    }
}
```

| Indicators            | Pre-optimization (pure SetValue) | Optimized (Scalar+Adds) |
| ------------- | --------------- | ---------------- |
| Scalar Call Number   | 6144            | 48               |
| Adds vector call times   | 0               | 127              |
| Scalar ratio  | 90.5%           | 55.1%            |
| VEC ratio     | 11.1%           | 58.7%            |
| Task Duration | 55.6 us         | 15.3 us          |

**Conditions applicable**:

- Offset table cyclical structure with equal number columns
- BaseCount = 16 ×C to meet Adds alignment requirements (32 bytes, i.e. baseCount ≥ 8 against int32)
- C ≤ 16 (typical scenario for small channels)

**General Mode**: Any search table with a cyclical structure (offset table, index table, etc.) can be optimized by using the "Scalar Generating Base Mode + Adds vector Extension" to reduce the Scalar operation from O(tileNA × C) to O(16 × C).

***

## 2. API-level hard bounds

### 2.1 `Gather` not directly processed `uint8`

The recommended route is:

```text
FP32 -> half -> TransDataTo5HD -> Gather(half) -> uint8
```

Do not try to make a gather extraction directly on `uint8`.

### 2.2 `repeats == 1` stride must set 0

```cpp
uint16_t dstRS = (repeats == 1) ? 0 : 16;
uint16_t srcRS = (repeats == 1) ? 0 : 1;
```

It's a little tile scene of hard constraints that can't be saved.

### 2.3 `VECOUT` depth shall > = 2

Even if the logic of `Compute` appears to be "Current Write", it is not necessary to narrow the `VECOUT` queue to one. When multiple files are mixed down the CopyOut with the subsequent Compute, single slots are susceptible to death.

### 2.4 GM mandatory `DataCopyPad` for reading and writing, compatible with 32B pairs of Zife

The output is `curN * channelCount` bytes. As long as it is not strictly 32 bytes aligned:

```cpp
DataCopyPad(yGm[gmOffset], outLocal, copyParams);
```

Do not introduce additional end block branch complexity in order to write less than one `Pad` path.

***

## 3. Reverse and Reverse Modes

| Inverse Mode                                      | Problem                  | Suggested replacement                                    |
| ---------------------------------------- | ------------------- | --------------------------------------- |
| `GetValue / SetValue` Element-by-Element Movement              | scalar UB reading and writing, very bad swallowing       | `DataCopy / DataCopyPad + vector route` |
| Pixels by Pixels `DataCopyPad(blockLen=channelCount)` | DMA setup costs much more than payload | Move through the channel, then do `vnchwconv + Gather`         |
| Default Apply Universal Transpose API                     | The internal costs under the small tunnel scenario are likely to be much greater than actually calculated. | Take a specific path to the small passageway.                               |
| Direct `float -> half -> uint8`              | Easy to quantify off-by-1     | In-place round, half                |
| Cross file manage one-time event                     | It's easy to write water as a one-time synchronized death lock.      | Manage with `TQue` 's `EnQue/DeQue`             |

