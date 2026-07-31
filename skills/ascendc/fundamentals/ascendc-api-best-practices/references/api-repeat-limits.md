# Victor API recapitTime Parameter Limit

> **Core issue**: maximum value when repeattime is uint8_t 255, over which spill will result in computational error

---

## Contents

1. [core constraints](#core constraints)
2. [How to confirm the range of parameters](# how to confirm the range of parameters)
3. [issue scene and solution](#problem scene and solution)
4. [Affected API](#Affected-api)

---

## Core constraints

**Before using Victor API, you must confirm the limits of the `repeatTime` parameter!**

When `repeatTime` is of `uint8_t` type, the maximum value**255**.

---

## How to confirm the range of parameters

### 1. View parameter type in function prototype

```cpp
// Sub API function prototype (asc-devkit/docs/api/context/Sub.md)
template <typename T, bool isSetMask = true>
__aicore__ inline void Sub(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src0,
    const LocalTensor<T>& src1,
    uint64_t mask,
    const uint8_t repeatTime,  // ← uint8_t Type
    const BinaryRepeatParams& repeatParams
);
```

### 2. Set the range by data type.

| data type | Value range | Meaning |
|---------|---------|------|
| `uint8_t` | 0 ~ 255 | Up to 255 inverts. |
| `uint16_t` | 0 ~ 65535 | Up to 65535. |
| `int32_t` | - | No such limit |

---

## Problem scenes and solutions

### Problem scene

```cpp
// Question: R=256, RowCount=256 > 255
uint32_t rowCount = 256;
AscendC::Sub<float>(
    dst[col],
    src0[col],
    src1[col],
    curMask,
    rowCount,  // uint32_t Import uint8_t Parameter,256 Spill As 0!
    {1, 1, 1, repStride, repStride, 0}
);
```

**Result**: `rowCount=256` has been cut to `0`, subdoes not perform any calculations, output data error.

### Option 1: Host Side Limit R_max (recommended)

```cpp
// Hostside R_max Calculation
constexpr uint32_t MAX_REPEAT_TIMES = 255;
uint32_t R_max = (UB_SIZE - overheadBytes) / bytesPerRow;
R_max = std::min(R_max, MAX_REPEAT_TIMES);  // Ensure R_max <= 255
```

### Option II: Kernel side in batch processing

```cpp
void SubWithBroadcast(
    AscendC::LocalTensor<float>& dst,
    AscendC::LocalTensor<float>& src0,
    AscendC::LocalTensor<float>& src1,
    uint32_t a0Count,
    uint32_t alignedCols,
    uint32_t rowCount)
{
    constexpr uint32_t MAX_REPEAT = 255;
    uint32_t repStride = alignedCols / BLOCK_ELEMENTS;  // BLOCK_ELEMENTS = 8

    for (uint32_t col = 0; col < a0Count; col += MASK_FP32) {
        uint32_t curMask = std::min(a0Count - col, MASK_FP32);

        // Batch processing
        uint32_t processedRows = 0;
        while (processedRows < rowCount) {
            uint32_t batchRepeat = std::min(rowCount - processedRows, MAX_REPEAT);
            uint32_t rowOffset = processedRows * alignedCols;

            AscendC::Sub<float>(
                dst[rowOffset + col],
                src0[rowOffset + col],
                src1[col],
                curMask,
                batchRepeat,
                {1, 1, 1, static_cast<uint8_t>(repStride), static_cast<uint8_t>(repStride), 0}
            );

            processedRows += batchRepeat;
        }
    }
}
```

---

## API affected

| API Category | API Name | Parameter Limit |
|---------|---------|---------|
| Binary Operations | Sub, Add, Mul, Div, Max, Min | `repeatTime` ≤ 255 |
| One-dollar operation | Exp, Log, Sqrt, Abs, Neg | `repeatTime` ≤ 255 |
| scalar Operations | Muls, Adds, Divs | `repeatTime` ≤ 255 |
| Other | And, Or, Xor, Not | `repeatTime` ≤ 255 |

---

## best practice

| Phase | Checkpoint |
|-----|-------|
| **API Before Usage** | View document confirmation `repeatTime` data type and scope |
| **Host Tiling** | Ensure that `R_chunk_size` / `tileRows` does not exceed the limit |
| **Kernel achieved** | If restrictions are to be exceeded, batch processing logic is to be achieved |
| **Debug** | If an error occurs on R=256, check repitattime spill |

---

## Document reference

- Sub API:`asc-devkit/docs/api/context/Sub.md`
- Other Victor API documents have the same path and search for the API name
