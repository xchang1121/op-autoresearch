# ElWise - Tiling Detailed Calculating

> **Applicable scene**: Input Output Shape is identical, element by element

---

## I. Multi-nuclei (blockFormer / blockNum)

### Core thinking

Ensure that the amount of data per nuclear process > = minimum threshold and is aligned to 512 bytes.

### Tiling Parameter Description

| Parameters | Meaning | Formula |
|------|------|---------|
| `dim0` | Total number of elements | Multiplication of all dimensions |
| `coreNum` | Actual number of cores used | `min (calculated core, maximum number) ' |
| `blockFormer` | Number of basic elements per nuclear (matching to 512 elements) | `((dim0 / coreNum) + 511) / 512 * 512` |
| `blockNum` | Virtual block number | `(dim0 + blockFormer - 1) / blockFormer` |

### Calculating steps

```cpp
// Step 1: Calculating Numeric Number (Ensure that at least 4KB data are processed per core)
// Note: MIN_TILING_BITS_SIZE_PER_CORE and minDtypeBits are bits
coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS_SIZE_PER_CORE - 1) /
           MIN_TILING_BITS_SIZE_PER_CORE;
coreNum = min(coreNum, availableCoreNum);

// Step 2: Number of basic elements per nuclear aligned to 512 elements
blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1) /
               ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;

// Step 3: Compute virtual block numbers
blockNum = (dim0 + blockFormer - 1) / blockFormer;
```

### Inter-nuclear data deflection

```cpp
template <typename DataType>
__aicore__ inline int64_t CalcBlockOffset() {
    return blockFormer * GetBlockIdx() * sizeof(DataType);
}
```

---

## II. UB Severation (ubFormer /ubLoop / tail)

### Core thinking

Ensures that the UB processing volume is twice the integer number of 256B (Vector command optimal).

### Tiling Parameter Description

| Parameters | Meaning | Formula |
|------|------|---------|
| `ubFormer` | Base size for each UB block (256B alignment) | Align to 256B. |
| `ubLoopOfFormerBlock` | Number of UB cycles in the first block | `(blockFormer + ubFormer - 1) / ubFormer` |
| `ubTailOfFormerBlock` | End size of the first block | `blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer` |
| `ubLoopOfTailBlock` | Number of UB loops in tail block | `(blockTail + ubFormer - 1) / ubFormer` |
| `ubTailOfTailBlock` | End size of tail block | `blockTail - (ubLoopOfTailBlock - 1) * ubFormer` |

### Calculating steps

```cpp
// Step 1: Calculate the maximum number of elements that UB can accommodate
bufferDivisor = bufferNum * elemBytes;
maxElemNum = (ubSize - extraSize) * 8 / bufferDivisor;

// Step 2: Align with 256B
alignFactor = REPEAT_BYTES * 8 / minDtypeBits;  // Like FP32 = 64 Elements
ubFormer = (maxElemNum / alignFactor) * alignFactor;

// Step 3: Calculate the number of cycles
ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
```

---

## Kernel Implementation Model

### Process Loop Structure

**Core logic**: distinction between head and tail block because their UB cycles may vary in number.

```cpp
// 1. Determination of whether or not the current treatment is the last
bool isLastBlock = (blockIdx == blockNum - 1);

// 2. Retrieving current block loops and tail sizes
//    The number of cycles/tail sizes of the first block and tail block may be different
loopNum = isLastBlock ? ubLoopOfTailBlock : ubLoopOfFormerBlock;
tailNum = isLastBlock ? ubTailOfTailBlock : ubTailOfFormerBlock;

// 3. Main cycle (process complete UB blocks)
for (uint64_t i = 0; i < loopNum - 1; i++) {
    ProcessTile(offset, ubFormer);
    offset += ubFormer;
}

// 4. End processing (processing of the last incomplete UB block)
ProcessTile(offset, tailNum);
```

**Why do you distinguish the head/tail, block?**

- `blockFormer` is aligned with 512B and may be slightly greater than the average distribution
- Last block allocated raw data `blockTail` may be less than `blockFormer`
- `ubLoopOfFormerBlock` ≠ `ubLoopOfTailBlock`, `ubTailOfFormerBlock` ≠ `ubTailOfTailBlock`

---

## Minimum implementable Tiling template

```cpp
struct TilingData {
    int64_t dim0;           // Total number of elements
    int32_t coreNum;        // Actual
    int64_t blockFormer;    // Basic data volume per nuclear unit
    int64_t blockNum;       // block Number
    int64_t ubFormer;       // UB Base Size
    int64_t ubLoopOfFormerBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfTailBlock;
};

TilingData ComputeTiling(int64_t dim0, int64_t elemBytes, int64_t ubSize,
                         int64_t bufferNum, int64_t availableCoreNum) {
    TilingData tiling;

    // Constant
    constexpr int64_t MIN_TILING_BITS = 32768;       // 4KB, units bits
    constexpr int64_t ELEM_ALIGN_FACTOR = 512;       // Multiple nucleotide element alignment factor
    constexpr int64_t ALIGN_256 = 256;               // UB Align bytes

    // Multi-nuclear cut (minDtypeBits = elemBytes *8)
    tiling.coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS - 1) / MIN_TILING_BITS;
    tiling.coreNum = std::min(tiling.coreNum, availableCoreNum);

    tiling.blockFormer = ((dim0 + tiling.coreNum - 1) / tiling.coreNum + ELEM_ALIGN_FACTOR - 1) / ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
    tiling.blockNum = (dim0 + tiling.blockFormer - 1) / tiling.blockFormer;

    // 2. UB Cut
    int64_t bufferDivisor = bufferNum * elemBytes;
    int64_t maxElemNum = (ubSize * 8) / bufferDivisor;
    int64_t alignFactor = ALIGN_256 * 8 / elemBytes;
    tiling.ubFormer = (maxElemNum / alignFactor) * alignFactor;

    // 3. Frequency of cycles
    tiling.ubLoopOfFormerBlock = (tiling.blockFormer + tiling.ubFormer - 1) / tiling.ubFormer;
    tiling.ubTailOfFormerBlock = tiling.blockFormer - (tiling.ubLoopOfFormerBlock - 1) * tiling.ubFormer;

    int64_t blockTail = dim0 - (tiling.blockNum - 1) * tiling.blockFormer;
    tiling.ubLoopOfTailBlock = (blockTail + tiling.ubFormer - 1) / tiling.ubFormer;
    tiling.ubTailOfTailBlock = blockTail - (tiling.ubLoopOfTailBlock - 1) * tiling.ubFormer;

    tiling.dim0 = dim0;
    return tiling;
}
```

---

## V. SUMMARY OF EXPERIENCES

| Experience | Annotations | Code Mode |
|------|------|----------|
| **Minimal particle size** | At least 4KB data per nuclei, otherwise it's not worth nuclei | `MIN_TILING_BITS = 32768` |
| **Multi-check** | Number of elements aligned to the multiple of 512 | `blockFormer = (original + 511) / 512 * 512 ` |
| **UB Alignment** | Align with 256B to ensure the effectiveness of the Vector command | `ubFormer = (original value / initialfactor) *alignFactor ', of which signFactor = 256 / elemBytes (alignFactor = 64 if FP32 = 4 bytes) |
| **Trans-nuclear offset** | Current nuclear GM offset = `blockFormer * blockIdx` | `CalcBlockOffset()` |

---

## VI. Dtype Branch: FP16/BF16 L ' UB budget for accuracy (Add/Sub)

**When to start**: Step 2 of patterns.md decided to use the hit-up accuracy branch.

### 1. Buffer planning variance

The lifting accuracy branch is additional to the original dtype Queue, introducing Buffer**in the middle of**K 's `ubFormer * sizeof(float)` 32 FP32. K is determined by aliases on the API level, and the tiling phase is simply used as a parameter into the UB budget.

| Item | Original dtype | accuracy branch |
|---|--------------|-----------|
| Buffer dtype | Old dtype (half/bf16) | Old dtype (No change) |
| FP32 intermediate Buffer copies | — | **K**(as given by API aliases) |
| Single FP32 Medium Size | — | `ubFormer * sizeof(float)` |
| Total UB Occupation | `bufferNum * ubFormer * elemBytes` | `bufferNum * ubFormer * elemBytes + K * ubFormer * sizeof(float)` |

### 2. ubFormer calculation adjustments

`bufferDivisor` under the accuracy branch shall contain both semi-accuracy Queue and FP32 Buffer components:

```cpp
// A straight-calculation branch
bufferDivisor = bufferNum * elemBytes;

// Upgrade accuracy branch: BufferNum semi accuracy + K FP32
bufferDivisor = bufferNum * elemBytes + K * sizeof(float);

maxElemNum = (ubSize * 8) / bufferDivisor;
alignFactor = ALIGN_256 * 8 / elemBytes;       // Align still by input dtype
ubFormer = (maxElemNum / alignFactor) * alignFactor;
```

The extraction value for K depends on whether the opt-in API phase supports the dst/src aliases, which are given by API to give details.

> accuracy branch does not add a TilingData field, and the Kernel side selects the branch staticly by dtype template parameters. Specific Cast/Add/Sub calls, RoundMode, alias API realization details, Tiling not involved.

---
