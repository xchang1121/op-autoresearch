# Broadcast - OneDim Branch

> **Applicable scene**: All dimensions of the axis are combined into one dimension, essentially Elementwise (part input may be scalar dim=1)

---

## I. SPECIFIC STATE

| Features | Annotations |
|------|------|
| **Post-axis dimensions** | 1-D |
| **Broadcasting mode** | scalar input prioritizes the TensorScalar interface (Adds/ Muls, etc.) and Duplicate expands when no corresponding interface |
| **Data continuity** | All data continuum, linear processing |
| **Calculated** | 1D vector equal to output |

---

## II. Buffer Planning

```cpp
// LiveNum = all survival nodes (input + output + middle buffer)
// maxDtypeBytes = the largest dtype bytes in the calculation diagram

pipe->InitBuffer(buf, ubFormer * maxDtypeBytes * aliveNum);
```

---

## III. Tiling Parameter Calculation

### 3.1 UB Cut

```cpp
// UB, go DB, 128B alignment.
int64_t ubFormerByte = (ubSize - extraSize) / aliveNum;
int64_t ubFormer = (ubFormerByte / CACHE_LINE) * CACHE_LINE / maxDtypeBytes;
// CACHE_LINE = 128
```

### 3.2 Polynuclear cut

```cpp
int64_t dimLength = outputDims[0];  //   When the axes close   1 V

int64_t ubOuter = ceil(dimLength / ubFormer);
int64_t ubTail = dimLength % ubFormer;  // 0 → ubFormer
int64_t blockFormer = ceil(ubOuter / coreNum);
int64_t blockTail = ubOuter % blockFormer;  // 0 → blockFormer
int64_t blockNum = ceil(ubOuter / blockFormer);
```

### 3.3 MNA

When blockNum is less than half the coreNum, reducing ubFormer doubles the core:

```cpp
if (blockNum < coreNum / 2 && ubFormer * maxDtypeBytes * aliveNum > 8 * 1024) {
    // Try reassigning by coreNum/2
    int64_t dimPerCore = dimLength * 2 / coreNum;
    int64_t alignDimPerCore = ceil_align(dimPerCore * maxDtypeBytes, CACHE_LINE) / maxDtypeBytes;
    ubFormer = min(ubFormer, alignDimPerCore);

    // Lower limit: 8KB per core after DB
    int64_t lowestUbFormer = (8 * 1024 / aliveNum / CACHE_LINE) * CACHE_LINE / maxDtypeBytes;
    ubFormer = max(ubFormer, lowestUbFormer);

    // Recalculate subnucleic parameters
    ubOuter = ceil(dimLength / ubFormer);
    blockFormer = ceil(ubOuter / coreNum);
    blockNum = ceil(ubOuter / blockFormer);
}
```

---

## IV. KERNEL IMPLEMENTS

### 4.1 Data flows

```
GM → DataCopyPad → UB [ubFormer]
  ↓
  scalarInput: Priority TensorScalar Interface(s)Adds/Subs/Muls Wait, no need to move in.
           If there's no match, TensorScalar It's the interface. It's the interface. Duplicate Expand Asvector + TensorTensor Interface
  NotscalarInput: DataCopyPad(inputGm, curLen)
  ↓
Compute (Add/Mul/Sub/... Element by Elements)
  ↓
UB → DataCopyPad → GM
```

### 4.2 scalar input detection

An input after the axis is dim=1 →, which is scalar. Marked with scalarFlag bitmap:

```cpp
// Host Side
int32_t scalarFlag = 0;
for (int i = 0; i < inputNum; i++) {
    if (dims[i][0] == 1) {
        scalarFlag |= (1 << i);
    }
}
```

### 4.3 Core code template

```cpp
__aicore__ inline void Process()
{
    int64_t blockLoopNum = (GetBlockIdx() == blockNum - 1) ? blockTail : blockFormer;
    int64_t offset = GetBlockIdx() * blockFormer * ubFormer;

    for (int64_t i = 0; i < blockLoopNum; i++) {
        int64_t curLen = (i == blockLoopNum - 1 && GetBlockIdx() == blockNum - 1)
                         ? ubTail : ubFormer;

        // CopyIn: non-scalar input with DataCopyPad
        DataCopyPad(input0Local, input0Gm[offset], {1, curLen * sizeof(T), 0, 0});

        // Compute: scalar input priority for TensorScalar interface
        if (scalarFlag & (1 << 1)) {
            // Mode 1 (recommended): Directly for the TensorScalar interface
            Adds(outputLocal, input0Local, scalar1, curLen);
            // Method 2 (dip): no matching for TensorScalar interface
            // Duplicate<T>(input1Local, scalar1, curLen);
            // CustomOp(outputLocal, input0Local, input1Local, curLen);
        } else {
            DataCopyPad(input1Local, input1Gm[offset], {1, curLen * sizeof(T), 0, 0});
            Add(outputLocal, input0Local, input1Local, curLen);
        }

        // CopyOut
        DataCopyPad(outputGm[offset], outputLocal, {1, curLen * sizeof(T), 0, 0});

        offset += ubFormer;
    }
}
```

> **scalar handles priority**: Adds /Subs/Muls/Divs et al. TensorScalar interface > Duplicate + TensorTensor interface.
> The TensorScalar interface saves the Duplicate operation and a Buffer with better performance.

---

## V. common issue

| Problem | Reason | Solutions |
|------|------|---------|
| scalar input result error | scalarFlag Calculator Error | Check if the dims [i][0] are 1 after combining the axes |
| Low utilization of nuclear weapons | ubFormer is too big to block Num. | Enable multi-nuclear optimization (bbFormer) |
| Non-match data error | DataCopy does not support non-matching | Use DataCopyPad |
