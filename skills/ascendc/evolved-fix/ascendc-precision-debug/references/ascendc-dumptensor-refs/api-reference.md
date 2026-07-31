# DumpTensor API Reference

## API Signature

```cpp
DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
```

## Parameters

| Parameter  | Type              | Description                                  |
|------------|-------------------|----------------------------------------------|
| `tensor`   | `LocalTensor<T>&` | The tensor to dump                           |
| `desc`     | `uint32_t`        | Unique identifier (use systematic numbering) |
| `dumpSize` | `uint32_t`        | Number of elements to dump                   |

## Desc Numbering Convention

| Range   | Stage                |
|---------|----------------------|
| 100-199 | Input tensors        |
| 200-299 | Intermediate results |
| 300-399 | Output tensors       |

Increment by 10 within each range.

## Best Practices

```cpp
// Control dump size - dump subset for large tensors
uint32_t dumpSize = std::min(tileLength, 32u);
DumpTensor(outputLocal, 300, dumpSize);

// Avoid dumping entire large tensors
DumpTensor(outputLocal, 300, 8192);  // ❌ Too much
DumpTensor(outputLocal, 300, 32);    // ✅ Better
```

## Important Notes

- DumpTensor outputs to system log
- Remove dumps in production code (performance overhead)
- Start with 32-64 elements, increase only if needed

---

## Call constraints

| Constraints             | Annotations                                                                       |
|------------------|----------------------------------------------------------------------------|
| Call Context       | Call to `LocalTensor` only in the Kernel function; `GlobalTensor` cannot be directly dump      |
| Synchronization request         | Data depends on handling/calculation to be completed. Dump, as detailed in parent document "Use trap §1 "                |
| dumpSize ceiling    | Can't exceed the actual number of elements of a tensor.                                     |
| dtype support       | Common `half / float / int32_t / bfloat16_t` supported, special dtype based on CANN version |
| Performance Impact         | Significant time-sequencing costs, possibly changing pipeline behaviour, must be removed when positioning is complete                         |

## dsc Number Extension

The base three-part formula (100/200/300) is enough to cover a single nucleotide operator. Complex operator suggests extension:

```
desc = base + blockOffset + stageOffset + iterOffset

base       : 100=Input, 200=Centre, 300=Output
blockOffset: GetBlockIdx() * 1000     // Multinuclear separation
stageOffset: 0/10/20...               // Multiple postpoints in the same paragraph
iterOffset : tileIdx                   // The same. tile Multiple inverted distinctions
```

Example: core 1, 2nd tile, 2nd Plugin Point in Compute:
```cpp
uint32_t desc = 200 + GetBlockIdx() * 1000 + 20 + tileIdx;
DumpTensor(midLocal, desc, 32);
```

## Relationship to PRINTF / printf

| API           | Call Location          | Purpose                                    |
|---------------|-------------------|----------------------------------------|
| `DumpTensor`  | Inner NPU Kernel     | Batch the LocalTensor element value              |
| `PRINTF`      | Inner NPU Kernel     | Look at scalar, control stream, tile.           |
| `printf`      | Host / CPU simulation   | Look at the host side buffer, CPU golden output     |
| `AscendC::Simt::printf` | SIMT VF inside | SIMT operator kernel printing (see SKILL SIMT section of father)|

Debug tensor data with DumpTensor, debug control streams (tileNum, blockIdx, loop count) to PRINTF.

## Common pedals.

- **dumpSize is too dead**: logs are flooded and dump itself is time-consuming to shield bug → default 32, position shrink and add
- **Multiple tile loop only last**: each tile is numbered separately, otherwise the back dump overwrites the front view
- **Modified Dump fully agreed**: doubt kernel cache, `rm -rf build/ $HOME/atc_data/kernel_cache/`
- **NAN appears in an uncertain position**: add paragraph by paragraph from input and find the first appearance of NAN
