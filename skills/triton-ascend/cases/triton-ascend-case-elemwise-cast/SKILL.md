---
name: triton-ascend-case-elemwise-cast
description: "Great Shape Type Conversion (int8→fp16) Optimization: Increase UB utilization through double-segregation (BLONK_SIZE+TILE_SIZE) with optimal performance for 2048, elementwise type conversion scenarios for larger (millions of elements) Shape"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Int8 to FP16 type conversion optimization cases

## Task characteristics
- **Operating type**: Elementwise, Type Conversion
- **Data size**: (128, 1024, 1024), Shape is larger
- **data type**: input int8, output fp16
- **Task characteristics**: can be sequenced in axle (which can flatten is an axis), parallel in the outer layer, and vector in the inner layer, with multiple cuts to be considered if the UB does not exist

## Optimization: binary split + use full UB

```python
# Triton kernel realization: Split BLONK_SIZE into sections, and move Tile_SIZE size data each time
configs = [
    triton.Config({"BLOCK_SIZE": 65536, "TILE_SIZE": 65536}), # Numerical2048, It's the best. Full.UBAnd there's no double cut.
    triton.Config({"BLOCK_SIZE": 65536, "TILE_SIZE": 32768}), # Numerical2048,butUBUnused
    triton.Config({"BLOCK_SIZE": 2097152, "TILE_SIZE": 65536}), # Numerical64, Low Parallel
    triton.Config({"BLOCK_SIZE": 4194304, "TILE_SIZE": 65536}), # Numerical32, The parallels are even lower.
]

# kernel operation:
block_start = pid * BLOCK_SIZE
for i in range(0, BLOCK_SIZE, TILE_SIZE):
    offsets = block_start + tl.arange(0, TILE_SIZE)
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask)
    output_data = tl.cast(input_data, tl.float16)
    tl.store(output_ptr + offsets, output_data, mask=mask)
```

### Optimizing content
- Triton kernel uses the foror cycle to try to double-slit each time Tile_SIZE size data is moved to increase UB utilization
- Raise the number of cores within a given range and try to use the full UB.
- No double-several-time performance in core (BLONK_SIZE = Tile_SIZE = 65536)

### Summary
1. In order to obtain better performance when the data is larger, the cut value is as large as possible to be severed by the size of the sape.
2. For a simple Elementwise operation, the elements of the multiple axes are expanded into a single axis, and they are split on this axis.
3. Allocating block to each thread block, with multiple cut-outs to be considered if the UB does not exist
