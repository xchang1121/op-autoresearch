---
name: triton-ascend-case-index-put
description: "Index value (index_put) optimization: Batch load index data into the UB cycle and reuse through Get_election (duplicate access to global memory) to significantly reduce memory access to latency for irregular memory access scenarios requiring multiple access to the same data in the cycle"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Index Put Index-Authorized Cases

## Task characteristics
- **Operating type**: indexed value, data written to target buffer based on index mapping
- **Data size**: input fractions (16,384, 4), group buffer zones (8,65536)
- **Characteristics**: irregular memory access, subject to element-by-fact processing to avoid writing conflicts

## Optimization: Batch Load + Data Reuse

### Error: Simple way: double load in cycle

```python
for i in tl.range(0, BLOCK_SIZE):
    if start_idx + i < total_elements:
        # Load index from global memory for each cycle
        unit_idx = tl.load(unit_indices_ptr + start_idx + i)
        pos_idx = tl.load(position_map_ptr + start_idx + i)
```

**Question**: global memory, latency is highly effective and inefficient for each cycle.

### Correct: Optimized: Batch load to UB, recycle

```python
# Out of circulation: load a piece of index data to UB (unified buffer zone)
unit_indices_tile = tl.load(unit_indices_ptr + offsets, mask=mask, other=0)
position_map_tile = tl.load(position_map_ptr + offsets, mask=mask, other=0)

# Loop: take numbers from UB through Get_election, reuse data
for i in tl.range(0, BLOCK_SIZE):
    if start_idx + i < total_elements:
        # Take the number from the UB and avoid accessing global memory
        unit_idx = tl.get_element(unit_indices_tile, [i])
        pos_idx = tl.get_element(position_map_tile, [i])
        # Follow-up...
```

### Optimizing content
- Outside the cycle, load the entire BLONK_SIZE index data batch to UB by an `tl.load` operation
- In the cycle, remove index values from the UB individually by `tl.get_element`
- Convert multiple global memory visits into one batch load + multiple chip caches
- Considerable reduction of memory access latency

### Summary
**[Universal Optimization]**When multiple accesss to the same data are required in the cycle, load the cache (UB) on the plate first in bulk, then re-use the data by getting_election on a case-by-case basis, reduce the number of global memory visits and enhance performance.
