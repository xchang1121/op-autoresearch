---
name: triton-ascend-case-index-histogram
description: "Histogram statistics (histogram) optimisation: pre-sorting + binary search for reduced algorithm complexity (O (n ×m) → O (n log n + m log n), performance increased 19 times, conversion to float32 calling for accelerated sorting of Vec Core hardware for large-scale statistical class operations (500,000 + elements)"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Histogram Histogram Statistical Optimization Case

## Task characteristics
- **Operating type**: histogram statistics, counting the number of times each expert ID appears
- **Data size**: index entry (65536, 8), number of experts 365
- **Characteristics**: need to optimize algorithm complexity from O (n×m) to O (n log n + m log n)

## Optimize 1: Pre-sorting + Double Search

### Error: Simple way: Count all over O (n×m)

```python
count = 0
for i in range(total_elements):  # 524288Secondary
    val = tl.load(indices_ptr + i)
    if val == expert_idx:
        count += 1
```

**Question**: Complexity O (n×m) = 524288 × 365 ≈ 190 million operations

### Correct: Optimized way: pre-sorting + binary search O (n log n + m log n)

```python
# Pre- sorting: O(n log n)
indices_flat = indices.flatten().to(torch.float32)
sorted_indices, _ = torch.sort(indices_flat)

# Triton Kernel inside binary search: execute O(log n) per extrat
@triton.jit
def histogram_kernel(sorted_indices_ptr, splits_ptr, total_elements):
    expert_idx = tl.program_id(0)
    expert_id = expert_idx.to(tl.float32)

    # Two-point search for lower bounds (O(log n), approximately 19 iteratives)
    left, right = 0, total_elements - 1
    start_pos = total_elements
    while left <= right:
        mid = (left + right) // 2
        mid_val = tl.load(sorted_indices_ptr + mid)
        if mid_val < expert_id:
            left = mid + 1
        else:
            if mid_val == expert_id:
                start_pos = tl.minimum(start_pos, mid)
            right = mid - 1

    # Two-point search for upper bounds (similar logic)
    # ...
    count = end_pos - start_pos + 1
```

**Performance comparison**:
- Statistics: 190 million operations
- Pre-sorting + Half Search: about 10 million operations
- **performance improvement: about 19 times**

## Optimizing 2: Float32 type conversion (Vec Core acceleration)

### Error: Simple way: directly using int32

```python
indices_flat = indices.flatten()  # int32
sorted_indices, _ = torch.sort(indices_flat)  # Possible CallAI CPU
```

**Question**: Possible retreat to AI CPU ranking, poor performance

### Correct: Optimized: converted to float32

```python
indices_flat = indices.flatten().to(torch.float32)  # Convert tofloat32
sorted_indices, _ = torch.sort(indices_flat)  # CallVec CoreSort
```

### Optimizing content
- Ascend chip includes AI Core, Vec Core, AI CPU
- Vec Core specifically optimized the sorting of float32, supporting SIMD parallels
- Int32 sorting could go back to AI CPU. It's a poor performance.
- Index value range is much smaller thanfloat32accuracyScope2^23) No loss for conversionaccuracy

### Summary
1. **[Archive optimization]**For statistical class operations, priority should be given to pre-ordering + binary search, reducing O(n×m) complexity to O(n log n + m log n)
2. **[Uternal interface optimized]**On the Ascend platform, for large-scale sorting, the type float32 should be used to call Vec Core hardware acceleration
3. Two-point search by 365 experts could be carried out in parallel, with one specialist per thread being processed independently
