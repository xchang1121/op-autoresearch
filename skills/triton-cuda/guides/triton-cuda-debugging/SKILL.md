---
name: triton-cuda-debugging
description: "Triton CUDA debugging checklists and common error tabulations, including compilation errors, runtime errors, accuracy questions and performance questions. The debugging scenes that apply to CUDA internal nuclear code errors require the reason for the error or need to verify the correctness of the code"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# Debugging and queuing lists

## Full debug list

### Memory access issues
- [ ] Do all the loads/store have a mask or baseary_check?
- [ ] Is the frame parameter setting correct?
- [ ] Is the array index crossed?
- [ ] Did you use `.contiguous()` to ensure continuity of memory?
- [ ] 2D data using `tl.make_block_ptr`?
- [ ] Do memory access merge (codesced)?

### Control flow check
- [ ] Did you miss /break/continue?
- [ ] Whether complex conditions are combined with a mask?
- [ ] Are `tl.constexpr` used only for kernel parameters?
- [ ] Ambda expression (not supported)?

### Grid and Block Configuration Check
- [ ] BLONK_SIZE is a 2-year-old?
- [ ] Are the num_warps reasonable (2-8)?
- [ ] Are num_stages reasonable (2-5)?
- [ ] Did Grid's total size not exceed the hardware limit?

### Conjunctive with Atomic Operations Inspection
- [ ] Did you use atomic operations (`tl.atomic_add`, etc.) for co-writing?
- [ ] Is atomic operation necessary (is it avoidable)?
- [ ] Is there a data competition (multiple programs are co-located)?

### Performance optimization check
- [ ] Did you use autotune?
- [ ] Grouped Ordering?
- [ ] Do you want to use float32 for intermediate accumulation?
- [ ] Does the Reduce operation have numerical stability processing?

## Common Error Spacing

### Compiler error

| Error Type | Typical symptoms. | Common causes | Solutions |
|---------|---------|---------|---------|
| **Return statement** | Compiled failed | Use Kernel to return | Remove return, use mask instead |
| **Break/Continue** | Compiled failed | Control flow jump is not supported | Use mask or recreate logic |
| **Lambda Expression** | Compiled failed | Unsupported lmbda | Change to a normal function or inline |
| **Type error** | Compiled failed | Constexpr type does not match | Check tl.constexpr declaration |

### runtime error

| Error Type | Typical symptoms. | Common causes | Solutions |
|---------|---------|---------|---------|
| **Memory crossed borders** | CUDA error | Missing Mask | Add a mark or baseary_check |
| **shape mismatch** | Dimension Error | stride calculation error | Check stride parameters |
| **Illegal memory access** | Segfault | Pointer Calculator Error | Validate offset calculation |
| **shared memory spill** | Launch failed | Num_stages are too big. | Reduction of num_stages |

### Numeric Error

| Error Type | Typical symptoms. | Common causes | Solutions |
|---------|---------|---------|---------|
| **NaN/Inf** | Turned out to be unusual. | Softmax Spill | Minus maximum |
| **accuracy losses** | The results are inaccurate. | Full use of fp16 add | Use float32 cumulative |
| **Debug Zero** | NaN | Difference or zero | Add eps |
| **Countries of negative numbers** | NaN | Square difference is negative | `tl.maximum(var, 0.0)` |

### Performance issues

| Type of problem | Typical symptoms. | Common causes | Solutions |
|---------|---------|---------|---------|
| **Poor performance** | Slower than PyTorch | Unused autotune | Add autotune |
| **bandwidth Low** | Memory restricted | Non-consolidated visits | Ensuring joint visits |
| **Occupancy Low** | Low utilization of GPU | Storer/shared memory Overlimit | Reduce BLONK_SIZE |
| **L2 Cache** | MatMul has low performance | Unused Grouping | Add L2 Cache Optimization |

## Classification debugging process

### 1. Compiled failed

**Steps**
1. Check keywords in error message (return, break,lambda)
2. See if unsupported syntax is used
3. Refer to the "API use limit" part to change the code

**Common restorations**:
```python
# Error: use return
@triton.jit
def kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return  # Compiler error!
    # ...

# Correct: use mask
@triton.jit
def kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    data = tl.load(ptr + offsets, mask=mask, other=0.0)
    # All codes are at the same level.
```

### 2. runtime crash

**Steps**
1. Add all load/store's mask
2. Check Grid and BLONK_SIZE Configuration
3. Validates whether the stide parameter is correct
4. Use small data testing
5. Check if shared memory is beyond limit (reduction num_stages)

**Debug techniques**:
```python
# Print debug information (host side)
print(f"Grid: {grid}, BLOCK_SIZE: {BLOCK_SIZE}")
print(f"Shape: {input_tensor.shape}, Stride: {input_tensor.stride()}")
print(f"Contiguous: {input_tensor.is_contiguous()}")
```

### 3. It didn't work right.

**Steps**
1. Check for numerical stability (if Softmax minus max)
2. Verify excise accuracy (use or not of float32)
3. Check border processing (mask correct)
4. Compare small hand count results

**Certification method**:
```python
# Compare to PyTorch Native
output_triton = model_new(x)
output_torch = torch.softmax(x, dim=-1)  # or other primary realization
diff = (output_triton - output_torch).abs().max()
print(f"Max diff: {diff.item()}")
assert diff < 1e-5, "Results mismatch!"
```

### 4. Poor performance.

**Steps**
1. Add autotune search optimal configuration
2. Check for conversion to continuous memory (`.contiguous()`)
3. Confirm if memory visits are merged
4. Checking L2 cache optimization (Grouped Ordering)
5. Use Nsight Compute analysis

**profiling**:
```python
import time

# Preheat
for _ in range(10):
    _ = model(x)

# Test
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = model(x)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Average time: {elapsed/100*1000:.2f} ms")
```

## Example of error fixes

### Example 1: Softmax spill

**Error code**:
```python
numerator = tl.exp(x)  # Possible spills
```

**Rehabilitation**:
```python
max_val = tl.max(x, axis=0)
x_stable = x - max_val
numerator = tl.exp(x_stable)
```

### Example 2: Non-merger access

**Error code**:
```python
# Every thread jump access
offsets = pid + tl.arange(0, BLOCK_SIZE) * stride
data = tl.load(ptr + offsets)
```

**Rehabilitation**:
```python
# Continuous visits
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets, mask=offsets < n)
```

### Example 3: shared memory spill

**Error code**:
```python
triton.Config({...}, num_stages=8, num_warps=8)  # shared memoryNot enough.
```

**Rehabilitation**:
```python
triton.Config({...}, num_stages=3, num_warps=4)  # Reduction stage Number
```

## Debug Tool

### 1. Nsight Compute

```bash
# Analyse Kernel Performances
ncu --set full python script.py

# Analyse memory bandwidth
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed python script.py
```

### 2. CUDA-MEMCHECK

```bash
# Check memory error
compute-sanitizer python script.py
```

### 3. Use small data testing

```python
# Big data is difficult to debug. Use small data first.
x_small = torch.randn(4, 8, device='cuda', dtype=torch.float16)
output = model(x_small)
print(output)  # Manual validation results
```

### 4. Compare reference implementation

```python
# Always compare with PyTorch native
torch.testing.assert_close(output_triton, output_torch, rtol=1e-4, atol=1e-5)
```

## Summary

Debug the Triton-CUDA code key:
1. **Compliance**: not using return/break/continue/lambda
2. **Memory security**: all visits add mask
3. **Stabilization**: Softmax minus maximum, float32 cumulative
4. **Consolidated access**: ensure a continuous address for the same warp insider
5. **Performance optimization**: using autotune, Grouped Ordering

**best practice**: First ensure correctness, then optimize performance!
