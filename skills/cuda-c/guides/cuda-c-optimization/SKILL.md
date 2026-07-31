---
name: cuda-c-optimization
description: "CUDA C performance optimization, numerical stability and debugging"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: cuda_c
structure:
  child_skills:
    - cuda-c-patterns
---

# CUDA C Performance Optimization Guide

## 1. Performance Optimization Policy

### 1.1 Block-size selection policy

- **Base**: use 2 thorium (128, 256, 512, 1024)
- **Recommended**: 256 or 512 threads per block
- **Restrict**: maximum 1024 threads per block (most GPU)
- **Modified**: Balancing parallelity and resource consumption to avoid oversized or too small

| operator Type | Recommended block size | Grid Configuration |
|---------|-----------|---------|
| Element-wise | 256 / 512 | One-dimensional. |
| Reduce | 256 | 1-D +shared memory |
| MatMul | dim3 (16, 16) or dim3 (32, 32) | Two-dimensional |
| Image Processing | dim3(16,16) | Two-dimensional |

### 1.2 Memory access optimization

#### Merge Access (Coalesced Access)
Continuous line access to a continuous memory address, and GPU consolidates multiple requests into a small number of memory services.

```cuda
// ✅ combined access (continuous thread access to continuous address)
__global__ void coalesced(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Continuous visits
    }
}

// ❌ non-merger access (jumping access)
__global__ void strided(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * stride < n) {
        data[idx * stride] = data[idx * stride] * 2.0f;  // Jump Access
    }
}
```

#### Alignment visits
The data are aligned to 128 bytes to increase the utilization of the memory bandwidth.

#### Avoid Bank Conflict
shared memory consists of 32 banks, avoiding multiple paths within the same warp to access the same bank.

```cuda
// ✅ No bank conflict
__shared__ float s[256];
s[threadIdx.x] = input[idx];  // Continuous linear access bank

// ❌ Bank Conflict
s[threadIdx.x * 32] = input[idx];  // All thread access is the same. bank
```

### 1.3 Calculation optimization

#### Avoid branching.
The same control path should be implemented for 32 threads within the same warp.

```cuda
// ❌ branch dispersed: different paths for the same warp inner range
if (threadIdx.x % 2 == 0) {
    // even-number line path
} else {
    // An odd-numbered route path
}

// ✅ Use Conditional Value Substitute Branch
float result = (threadIdx.x % 2 == 0) ? value_a : value_b;
```

#### Use built-in fast mathematical functions
```cuda
// Standard accuracy
float r = expf(x);

// ✅ Fast Version (accuracy slightly lower but faster)
float r = __expf(x);
float r = __logf(x);
float r = __sinf(x);
```

#### Atom Reduction Operations
Use block-incorporation instead of global atomic operations as much as possible.

```cuda
// ❌ Mass Atom Operations
atomicAdd(&global_sum, local_val);

// ✅ internal union, then atom writing back
__shared__ float sdata[256];
sdata[tid] = local_val;
__syncthreads();

// Internal Convention
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}

// Only one atomic operation.
if (tid == 0) atomicAdd(&global_sum, sdata[0]);
```

### 1.4 Occupancy Optimization

- **Repositor used**: Reduced usage of register per thread, increased number of blocks distributed
- **shared memory**: Rational use of shared memory, not exceeding hardware limitations
- **Block Size**: Select the block size that will remove the maximum number of SM threads

## 2. Numerical stability technique

### 2.1 Spill-proofing

```cuda
// Softmax numeric stabilization: subtract maximum first
float max_val = -INFINITY;
for (int i = 0; i < n; i++) {
    max_val = fmaxf(max_val, input[i]);
}

float sum = 0.0f;
for (int i = 0; i < n; i++) {
    sum += __expf(input[i] - max_val);
}

float result = __expf(input[idx] - max_val) / sum;
```

### 2.2 accuracy Upgrade

- **Intermediate calculation**: Use `float` type sufficient (no need to upgrade to `double`)
- **Accumulation operation**: using a high accuracy loader to prevent the loss of accuracy
- **Avoiding Numeric Overlay**: Check for exclusion of zero and negative openings

```cuda
// Division of security
float safe_div = (denominator != 0.0f) ? numerator / denominator : 0.0f;

// Safe.
float safe_sqrt = sqrtf(fmaxf(variance + eps, 0.0f));
```

### 2.3 At the beginning of the defence value

```cuda
// Ensure non-negative before variance is calculated
float variance = fmaxf(var_computed, 0.0f);
float std = sqrtf(variance + eps);
```

## 3. Programming constraints and best practice

### 3.1 Rules to be followed

- **Boundary check**: all arrays must check the border before visiting
- **Error check**: Check return value after every CUDA API call
- **Memory alignment**: ensure that data are aligned to the appropriate boundaries
- **SwireSync**: shared memory must be `__syncthreads()` before and after use

### 3.2 Principles of kernel design

- **Single function**: Each kernel does one thing only.
- **Parameters Simple**: Avoid complex data structure transfer
- **Memorial Locality**: access to adjacent memory as much as possible
- **Avoid dynamic distribution**: kernel does not use `malloc` / `new`

### 3.3 Code instruction

- Add sufficient comments to explain the calculation logic
- Use descriptive variable names
- Keep kernel simple and clear.
- Uniform error-processing mode

### 3.4 ⚠ ️ prohibition

- **Test code banned**: kernel code generated does not contain test clips
- **Print statement banned**: not using `printf()`
- **Bans abnormally eject**: does not use `throw std::runtime_error()` etc.
- **Ban on dynamic distribution**: kernel does not use `malloc` / `new`

## 4. Debugging and queuing lists

### Memory access issues
- [ ] Do all array visits have border checks?
- [ ] Is the memory correctly allocated and released?
- [ ] Are the copies of host-device memory in the right direction?
- [ ] Is the pointer within the limits of effectiveness?

### kernel implementation issues
- [ ] Are the grids and blocks reasonable in size?
- [ ] Is kernel activation parameters correct?
- [ ] Could not close temporary folder: %s
- [ ] Is device memory enough?

### Performance issues
- [ ] Whether the size of the block is 2 or not?
- [ ] Do memory access merge?
- [ ] Has the branch been avoided?
- [ ] Is the use of shared memory efficient?
- [ ] Is there an unnecessary atomic operation?

### Sync Problem
- [ ] Do you have `__syncthreads()` before and after reading and writing shared memory?
- [ ] Can all threads reach the sync point (avoiding the dead lock)?
- [ ] Do you need `__threadfence()` to ensure global visibility?

## 5. Frequent Error Scanning

| Error Type | Symptom | Solutions |
|---------|------|---------|
| Cross-border visits | runtime error or abnormal result | Add Border Check `if (idx < n)` |
| Memory Leak | Process memory continues to grow | Check `cudaFree` call |
| Sync Error | Uncertainty/inconsistencies | Add `__syncthreads()` |
| Type does not match | Compiler error | Check data type conversion |
| device memory is inadequate | kernel activation failed. | Decrease block size or batch processing |
| Bank Conflict | Poor performance. | Adjust shared memory access mode |
| Branches spread out. | Poor performance. | Use Conditional Value Substitute Branch |
| Non-consolidated visits | Low utilization of memory bandwidth | Adjust data layout and access mode |

## Summary of best practice

1. **Authority before performance**: ensuring kernel correctness before optimization
2. **Merge memory access**: Continuous thread access to continuous address
3. **Rational block size**: use 256 or 512, a 2-bit
4. **Avoid branching**: Same warp inner process follows the same path
5. **Use shared memory**: Cache data frequently accessed
6. **Decreasing atom operations**: pre-correspondence before global writing back
7. **Value stable**: spill-proofing, safe partitioning and opening
8. **JIT integration**: using `load_inline` integration with PyTorch
