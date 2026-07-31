---
name: tilelang-cuda-optimization
description: "TileLang CUDA performance optimizes a generic policy, best practice and debugging techniques. This applies to the generation and optimization of kernel codes that require upgrading of TileLang internal nuclei, checking in case of compilation/run error, or knowledge of TileLang platform limitations"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
structure:
  child_skills:
    - tilelang-cuda-memory
    - tilelang-cuda-synchronization
    - tilelang-cuda-gemm
---

# TileLang CUDA Performance Optimization Guide

## 1. Performance Optimization Policy

### 1.1 Segment size selection

- **Principle**: Balancing parallelity and resource occupation
- **Recommend**: use 2 quartz
- **Common use**: block_M/block_N = 64, 128, 256; block_K = 16, 32, 64

| operator Type | Recommended block size | Threads |
|---------|-------------|-------|
| Element-wise | block = 256-1024 | 128-256 |
| GEMM | block_M=128, block_N=128, block_K=32 | 128 |
| Reduce | block = 256-512 | 128-256 |

### 1.2 Software pipeline Optimization

```python
def pipelined_computation():
    # Select the appropriate pipeline depth
    num_stages = 3  # Usually. 2-4 It's the best stage.

    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
        # Overlap memory operations and calculations
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[ko * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)
```

**pipeline Depth selection**:
- `num_stages=2`: Minimum shared memory usage
- `num_stages=3`: Usually best (recommended default value)
- `num_stages=4`: More overlap but more shared memory
- `num_stages=5+`: Could exceed the shared memory limit

### 1.3 Parallelization strategies

```python
# 1. Coincidence of fine particles
for i, j in T.Parallel(block_M, block_N):
    # AutoMap to Thread
    pass

# 2. vector optimization
for k in T.vectorized(TILE_K):
    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]

# 3. Serial cycle (used when necessary)
for k in T.serial(block_K):
    # Order execution
    pass
```

### 1.4 data type Optimization

```python
# 1. Use mixed accuracy
input_dtype = "float16"    # Enter Data
accum_dtype = "float"      # Increased use of compressoraccuracy

# 2. Type conversion optimization
result = A[i].astype(accum_dtype) * B[i].astype(accum_dtype)

# 3. Avoid unnecessary type conversion (Uniform conversion before calculation)
```

## 2. Memory Optimization Policy

### 2.1 Memory level structure optimization

```python
def memory_optimized_matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 1. shared memory Allocation - Cache data for frequent access
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")

            # 2. Depositor segment distribution - cumulative and temporary storage
            C_local = T.alloc_fragment((block_M, block_N), "float")

            # 3. Enable swizzle to enhance the L2 cache locale
            T.use_swizzle(panel_size=10, enable=True)

            T.clear(C_local)

            # 4. Software pipeline optimized memory bandwidth
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

### 2.2 Memory access mode optimization

```python
# 1. Load with vector
for k in T.vectorized(TILE_K):
    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]

# 2. Consolidation of memory access
T.copy(A[start:end], A_shared)  # Automatically merge access

# 3. Optimizing the use of layout to avoid bank conflicts
from tilelang.intrinsics import make_mma_swizzle_layout
T.annotate_layout({
    A_shared: make_mma_swizzle_layout(A_shared),
    B_shared: make_mma_swizzle_layout(B_shared),
})
```

### 2.3 L2 Cache Optimization

```python
# Enable rasterization to improve L2 cache locality
T.use_swizzle(panel_size=10, enable=True)
```

## 3. Sync Security Optimization

### 3.1 Use of built-in attribution functions (recommended strongly)

```python
# ✅ recommended: use an internal attribute function without manual synchronization
T.reduce_sum(input_tensor, output_tensor, dim=axis)
T.reduce_max(input_tensor, output_tensor, dim=axis)
T.reduce_min(input_tensor, output_tensor, dim=axis)
T.reduce_mean(input_tensor, output_tensor, dim=axis)
```

### 3.2 Avoidance of manual return

```python
# ❌ absolutely forbidden: date by hand leads to death of thread card
while stride > 0:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    T.sync_threads()  # The risk of a dead lock.
    stride //= 2

# ✅ Correct: Use internal union
T.reduce_sum(input, output, dim=1)  # Synchronising folder
```

### 3.3 Recommended parallel mode of calculation

```python
# 1. Parallel calculations using T. Parallel
for i, j in T.Parallel(M, N):
    result[i, j] = input[i, j] * scale[i]

# 2. vector conversion using T.vectorized
for i in T.vectorized(N):
    result[i] = input[i] * scale

# 3. Use T. Pipelined for pipeline
for k in T.Pipelined(K, num_stages=3):
    T.copy(A[k], shared_A)
    T.gemm(shared_A, shared_B, result)
```

## 4. Numerical stability

### 4.1 Spill protection

```python
# Softmax numerical stabilization
T.fill(scores_max, -T.infinity("float"))
T.reduce_max(acc_s, scores_max, dim=1)
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
```

### 4.2 accuracy Upgrade

- **excrete with float32**: even if input is float16/bflota16
- **Final re-conversion**: returns target accuracy after calculation is completed

```python
# Use high accuracy
C_local = T.alloc_fragment((block_M, block_N), "float")  # float32 Gradient
# Conversion after calculation
result = C_local.astype("float16")
```

## 5. Performance Checklist

### Memory Access
- [ ] Optimizing memory access mode (merger access, vector)
- [ ] Rational use of shared memory cache data
- [ ] Enable swizzle to optimize the L2 cache

### Parallel Configuration
- [ ] Whether the size of the fraction is reasonable (2 times)
- [ ] Suitability of threads (128/256)
- [ ] Appropriate depth of software pipeline (2-4)

### Calculator Optimization
- [ ] Rational use of software pipeline `T.Pipelined`
- [ ] Calculate using mixed accuracy
- [ ] Use the inline language (T.gemm, T.reduce_*, etc.)

### Security
- [ ] **Check the use security**: ensure that all threads are synchronized
- [ ] **Avoiding Synchronization in Conditional Branch**: Preventing Death Locks
- [ ] **Use internal attribute function**: Avoiding the death of a linear card as a result of manual surrender
- [ ] **Correct use of thread index**: use T. Parallel instead of T.get_thread_binding

### Numerical stability
- [ ] Reduce Operation Is There Spill Responsive
- [ ] Whether to use float32 for intermediate accumulation
- [ ] Whether border situations, such as zero, negative number openings, have been addressed

## 6. Common performance traps

1. **Overdivided**: Too small file leads to inefficient memory access
2. **pipeline Depth inappropriate**: too deep or too shallow pipeline influence performance
3. **Memory Bank Conflict**: shared memory Access Malformed Mode
4. **Typologies conversion costs**: Frequent type conversions affect performance
5. **Sync cost**: unnecessary thread sync
6. **Sync Deadlock**: Synchronization in Conditional Branch resulted in the death of a thread card
7. **Line Index Error**: Using Wrong Thread Index
8. **shared memory distribution error**: allocation of shared memory in conditional branch

## Summary of best practice

1. **Correct performance before performance**: Ensure kernel correctness before optimizing performance
2. **Memory priority**: Prioritize memory access mode
3. **Synthetic security**: strict adherence to the synchronous use code, avoiding death locks
4. **Inline language**: priority is given to T.gemm, T.reduce_*, etc.
5. **Mixed accuracy**: Input with low accuracy, cumulative with high accuracy
6. **pipeline**: Hide memory latency by T. Pipelined
