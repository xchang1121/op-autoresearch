---
name: tilelang-cuda-memory
description: "TileLang CUDA memory access optimization policy, including T.alloc_shared/fragment use, data layout optimization, combined access and Bank Conflict avoidance techniques. TileLang kernel performance optimization scenarios for memory bandwidth limited and need to optimize data removal efficiency"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
---

# TileLang CUDA memory access optimization

Memory access is a key bottleneck for GPU performance. This document provides a memory access optimization policy for TileLang CUDA.

---

## 1. GPU Memory Level

### Memory bandwidth and latency

| Memory Type | bandwidth (A100) | latency | Capacity | TileLang API |
|---------|-------------|------|------|-------------|
| Organisation | ~19 TB/s | 1 cycle | 256 KB/SM | `T.alloc_fragment` / `T.alloc_local` |
| shared memory | ~19 TB/s | ~20 cycles | 164 KB/SM | `T.alloc_shared` |
| L2 Cache | ~5 TB/s | ~100 cycles | 40 MB | `T.use_swizzle` Optimization |
| global memory (HBM) | ~2 TB/s | ~400 cycles | 40/80 GB | `T.Tensor` |

### Optimization principle
- **Reduced global memory access**: Using shared memory and depository cache data
- **Merge Access**: Efficient data transfer using `T.copy`
- **Increased L2 Cache Liferate**: Optimizing locality of data using `T.use_swizzle`

---

## 2. Memory distribution best practice

### shared memory (data frequently accessed)

```python
# shared memory for caches loaded from global memory
A_shared = T.alloc_shared((block_M, block_K), "float16")
B_shared = T.alloc_shared((block_K, block_N), "float16")

# Efficient data loading
T.copy(A[by * block_M, ko * block_K], A_shared)
```

**Applicable scene**:
- Input block in matrix multiplication
- Median data from multiple visits
- Need data shared between threads

### Repository Snippets (cumulators and temporary storage)

```python
# Storer used for cumulative and local calculations
C_local = T.alloc_fragment((block_M, block_N), "float")
T.clear(C_local)

# Aggregation Operations
T.gemm(A_shared, B_shared, C_local)
```

**Applicable scene**:
- matrix multiplication Composer
- Provisional outcome of the return
- Local calculations

### Local memory (lined private storage)

```python
# Local memory for line private variables
C_reg = T.alloc_local((1,), "float")
T.clear(C_reg)
```

**Applicable scene**:
- Accumulation of single threads
- Temporary variable for a local thread

---

## 3. Data transfer optimization

### Use T.copy for combined access

```python
# ✅ Recommendations: Automatically merge access with T.copy
T.copy(A[by * block_M, ko * block_K], A_shared)
T.copy(B[ko * block_K, bx * block_N], B_shared)

# ✅ results returned
T.copy(C_local, C[by * block_M, bx * block_N])
```

### Use T. Parallel for parallel data copying

```python
# Parallel data copying
for k, j in T.Parallel(block_K, block_N):
    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
```

### Load with vector

```python
# vector loading to increase bandwidth utilization
for k in T.vectorized(TILE_K):
    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
```

---

## 4. Software pipeline

### Basic use

```python
# Use T. Pipelined to achieve software pipeline
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    # Loading data and calculating automatic overlap
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

### Num_stages Selection Guide

| num_stages | Use shared memory | Performance | Apply scene |
|-----------|-------------|------|---------|
| 2 | At least. | Basis | When shared memory is nervous |
| 3 | Medium | Usually the best. | Default Recommended |
| 4 | More. | It's better when it's a big matrix. | When shared memory is sufficient |
| 5+ | A lot. | It could drop. | Test Validation Required |

---

## 5. L2 Cache Optimization

### Swizzle Scatter

Improves the locality of the L2 cache by `T.use_swizzle`, especially for the calculation of 2D for matrix multiplication etc.:

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # Enable swizzle to increase the L2 Cache Hit Rate
    T.use_swizzle(panel_size=10, enable=True)

    # ...calculating logic...
```

### Layout Comment

```python
from tilelang.intrinsics import make_mma_swizzle_layout

# Use layout note to optimize shared memory access mode
T.annotate_layout({
    A_shared: make_mma_swizzle_layout(A_shared),
    B_shared: make_mma_swizzle_layout(B_shared),
})
```

---

## 6. Block Size Selection

### Recommended Settings

| operator Type | Block Size | Threads | Annotations |
|---------|-----------|-------|------|
| Element-wise | block=256~1024 | 128-256 | One-dimensional parallel. |
| GEMM | M=128, N=128, K=32 | 128 | Two-dimensional segment |
| GEMV | N=64~256, K=32~128 | 128 | 1-D + Serial |
| Reduce | block=256~512 | 128-256 | Reunification dimension segment |
| LayerNorm | block=256 | 256 | Deal by Line |

### Selection principle
- **Balancing parallelity and resource consumption**: avoid being too large or too small
- **Use 2 indents**: to facilitate hardware optimization
- **Consider shared memory limits**: Each SM normal 164 KB
- **Consider the memory pressure**: too large a fragment could cause spills

---

## 7. Full Optimization Example

### Optimized matrix multiplication

```python
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

@tilelang.jit(out_idx=[-1])
def optimized_matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 1. Distribution of memory
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")

            # 2. Layout optimization
            T.annotate_layout({
                A_shared: make_mma_swizzle_layout(A_shared),
                B_shared: make_mma_swizzle_layout(B_shared),
            })

            # 3. L2 Cache optimization
            T.use_swizzle(panel_size=10, enable=True)

            T.clear(C_local)

            # 4. Software pipeline
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # 5. Return of results
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

---

## 8. Summary of best practice

### Memory Allocation
1. **shared memory**: Data block for frequent access
2. **Repositor clip**: used for compressors and temporary calculations
3. **Local memory**: for linear private variables

### Data Transfer
1. **Use T.copy**: Automatically merge memory access
2. **Using T. Pipelined**: Overlap data loading and calculation
3. **Loaded with T.vectorize**: vector

### Cache Optimization
1. **T.use_swizzle**: Optimizing L2 Cache Locality
2. **T. annotate_layout**: Optimizing shared memory access mode
3. **Rational segment**: balancing parallelity and cache utilization

### A trap to avoid.
- Oversized shared memory distribution caused occupancy to drop
- Ignore software pipeline optimization
- Inefficient memory access due to inappropriate block size setting
- Forget L2 Cache Optimization
