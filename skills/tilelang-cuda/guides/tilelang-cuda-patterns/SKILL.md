---
name: tilelang-cuda-patterns
description: "TileLang CUDA core programming mode (element by element, attribution, matrix multiplication, GEMV) standard realization paradigm and code template. This applies to internal code generation scenarios that need to quickly determine which type of operator is programmed or that need to understand the structure of the TileLang model basic code"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
  operator_patterns: "elementwise, reduce, matmul, gemv"
---

# TileLang CUDA programming mode

## 1. Element-by-Element Operating Mode

For element-level operations: Adding, Multiplication, Activation Functions, etc.

### Standard code structure

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def elementwise_op(M, N, block_M, block_N, threads):
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"),
             B: T.Tensor((M, N), "float32"),
             C: T.Tensor((M, N), "float32")):

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            for (local_y, local_x) in T.Parallel(block_M, block_N):
                y = by * block_M + local_y
                x = bx * block_N + local_x
                C[y, x] = A[y, x] + B[y, x]

    return main
```

### Apply operator
- Algorithmic Operations: add, Mul, sub, div
- Activate function: relu, sigmoid, tanh, gelu
- Mathematical functions: ext, log, sqrt, pow
- Type conversion:cast
- Broadcast operation: Broadcast

### Key points
- Map to range using `T.Parallel`
- Calculate global index from block index
- Direct access to global memory, without shared memory
- Appropriate treatment of border conditions

### Conditional Element-by-Element Operations

```python
@tilelang.jit(out_idx=[-1])
def conditional_elementwise(N, threads):
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"),
             B: T.Tensor((N,), "float32"),
             C: T.Tensor((N,), "float32")):

        with T.Kernel(T.ceildiv(N, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < N:
                    C[idx] = T.if_then_else(
                        A[idx] > 0,
                        A[idx] + B[idx],
                        A[idx] - B[idx]
                    )

    return main
```

## 2. Reunification Mode

applies to aggregation, max, harmony, etc.

### Standard code structure

```python
@tilelang.jit(out_idx=[-1])
def reduction_op(M, N, block_size):
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"),
             C: T.Tensor((M,), "float32")):

        with T.Kernel(M, threads=block_size) as bx:
            # Distribute Memory Snippets
            A_local = T.alloc_fragment((N,), "float32")
            C_local = T.alloc_fragment((1,), "float32")

            # Loading data
            T.copy(A[bx, 0:N], A_local)

            # ✅ uses the built-in attribute function
            T.reduce_sum(A_local, C_local, dim=0)

            # Write back the results.
            C[bx] = C_local[0]

    return main
```

### Apply operator
- Som, mean, max, min, prod
- Normalize: softmax, playnorm, watchnorm, rmsnorm
- Statistics: varance, std
- Weighted sum

### Key points
- **✅ must use the built-in attribute function**: `T.reduce_sum`, `T.reduce_max`, `T.reduce_min`, `T.reduce_mean`
- **❌ Ban on manual return**: manual return requires `T.sync_threads()`, which in the sub-division of conditions can easily lead to death locks
- Use memory clips `T.alloc_fragment` as temporary storage
- Note numerical stability (e.g. maximum value subtracted from softmax)

### "Layer Norm," for example

```python
@tilelang.jit(out_idx=[-1])
def layer_norm(M, N, block_size):
    @T.prim_func
    def main(x: T.Tensor((M, N), "float32"),
             y: T.Tensor((M, N), "float32")):

        with T.Kernel(M, threads=block_size) as bx:
            A_local = T.alloc_fragment((N,), "float32")
            A_pow_local = T.alloc_fragment((N,), "float32")
            A_sum = T.alloc_fragment((1,), "float32")
            A_powsum = T.alloc_fragment((1,), "float32")

            # Loading data
            for tid in T.Parallel(N):
                A_local[tid] = x[bx, tid]
                A_pow_local[tid] = x[bx, tid] * x[bx, tid]

            # ✅ uses the built-in contract
            T.reduce_sum(A_local, A_sum, dim=0)
            T.reduce_sum(A_pow_local, A_powsum, dim=0)

            # Calculate average and variance
            for tid in T.Parallel(N):
                mean_val = A_sum[0] / N
                var_val = A_powsum[0] / N - mean_val * mean_val
                y[bx, tid] = (A_local[tid] - mean_val) / T.sqrt(var_val + 1e-5)

    return main
```

## 3. matrix multiplication mode

For multi-dimensional block calculations such as matrix multiplication.

### Standard code structure

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Distribution of shared memory
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")

            T.clear(C_local)

            # K Dimension Cycle (Software pipeline)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # Turn it back on.
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

### Apply operator
- Matrix operation: matmul, bmm (batch matmul), linear
- Volume: conv2d, conv3d
- Attention mechanisms: attention (Q*K^T, scores*V)

### Key points
- **2D Grid**: using `T.Kernel(grid_x, grid_y)` 2D parallel
- **shared memory**: Use `T.alloc_shared` Cache Data Block
- **K dimension cycle**: pipeline software using `T.Pipelined`
- **Inlined GEMM**: use `T.gemm` and use Tensor Core to accelerate
- **Mixed accuracy**: input for float16, loader for float32

> 💡 Full GEMM Optimization Guide (Swizzling, Autotuning, Persistent, Split-K, Stream-K, FP8, Int4, Fine-grained MMA, etc.) is available at [tilelang-cuda-gemm] (../tilelang-cuda-gemm/SKILL.md) Skill.

## 4. GEMV Mode

This applies to irregular access modes such as the matrix vector multiplication.

### Standard code structure

```python
@tilelang.jit(out_idx=[-1])
def gemv(N, K, BLOCK_N, BLOCK_K):
    @T.prim_func
    def main(A: T.Tensor((K,), "float16"),
             B: T.Tensor((N, K), "float16"),
             C: T.Tensor((N,), "float16")):

        with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
            A_shared = T.alloc_shared((BLOCK_K,), "float16")
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), "float16")

            for tn in T.Parallel(BLOCK_N):
                C_reg = T.alloc_local((1,), "float")
                T.clear(C_reg)

                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for tk in T.serial(BLOCK_K):
                        A_shared[tk] = A[bk * BLOCK_K + tk]
                        B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]

                    for tk in T.serial(BLOCK_K):
                        C_reg[0] += A_shared[tk].astype("float") * B_shared[tn, tk].astype("float")

                C[bn * BLOCK_N + tn] = C_reg[0]

    return main
```

### Key points
- **Thread Index**: Get Thread Index with `T.Parallel`
- **Serial cycle**: use `T.serial` for K-DVM
- **Local repository**: Private cumulative value using `T.alloc_local` storage lines
- **Type conversion**: accuracy conversion `.astype("float")` during calculation

## Mode Selection Guide

| operator Type | Recommended Mode | Key features | Memory Usage |
|---------|---------|---------|---------|
| Element-wise | Element by Elements Operation | Element-by-Element calculation | global memory |
| Reduction | Reunification Mode | Multiple values need to be aggregated | Can not open message |
| MatMul/Conv | matrix multiplication mode | Multi-dimensional block calculations, 2D Grid | shared memory+ repository |
| GEMV | GEMV Mode | vector-matrix multiplication | shared memory + Local memory |
| Attention | Convention +matrix multiplication | Group Mode | shared memory+ repository |

## best practice

1. **Select the appropriate mode**: Select the base mode based on operator characteristics
2. **Optimized segment size**: balancing parallelity and resource occupancy
3. **Inline language**: preferred `T.gemm`, `T.reduce_*`, etc.
4. **Note numerical stability**: special attention for reduce class operator
5. **Memorial access optimization**: combined access using `T.copy`
6. **Software pipeline**: Hide Memory latency with `T.Pipelined`
7. **Avoiding Manual Synchronization**: replace manual surrender with an built-in assign function
