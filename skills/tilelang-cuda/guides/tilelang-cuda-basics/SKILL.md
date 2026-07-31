---
name: tilelang-cuda-basics
description: "TileLang CUDA Core Concept, kernel structure and standard programming model"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
  operator_patterns: "all"
---

# TileLang CUDA programming base

## 1. Core concepts

### TileLang Profile
- **Definition: TileLang is an area-specific language (DSL) designed for high performance GPU/CPU nuclear development using a syntax similar to Python based on TVM compiler
- **Characteristics**: Focus on productivity at the expense of bottom-up optimization, providing three layers of abstraction

### Programming interface level
- **Level 1 (non-hardware related)**: compiler automated processing memory level and hardware specific optimization appropriate for fast prototype development
- **Level 2 (hardware perception + Tile library)**: Provide predefined Tile library operations and models for most high-performance computing applications
- **Level 3 (hardware perception + thread source)**: Provide direct access to linear originals and low-level structures, suitable for extreme performance optimization

### Kernel
- **Definition**: using `@tilelang.jit` decorative functions, compiled and executed in parallel on GPU
- **Structure**: Main function containing `@T.prim_func` decorations internally, defining parallel implementation logic through `T.Kernel` context manager

### Grid (Grid) and Thread
- **Grid**: Parallel dimensions configuration at kernel startup, calculation of blocks using `T.ceildiv`
- **Line block**: Each block contains a specified number of threads, set by `threads` parameters
- **block index**: `T.Kernel` context returns `(bx, by)` corresponding to `blockIdx.x, blockIdx.y`

### Memory Level
- **global memory (Global Memoory)**: GPU Main Memory (HBM), all threads accessible
- **shared memory (Shared Memoory)**: SM Internal Sharing, Distribution through `T.alloc_shared`
- **Fragment**: Corresponds to GPU repository file, distributed via `T.alloc_fragment`
- **Local memory (Local)**: thread local storage, distributed via `T.alloc_local`

## 2. Standard kernel structure

TileLang kernel standard structure model:

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def my_kernel(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):

        # 1. Definition of kernel context (grid and thread configuration)
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 2. Distribution of memory
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")

            # 3. Initialization
            T.clear(C_local)

            # 4. Calculation logic (including data loading and calculation)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # 5. Return of results
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

## 3. kernel call engagement (out_idx)

The `@tilelang.jit`/ `tilelang.compile` of TileLang specifies by `out_idx` which tensor belongs to the output.

### Basic use

```python
@tilelang.jit(out_idx=[1])
def parallel_elementwise_static(length=256):
    @T.prim_func
    def main(A: T.Tensor((length,), "float32"),
             B: T.Tensor((length,), "float32")):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length):
                B[i] = A[i] + 1.0
    return main

kernel = parallel_elementwise_static()
result = kernel(data)  # ✅ Pass input only data;TileLang Based on out_idx Return Output
```

### Out_idx Rules

- `out_idx=[-1]`: Last tensor is the output
- `out_idx=[1]`: The second tensor is output
- Support multiple output: `out1, out2 = kernel(x, y)`

### ⚠ ️'s common error

```python
# ❌ error: Extra transfer of tensor
y = torch.empty_like(x)
kernel(x, y)  # ValueError: Expected 2 inputs, got 3 with 2 inputs and 1 outputs

# ✅ Correct: Only pass input, out_idx automatically create output
result = kernel(x)
```

**Practical recommendations**
1. **Recommended method**: `out_idx` is retained and only entered when called
2. **Manually manage output**: without setting `out_idx`, the output is also declared as a parameter in `prim_func`, ensuring that "as many parameters as defined" is passed.

## 4. Basic programming mode

### 4.1 Element by Element Operations

```python
@tilelang.jit(out_idx=[-1])
def elementwise_add(M, N, block_M, block_N, threads):
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

**Key concepts**
- **Parallel map**: processing of one or more elements per thread
- **Indexing**: global indexing from linear block indexing
- **Memory access**: direct access to global memory

### 4.2 matrix multiplication (GEMM)

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

**Key concepts**
- **shared memory cache**: data frequently accessed using `T.alloc_shared` cache
- **Software pipeline**: `T.Pipelined` overlapping memory loading and calculation
- **Interim matrix multiplication**: `T.gemm` with Tensor Core

### 4.3 Element-by-Element Operations (Shared Memoory + Fragment mode)

```python
@tilelang.jit(out_idx=[-1])
def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):
    @T.prim_func
    def elem_add(A: T.Tensor((M, N), in_dtype),
                 B: T.Tensor((M, N), in_dtype),
                 C: T.Tensor((M, N), out_dtype)):

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            for local_y, local_x in T.Parallel(block_M, block_N):
                C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return elem_add
```

**Key concepts**
- **Standard data stream**: GM → Shared → Fragment → calculates → Fragment → Shared → GM
- **ReLU Mode**: `T.max(x, 0)` in fragment, not `T.relu`
- **tilelang does not have `T.tile.relu`**, CUDA backend uses `T.max(value, 0)` to express ReLU

### 4.4 Dynamic Shape

```python
@tilelang.jit(out_idx=[-1])
def relu_dynamic(block_M, block_N):
    M = T.dynamic("m")
    N = T.dynamic("n")

    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"),
             Y: T.Tensor((M, N), "float32")):

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), "float32")
            Y_local = T.alloc_fragment((block_M, block_N), "float32")
            Y_shared = T.alloc_shared((block_M, block_N), "float32")

            T.copy(X[by * block_M, bx * block_N], X_shared)
            T.copy(X_shared, Y_local)
            for i, j in T.Parallel(block_M, block_N):
                Y_local[i, j] = T.max(Y_local[i, j], 0)
            T.copy(Y_local, Y_shared)
            T.copy(Y_shared, Y[by * block_M, bx * block_N])

    return main

# As approved in the same translation for different uses
# kernel = relu_dynamic(128, 256)
# y1 = kernel(torch.randn(1024, 2048, device="cuda"))
# y2 = kernel(torch.randn(512, 128, device="cuda"))
```

### 4.5 Matrix vector Multiplication (GEMV)

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

            # ✅ Correct: Use T. Parallel to get thread index
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

**Key concepts**
- **Thread Index**: Acquisition of Thread Index with `T.Parallel()` (recommended)
- **Serial cycle**: `T.serial()` for operations that require serial execution
- **Type conversion**: accuracy conversion with `.astype()` calculation
- **⚠ ️**: Avoid `T.get_thread_binding()`, recommend `T.Parallel()`

## 5. Advanced Programming Mode

### 5.1 Macro definition

```python
@T.macro
def Softmax(acc_s, acc_s_cast, scores_max, scores_sum, logsum):
    T.copy(scores_max, scores_max_prev)
    T.fill(scores_max, -T.infinity("float"))
    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

    for i in T.Parallel(block_M):
        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

    T.reduce_sum(acc_s, scores_sum, dim=1)
    for i in T.Parallel(block_M):
        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    T.copy(acc_s, acc_s_cast)
```

### 5.2 Conditional implementation and border management

```python
@tilelang.jit(out_idx=[-1])
def conditional_kernel(N, threads):
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

### 5.3 Atomic Operations and Conscription

```python
@tilelang.jit(out_idx=[-1])
def atomic_reduction(N, K, BLOCK_N, reduce_threads):
    @T.prim_func
    def main(A: T.Tensor((N, K), "float32"),
             C: T.Tensor((N,), "float32")):

        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            C_shared = T.alloc_shared((BLOCK_N,), "float32")
            C_accum = T.alloc_local((1,), "float32")

            T.clear(C_accum)

            for tn in T.Parallel(BLOCK_N):
                for k in T.serial(K):
                    C_accum[0] += A[bn * BLOCK_N + tn, k]

                T.atomic_add(C_shared[tn], C_accum[0])
                C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

## 6. Summary of best practice

### Programming Mode Selection
- **Simple operation**: use of element-by-component mode of operation
- **Matrix operation**: use GEMM mode, use `T.gemm` built-in language
- **Irregular access**: use GEMV mode
- **Complex calculation**: use macro definition `@T.macro` organizational code

### Elements of performance optimization
1. **Rational selection of fraction size**: balance memory use and computational efficiency
2. **Software pipeline**: `T.Pipelined` overlapping memory operations and calculations
3. **Parallel data movement**: optimize memory access using `T.Parallel`
4. **Select the appropriate thread**: usually 128 or 256
5. **Use of built-in original language**: Optimization of original language using `T.gemm`, `T.reduce_sum`, etc.

### Common error avoidance
1. **Memory distribution too large**: beyond hardware limitations
2. **Index calculation error**: causing memory access to cross-border
3. **data type's mismatch**: accuracy's loss or decline in performance
4. **pipeline Depth inappropriate**: influence performance
5. **⚠ ️ Synchronise Use Error**: Synchronization in conditional branches leads to dead locks
6. **⚠ ️ Thread Index Retrieving Error**: Using `T.get_thread_binding()` instead of `T.Parallel()`
7. **⚠ ️ out_idx used error**: extra transfer of tensor leads to a mismatch of parameters
8. **⚠️ ReLUWriting Error**:CUDAUse`T.max(x, 0)`No, it's not.`T.tile.relu`  That's  Ascend backend)

## 7. Compile and Profiling

### 7.1 Compiled API

```python
# Mode I: @tilelang.jit Decorator (recommended)
@tilelang.jit(out_idx=[-1], target="cuda")
def my_kernel(M, N, ...):
    @T.prim_func
    def main(...): ...
    return main

kernel = my_kernel(1024, 1024, ...)
output = kernel(input_tensor)

# Mode 2: tilelang.compile function
func = my_kernel(1024, 1024, ...)
kernel = tilelang.compile(func, out_idx=[-1], target="cuda")
output = kernel(input_tensor)
```

- **target**: `"cuda"` | `"cuda -arch=sm_80"` | `"cuda -arch=sm_90"` | `"hip"` | `"cpu"` | `"auto"`
- **out_idx**: Specified index for output tensor, `-1` for last argument
- **When target is not specified**: Auto deduce from device input tensor

### 7.2 Profiling / Benchmark

```python
from tilelang.profiler import do_bench

kernel = my_kernel(1024, 1024, ...)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# Basic use
latency = do_bench(lambda: kernel(x), backend="event")

# Specify warmup/repeat time and return mode
latency = do_bench(
    lambda: kernel(x),
    warmup=25,       # warmup Target time(ms)
    rep=100,         # Target time for evaluation(ms)
    backend="event", # "event" | "cupti" | "cudagraph"
    return_mode="min" # "mean" | "median" | "min" | "max"
)
```

- **do_bench Automanage**: L2 carche flush (256MB), warmup iterative algebra calculation, CUDA Event high accuracy time
- **backend= "event"**: Default, time with CUDA Event
- **backend = "cupti"**: more precise but required CUPTI
- **backend = "cudagraph"**: Minimise with CUDA graph play
