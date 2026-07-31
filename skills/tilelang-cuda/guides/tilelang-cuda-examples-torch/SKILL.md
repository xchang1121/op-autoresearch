---
name: tilelang-cuda-examples-torch
description: "PyTorch + TileLang CUDA full example code"
category: example
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
  framework: torch
  examples: "matmul, elementwise, layernorm, gemv, flash_attention"
---

# PyTorch + TileLang CUDA Example Code

This Skill contains a full runable example code showing how to use TileLang CUDA in PyTorch to write high performance kernel.

## Example List

### 1. matrix multiplication (GEMM)
**operator type**: MatMul
**Key points**:
- shared memory Cache Input Block
- `T.gemm` with Tensor Core
- Software pipeline `T.Pipelined`
- Mixed accuracy (float32 presser)

```python
import torch
import tilelang
import tilelang.language as T

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

# Call Method
def matmul_call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    block_M, block_N, block_K = 128, 128, 32
    kernel = matmul(M, N, K, block_M, block_N, block_K)
    C = kernel(A, B)  # out_idx=[-1], only input
    return C
```

### 2. matrix multiplication (float32, manual managing output)
**operator type**: MatMul
**Key points**:
- Do not use `out_idx`, manage output manually
- Float32 data type
- Create output tensor manually and enter it together

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def square_matrix_multiply(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
            A: T.Tensor((M, K), "float32"),
            B: T.Tensor((K, N), "float32"),
            C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float")

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

def square_matrix_multiply_call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N = A.size(0)
    block_M, block_N, block_K = 128, 128, 32
    # Create output tensor manually when out_idx is not used
    C = torch.empty_like(A)
    kernel = square_matrix_multiply(N, N, N, block_M, block_N, block_K)
    kernel(A, B, C)  # Enter all parameters including output
    return C
```

### 3. Element-wide operation (Element-wise Add, Global Memoory)
**operator type**: Element-wise
**Key points**:
- Simple list of kernels for TileLang
- Use `T.Parallel` for parallel calculations
- Direct access to global memory

```python
import torch
import tilelang
import tilelang.language as T

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

# Call Method
def add_call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    block_M, block_N = 32, 32
    threads = 256
    kernel = elementwise_add(M, N, block_M, block_N, threads)
    return kernel(A, B)
```

### 3b. Element-by-Element Operations (Shared Memoory + Fragment mode)
**operator type**: Element-wise
**Key points**:
- Standard data stream: GM → Shared → Fragment → Calculated → Fragment → Shared → GM
- Accelerating calculations using fragment (depositor)
- `T.copy` Automatically merge memory access

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def elementwise_add_shared(M, N, block_M, block_N, in_dtype, out_dtype, threads):
    @T.prim_func
    def main(A: T.Tensor((M, N), in_dtype),
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

    return main

# Call Method
def add_call_shared(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    kernel = elementwise_add_shared(M, N, 32, 32, T.float32, T.float32, 128)
    return kernel(A, B)
```

### 3c. ReLU (Shared Memoory + Fragment)
**operator type**: Element-wise
**Key points**:
- CUDA uses `T.max(x, 0)` to express ReLU (**not**`T.tile.relu`, that's Ascend backend)
- Standard data stream: GM → Shared → Fragment → `T.max` → Fragment → Shared → GM

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def relu_kernel(M, N, block_M, block_N, dtype):
    @T.prim_func
    def main(X: T.Tensor((M, N), dtype),
             Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), dtype)
            Y_local = T.alloc_fragment((block_M, block_N), dtype)
            Y_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(X[by * block_M, bx * block_N], X_shared)
            T.copy(X_shared, Y_local)
            for i, j in T.Parallel(block_M, block_N):
                Y_local[i, j] = T.max(Y_local[i, j], 0)
            T.copy(Y_local, Y_shared)
            T.copy(Y_shared, Y[by * block_M, bx * block_N])

    return main

# Call Method
def relu_call(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    kernel = relu_kernel(M, N, 128, 256, x.dtype)
    return kernel(x)
```

### 4. Layer Norm
**operator type**: Reduce + Element-wise
**Key points**:
- Use `out_idx=[-1]` to specify output
- `T.reduce_sum` built-in contract (avoiding manual synchronization)
- Border check to process non-matched data
- Float32 Intermediate calculation guaranteed accuracy

```python
import tilelang as tl
import tilelang.language as T
import torch

@tl.jit(out_idx=[-1])
def layer_norm_kernel(batch_size, features, dim1, dim2, block_size):
    @T.prim_func
    def main(x: T.Tensor((batch_size, features, dim1, dim2), "float16"),
             y: T.Tensor((batch_size, features, dim1, dim2), "float16")):

        total_size = features * dim1 * dim2

        with T.Kernel(batch_size, T.ceildiv(total_size, block_size), threads=block_size) as (sample_idx, bx):
            A_shared = T.alloc_shared((block_size,), "float32")
            A_pow_local = T.alloc_fragment((block_size,), "float32")
            A_powsum = T.alloc_fragment((1,), "float32")

            # Loading and Calculating Data
            for tid in T.Parallel(block_size):
                elem_idx = bx * block_size + tid

                if elem_idx < total_size:
                    c = elem_idx // (dim1 * dim2)
                    h = (elem_idx % (dim1 * dim2)) // dim2
                    w = elem_idx % dim2
                    input_val = x[sample_idx, c, h, w].astype("float32")

                    A_shared[tid] = input_val
                    A_pow_local[tid] = input_val * input_val
                else:
                    A_shared[tid] = 0.0
                    A_pow_local[tid] = 0.0

            # ✅ uses the built-in contract to avoid sync and thread-calcination
            T.reduce_sum(A_pow_local, A_powsum, dim=0)

            # Compute and apply the normalized factor
            for tid in T.Parallel(block_size):
                elem_idx = bx * block_size + tid

                if elem_idx < total_size:
                    c = elem_idx // (dim1 * dim2)
                    h = (elem_idx % (dim1 * dim2)) // dim2
                    w = elem_idx % dim2
                    input_val = x[sample_idx, c, h, w].astype("float32")

                    mean_val = A_powsum[0] / total_size
                    var_val = A_powsum[0] / total_size - mean_val * mean_val
                    normalized = (input_val - mean_val) / T.sqrt(var_val + 1e-5)
                    y[sample_idx, c, h, w] = normalized.astype("float16")

    return main

# Call Method
def layer_norm(input_tensor: torch.Tensor):
    batch_size, features, dim1, dim2 = input_tensor.shape
    block_size = 256
    kernel = layer_norm_kernel(batch_size, features, dim1, dim2, block_size)
    y = kernel(input_tensor)  # out_idx=[-1], create output automatically
    return y
```

### 5. GEMV (matrix vector multiplier)
**operator type**: GEMV
**Key points**:
- Get thread index with `T.Parallel`
- Use `T.serial` for serial cycle
- Use `T.alloc_local` as a private cumulation of threads
- Type conversion `.astype("float")` guaranteed accuracy

```python
import torch
import tilelang
import tilelang.language as T

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

# Call Method
def gemv_call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N, K = B.shape
    BLOCK_N, BLOCK_K = 128, 32
    kernel = gemv(N, K, BLOCK_N, BLOCK_K)
    return kernel(A, B)
```

## Universal Mode

All TileLang CUDA examples follow the same structure:

### Definition of kernel
```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def kernel_name(shape_params, block_params):
    @T.prim_func
    def main(input1: T.Tensor(shape, dtype),
             input2: T.Tensor(shape, dtype),
             output: T.Tensor(shape, dtype)):

        with T.Kernel(grid_x, grid_y, threads=N) as (bx, by):
            # 1. Distribution of memory
            shared = T.alloc_shared(shape, dtype)
            local = T.alloc_fragment(shape, dtype)

            # 2. Data loading and calculation
            T.copy(input[...], shared)
            # ...calculating logic...

            # 3. Reverting the results
            T.copy(local, output[...])

    return main
```

### Call Functions
```python
def call_function(input_tensor: torch.Tensor) -> torch.Tensor:
    # Determine shape and split parameters
    M, N = input_tensor.shape
    block_M, block_N = 128, 128

    # Compile kernel
    kernel = kernel_name(M, N, block_M, block_N)

    # When out_idx is used: transfer input only
    result = kernel(input_tensor)

    # When out_idx is not used: create output manually
    # output = torch.empty_like(input_tensor)
    # kernel(input_tensor, output)

    return result
```

## Key note

### 1. Out_idx Usage Code
```python
# ✅ uses out_idx: pass input only, output automatically created
@tilelang.jit(out_idx=[-1])
result = kernel(input_data)

# ✅ does not use out_idx: manually manage all tensor
@tilelang.jit
output = torch.empty_like(input_data)
kernel(input_data, output)

# ❌ error: use out_idx but extra transfer
@tilelang.jit(out_idx=[-1])
output = torch.empty_like(input_data)
kernel(input_data, output)  # ValueError!
```

### 2. tensor device and data type.
```python
# Make sure it's entered on CUDA device.
input_tensor = input_tensor.cuda()

# Type conversion within kernel completed
input_val = x[i].astype("float32")  # float16 -> float32
result = normalized.astype("float16")  # float32 -> float16
```

### 3. Internal Reunification Instead of Manual Sync
```python
# ✅ Correct: Use internal union
T.reduce_sum(input_local, output_local, dim=0)

# ❌ error: manual return (caused death lock)
# while stride > 0:
#     if tid < stride:
#         shared[tid] += shared[tid + stride]
#     T.sync_threads()
#     stride //= 2
```

## Validate correctness

```python
# Compare to PyTorch Native
x = torch.randn(128, 256, device='cuda', dtype=torch.float16)
output_tilelang = kernel_call(x)
output_torch = torch_reference(x)

# Check discrepancies
diff = (output_tilelang - output_torch).abs().max()
print(f"Max difference: {diff.item()}")
assert diff < 1e-3, "Results mismatch!"
```

## Profiling / Benchmark

Use `do_bench`, which is embedded in tilelang, for performance evaluation:

```python
from tilelang.profiler import do_bench

# Compile kernel
kernel = matmul(M, N, K, block_M=128, block_N=128, block_K=32)
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

# Basic profiling
latency_ms = do_bench(lambda: kernel(a, b))

# Full profling (specify mode)
latency = do_bench(
    lambda: kernel(a, b),
    warmup=25,        # warmup Target time(ms)
    rep=100,          # Target time for evaluation(ms)
    backend="event",  # "event" | "cupti" | "cudagraph"
    return_mode="min" # "mean" | "median" | "min" | "max"
)
print(f"Latency: {latency:.3f} ms")

# Use Profiller class (inline correctness verification + protection)
profiler = kernel.get_profiler()
profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
latency = profiler.do_bench(backend="event")
```

- **do_bench Automanaging**: L2 carche flush (256MB), warmup/rep iterative algebra calculation, CUDA Event high accuracy timing
- **backend= "event"**: default, CUDA Event time
- **backend = "cupti"**: CUPTI programmer, more precise
- **backend = "cudagraph"**: CUDA graph play, minimise launch overhead
