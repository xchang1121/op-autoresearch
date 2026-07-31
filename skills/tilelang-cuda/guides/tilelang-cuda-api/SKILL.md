---
name: tilelang-cuda-api
description: "TileLang CUDA API complete reference manual for any TileLang CUDA internal nuclear code generation scenario that requires access to specific API usages and the meaning of function parameters"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
---

# TileLang CUDA API reference manual

This document provides a detailed reference to the TileLang core API, including a function signature, a parameter description and examples of how to use it.

## 1. kernel definition and compilation

### @tilelang.jit(out_idx)
```python
@tilelang.jit(out_idx=[-1])
def my_kernel(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        # kernel realization
        pass
    return main
```
- **Fact**: Compile the TileLang function into the GPU kernel
- **Parameter**: `out_idx` - Specifies the index list for the output tensor (e. g. `[-1]` means the last parameter is the output)
- **Call**: After setting `out_idx`, only enter tensor for running kernel, with output created automatically by TileLang

### tilelang.compile
```python
kernel = tilelang.compile(my_func, out_idx=[-1])
```
- **Fact**: compile TileLang function (equivalent to `@tilelang.jit`)

## 2. kernel context

### T.Kernel(grid_x, grid_y, threads)
```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # bx, by blockIdx.x, blockIdx.y.
    pass
```
- **Parameters**:
  - `grid_x`: Grid X dimension size
  - `grid_y`: Grid Y dimension size (optional)
  - `threads`: Number of threads per thread
- **Return**: Thread block index `(bx, by)`

### T.ceildiv(a, b)
```python
grid_size = T.ceildiv(N, block_N)
```
- **Parameters**: `a`, `b` - divided and divided
- **Return**: remove the division result upwards
- **Use**: Calculating Grid Size

## 3. Memory Allocation API

### T.alloc_shared(shape, dtype)
```python
A_shared = T.alloc_shared((block_M, block_K), "float16")
```
- **Role**:Allocationshared memoryCounterpartGPU shared memory)
- **Parameters**:
  - `shape`: tensorZ1XQ
  - `dtype`: data type
- **Pilot**: cache of frequently accessed data

### T.alloc_fragment(shape, dtype)
```python
C_local = T.alloc_fragment((block_M, block_N), "float")
```
- **Activation**: Distribution of repository clips (relative to GPU repository files)
- **Parameters**:
  - `shape`: tensorZ1XQ
  - `dtype`: data type
- **Use**: Thrusters and temporary storage

### T.alloc_local(shape, dtype)
```python
temp = T.alloc_local((1,), "float32")
```
- **Activation**: Distribution thread local memory
- **Parameters**:
  - `shape`: tensorZ1XQ
  - `dtype`: data type
- **Pilot variable for the private segment of the thread**

## 4. Data Operation API

### T.copy(src, dst)
```python
# global memory to shared memory
T.copy(A[by * block_M, ko * block_K], A_shared)

# Storage to global memory
T.copy(C_local, C[by * block_M, bx * block_N])
```
- **Activation**: efficient memory copying, auto-merger access
- **Parameters**:
  - `src`: Source data (which can be global memory slices or memory clips)
  - `dst`: Target data

### T.clear(tensor)
```python
T.clear(C_local)
```
- **Fact**: clear tensor
- **Parameters**: `tensor` - tensor to clear zero

### T.fill(tensor, value)
```python
T.fill(buffer, -T.infinity("float"))
```
- **Activation**: Fill tensor with specified values
- **Parameters**:
  - `tensor`: Target tensor
  - `value`: Filling value

## 5. Cycle Control API

### T.Pipelined(count, num_stages)
```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.gemm(A_shared, B_shared, C_local)
```
- **Activation**: software pipeline cycle, overlapping memory operations and calculations
- **Parameters**:
  - `count`: Number of cycles
  - `num_stages`: pipeline Depth (usually the best effect 2-4)

### T.Parallel(dim1, dim2, ...)
```python
# One-dimensional parallel
for i in T.Parallel(block_M):
    pass

# Multi-dimensional Parallel
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = A_shared[i, j] + B_shared[i, j]
```
- **Activation**: Parallel cycle, automatic mapping to linear range
- **Parameter**: dimensions
- **Strength**: Automatically process linear mapping to avoid manual linear indexing errors

### T.serial(count)
```python
for k in T.serial(block_K):
    pass
```
- **Activation**: serial cycle, sequential execution
- **Parameter**: `count` - Number of cycles

### T.vectorized(count)
```python
for k in T.vectorized(TILE_K):
    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
```
- **Activation**: vector cycle, using vector command
- **Parameter**: `count` - length of vectorization

## 6. Inline Calculating Original Language

### T.gemm(A, B, C, transpose_B, policy)
```python
# Base matrix multiplication
T.gemm(A_shared, B_shared, C_local)

# matrix multiplication with transferred
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

# Specify Warp Policy
T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
```
- **Activation**: matrix multiplication at Tile level, accelerated with Tensor Core
- **Parameters**:
  - `A`, `B`: Enter matrix (usually shared memory)
  - `C`: Output/cumulator (usually memory clips)
  - `transpose_B`: Whether to convert the B matrix
  - `policy`: Warp Policy

### T.reduce_max(input, output, dim, clear)
```python
T.reduce_max(acc_s, scores_max, dim=1)
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
```
- **Activation**: Maximum return
- **Parameters**:
  - `input`: Enter tensor
  - `output`: Output tensor
  - `dim`: Reunification dimension
  - `clear`: Whether to clear zero output first (default True)

### T.reduce_sum(input, output, dim)
```python
T.reduce_sum(acc_s, scores_sum, dim=1)
```
- **Activation**: peace of return
- **Parameters**: Same `T.reduce_max`

### T.reduce_min(input, output, dim)
```python
T.reduce_min(input_tensor, output_tensor, dim=axis)
```
- **Activation**: Minimum return

### T.reduce_mean(input, output, dim)
```python
T.reduce_mean(input_tensor, output_tensor, dim=axis)
```
- **Activation**: average return

## 7. Math Functions

```python
T.exp(x)          # Index Functions
T.exp2(x)         # Here. 2 Index as Bottom
T.log(x)          # Natural log [N]
T.sqrt(x)         # Square root
T.rsqrt(x)        # Square root countdown
T.infinity(dtype)  # Infinite constants
```

## 8. Conditional and Logical Operations

### T.if_then_else(condition, true_val, false_val)
```python
result = T.if_then_else(A[idx] > 0, A[idx] + B[idx], A[idx] - B[idx])
```
- **Activation**: Conditional selection (similar to a three-track operator)

### Conditional Branch
```python
for i in T.Parallel(block_M):
    if i < N:
        # Conditional Operations
        pass
```

## 9. Atomic Operations

### T.atomic_add(target, value)
```python
T.atomic_add(C_shared[tn], C_accum[0])
```
- **Activation**: linearly safe atoms added
- **Parameters**:
  - `target`: Target RAM position
  - `value`: Value to add

## 10. Thread Index Retrieving

### ✅ Recommendation: T. Parallel
```python
for tn in T.Parallel(BLOCK_N):
    # tn corresponds to radIdx.x
    pass
```
- **Strength**: Automatic processing of linear mapping, safer

### ⚠ ️ does not recommend: T. Get_thread_binding
```python
# Could cause problems, not recommended for use
tn = T.get_thread_binding(0)  # threadIdx.x
tk = T.get_thread_binding(1)  # threadIdx.y
```

## 11. Sync Operation

### T.sync_threads()
```python
T.sync_threads()  # All threads must perform this operation
```
- **Activate**: Sync within a thread block
- **⚠ ️ is strictly prohibited**: use in conditional branches (causes death locks)

## 12. Advanced Features

### @T.macro
```python
@T.macro
def Softmax(acc_s, acc_s_cast, scores_max, scores_sum):
    T.reduce_max(acc_s, scores_max, dim=1)
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
    T.reduce_sum(acc_s, scores_sum, dim=1)
    T.copy(acc_s, acc_s_cast)
```
- **Fact**: define reusable macro operations

### T.annotate_layout / T.use_swizzle
```python
from tilelang.intrinsics import make_mma_swizzle_layout
T.annotate_layout({
    A_shared: make_mma_swizzle_layout(A_shared),
    B_shared: make_mma_swizzle_layout(B_shared),
})
T.use_swizzle(panel_size=10, enable=True)
```
- **Activation**: Memory Layout Optimization and L2 Cache Scanning

### Type Conversion
```python
result = A[i].astype("float") * B[i].astype("float")
normalized.astype("float16")
```
- **Activation**: Visible data type conversion

## data type support

### Enter data type
- `float16`: Semi-accuracy floating point
- `float32`: Single accuracy floating point
- `bfloat16`: Brain Float 16
- `int8`: 8-bit integer
- `int32`: 32-bit integer

### Cumulative data type
- `float` / `float32`: Single accuracy floating point (recommended for a pressurizer)

### Output data type
- Usually the same type of input
- Assignable via `.astype()`

## Use recommendations

1. **Select the appropriate abstract level**: Level 2 for most applications, Level 3 for extreme performance optimization
2. **Rational memory allocation**: shared memory data for frequent access, memory clips for accumulation and temporary storage
3. **Optimized data movement**: copying with `T.Parallel` data in parallel, overlapping with `T.Pipelined`
4. **Select the appropriate thread**: usually 128 or 256, taking into account hardware characteristics and work loads
5. **Using the built-in original language**: Optimizing the original language using `T.gemm`, `T.reduce_sum`, etc., and avoiding duplication of existing functions

## 13. Step Copy (async_copy)

```python
# → shared copy of the walk-through global
T.async_copy(A[by * BM, ko * BK], A_s)
# ..independent work...
T.ptx_wait_group(0)  # Must be consumed A_s Call Before
# ThreadSync ("shared") automatically inserts the shared memory barrier before the first reading of A_s
T.gemm(A_s, B_s, C_f)
```

- `T.async_copy` has to be down to `cp.async` or the compilation failed.
- compiler can freely choose how to sync/step down when using `T.copy`

## 14. TMA (Tensor Memory Accelerator, SM90+)

```python
T.tma_copy(desc, A_shared)      # TMA Off-Step Copy
T.alloc_descriptor(kind, dtype) # Distribute Descriptors
T.tma_store_arrive(...)
T.tma_store_wait(...)
```

## 15. Memory Distribution Extension

```python
T.alloc_tmem(shape, dtype)        # Tensor Memory (SM100+)
T.alloc_barrier(arrive_count)     # Allocation mbarrier
T.alloc_wgmma_desc(dtype='uint64')      # WGMMA Descriptor
T.alloc_tcgen05_smem_desc(dtype='uint64') # TCGEN05 shared memoryDescriptor
T.empty(shape, dtype='float32')   # Function OutputtensorStatement
```

## 16. Warp Operations

### 16.1 Warp Vote / Ballot

```python
T.any_sync(predicate [, mask])  # __any_sync
T.all_sync(predicate [, mask])  # __all_sync
T.ballot(predicate)             # All votes cast. (uint64)
T.ballot_sync(predicate [, mask])  # Conditional vote
T.activemask()                  # Active Thread Mask
```

### 16.2 Warp Shuffle

```python
T.shfl_sync(value, src_lane)     # Radio
T.shfl_xor(value, delta)         # XOR Switch.
T.shfl_down(value, delta)        # Move Down
T.shfl_up(value, delta)          # Move Up
```

### 16.3 Warp Match (SM70+, non-HIP)

```python
T.match_any_sync(value [, mask])
T.match_all_sync(value [, mask])
```

### 16.4 Block Predicated Sync

```python
T.syncthreads_count(predicate)
T.syncthreads_and(predicate)
T.syncthreads_or(predicate)
```

## 17. Advanced Sync

```python
T.sync_threads(barrier_id, arrive_count)  # The tape. mbarrier
T.sync_warp(mask)                         # Warp Sync
T.sync_grid()                             # Collaborative grid barrier
T.pdl_trigger()                           # programmable start signal complete.
T.pdl_sync()                              # Waiting for dependency satisfaction
```

## 18. WMMA and Warp Group

```python
T.mbarrier_wait_parity(barrier, parity)
T.mbarrier_arrive(barrier)
T.fence_proxy_async(...)
T.warpgroup_fence_operand(...)
T.warpgroup_arrive()
T.warpgroup_commit_batch()
T.warpgroup_wait(num_mma)
T.wait_wgmma(id)
```

## 19. Storer Control (SM90+)

```python
T.set_max_nreg(reg_count, is_inc)
T.inc_max_nreg(n) / T.dec_max_nreg(n)
T.annotate_producer_reg_dealloc(n=24)
T.annotate_consumer_reg_alloc(n=240)
```

## 20. Custom Inline Functions

```python
T.dp4a(A, B, C)       # 4 Element point accumulation plus
T.clamp(x, lo, hi)    # The plier is here. [lo, hi]
T.loop_break()        # Loop Break
```

## Common Errors

1. **Memory distribution too large**: exceeding hardware limitations
2. **pipeline is not Depthly**: Impact Performance
3. **Thread count mismatch**: low hardware utilization
4. **data type's mismatch**: accuracy's loss or reduced performance
5. **⚠ ️ Synchronise Use Error**: `T.sync_threads()` in Conditional Branch leads to a dead lock
6. **⚠️ Thread Index Retrieving Error**: Using `T.get_thread_binding()` instead of `T.Parallel()`
