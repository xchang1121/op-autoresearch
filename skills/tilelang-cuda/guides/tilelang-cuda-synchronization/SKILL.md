---
name: tilelang-cuda-synchronization
description: "TileLang CUDA Synchronization, including T.sync_threads() use rules, thread security best practice and deathlock prevention policy. This applies to the preparation of a TileLang nuclear code generation scenario involving shared memory access, multi-wire collaboration, or the need to avoid synchronisation of dead locks."
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
---

# TileLang CUDA Sync and Thread Security

This document details the specifications and thread security best practice for `T.sync_threads()` in TileLang.**Synchronization is the most common and dangerous source of error in TileLang programming.**

---

## 1. T.sync_threads() Use norms

### ⚠ ️, strictly forbidden use of scenes.

#### 1.1 Synchronization in Conditional Branches

```python
# Example of ❌ error - could cause a dead lock
if condition:
    T.sync_threads()  # Only part of the thread is executed, and the others are always waiting.

# ✅ Correct Practice - Synchronize all threads
T.sync_threads()
if condition:
    # Synchronization Operations
```

#### 1.2 Synchronization of conditions in the cycle

```python
# ❌ Error Example - Deadlock Risk
for i in range(n):
    if tid < threshold:
        T.sync_threads()  # The risk of a dead lock.

# ✅ Correct Practices
for i in range(n):
    T.sync_threads()  # Synchronize all threads
    if tid < threshold:
        # Synchronization Operations
```

#### 1.3 Synchronization of conditions after shared memory distribution

```python
# Example of ❌ error
if tid < N:
    shared_mem = T.alloc_shared((N,), dtype)
    T.sync_threads()  # Only part of the line is allocated.shared memory

# ✅ Correct Practices
shared_mem = T.alloc_shared((N,), dtype)
T.sync_threads()
if tid < N:
    # Use shared memory
```

---

## 2. ❌ Absolute prohibition: manual return

Manual surrender is the most common cause of death of a linear card.**The built-in attribute function**must be used.

### The danger of a manual return.

```python
# ❌ absolutely forbidden: date by hand leads to death of thread card
while stride > 0:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    T.sync_threads()  # It's a dead lock. It's a dead thread.
    stride //= 2

# ❌ is absolutely forbidden: Synchronization in the conditional branch
if condition:
    T.sync_threads()  # The risk of a dead lock.

# Absolute prohibition of ❌: Synchronization of conditions in the cycle
for i in range(n):
    if tid < threshold:
        T.sync_threads()  # The risk of a dead lock.
```

### Symptom recognition
- **UTL (GPU utilization) is full but MEM is low**: usually the thread card dies at the sync point
- **kernel never returns**: Infinite waiting by death lock
- **Very poor performance**: Improper sync mode leads to serialization

---

## 3. ✅ correctly synchronized mode

### 3.1 Synchronize shared memory operations

```python
# Sync after ✅ writing shared memory
shared_mem[tid] = value
T.sync_threads()  # Ensure that all writings are completed

# ✅ Read shared memory Sync
T.sync_threads()  # Ensure that all writings are completed
result = shared_mem[tid]
```

### 3.2 Ensure that all threads are synchronized

```python
# ✅ correctly synchronized mode
T.sync_threads()  # All threads must be executed.
# Follow-up
```

### 3.3 Avoiding unnecessary synchronization

```python
# ❌ Oversync
for i in range(n):
    T.sync_threads()  # It's all synchronized. It's expensive.

# ✅ only sync if necessary
# Add Sync only when data is dependent
```

---

## 4. Internal function recommended by ✅ (replaces manual sync)

### 4.1 Inline subdivision function

```python
# ✅ Recommendation: Internal attribute function, no manual synchronization
T.reduce_sum(input_tensor, output_tensor, dim=axis)     # Peace be with you.
T.reduce_max(input_tensor, output_tensor, dim=axis)     # Maximum return
T.reduce_min(input_tensor, output_tensor, dim=axis)     # Minimal return
T.reduce_mean(input_tensor, output_tensor, dim=axis)    # Average return
```

### 4.2 Internal memory operation

```python
# ✅ recommended: built-in memory operation, autoprocessing sync
T.copy(src, dst)   # Efficient memory copying
T.clear(tensor)    # Zero Operations
T.fill(tensor, value)  # Filling Operations
```

### 4.3 Inline Mathematical Functions

```python
# ✅ Recommendations: Inline Math Functions
T.rsqrt(x)        # Square root countdown
T.sqrt(x)         # Square root
T.exp(x)          # Index Functions
T.log(x)          # Logarithm
```

---

## 5. Use example for the built-in attribute function

### 5.1 Convention in Layer Norm

```python
A_pow_local = T.alloc_fragment((M, N), "float32")
A_powsum = T.alloc_fragment((M,), "float32")

# ✅ replaces manual sync with built-in contract
T.reduce_sum(A_pow_local, A_powsum, dim=1)  # We'll make it right.
```

### 5.2 Contracts in Softmax

```python
A_exp = T.alloc_fragment((M, N), "float32")
A_sum = T.alloc_fragment((M,), "float32")
A_max = T.alloc_fragment((M,), "float32")

# ✅ uses the built-in contract
T.reduce_max(A_exp, A_max, dim=1)    # Maximum value by line
T.reduce_sum(A_exp, A_sum, dim=1)    # We'll make it right.
```

### 5.3 Example of complete safe return

```python
@tilelang.jit(out_idx=[-1])
def safe_reduction(M, N, block_size):
    @T.prim_func
    def main(x: T.Tensor((M, N), "float32"),
             y: T.Tensor((M,), "float32")):

        with T.Kernel(M, threads=block_size) as bx:
            # Distribute Memory Snippets
            x_local = T.alloc_fragment((N,), "float32")
            sum_local = T.alloc_fragment((1,), "float32")

            # Loading data
            for tid in T.Parallel(N):
                x_local[tid] = x[bx, tid]

            # ✅ safe return
            T.reduce_sum(x_local, sum_local, dim=0)

            # Write back the results.
            y[bx] = sum_local[0]

    return main
```

---

## 6. Recommended parallel mode of calculation

### 6.1 Use of T. Parallel

```python
# ✅ recommended: use T. Parallel for parallel calculations
for i, j in T.Parallel(M, N):
    result[i, j] = input[i, j] * scale[i]
```

### 6.2 Use of T.vectorized

```python
# ✅ Recommendation: vector integration operation
for i in T.vectorized(N):
    result[i] = input[i] * scale
```

### 6.3 Use of T. Pipelined

```python
# ✅ Recommendation: Software pipeline
for k in T.Pipelined(K, num_stages=3):
    T.copy(A[k], shared_A)
    T.gemm(shared_A, shared_B, result)
```

---

## 7. Thread Index Retrieval

### ✅ Recommendation: T. Parallel

```python
# ✅ recommendations: use T. Parallel to get a thread index
for tn in T.Parallel(BLOCK_N):
    # tn corresponds to threadIdx.x, automatic processing thread map
    pass
```

### ⚠ ️ does not recommend: T. Get_thread_binding

```python
# ⚠ ️ does not recommend: may cause problems
tn = T.get_thread_binding(0)  # threadIdx.x
tk = T.get_thread_binding(1)  # threadIdx.y
```

**Not recommended**:
- Could result in a linear map error
- Manually manage a thread index that is prone to error
- `T.Parallel` provides safer, simpler alternatives

---

## 8. Summary of Important Warnings

### ❌ is absolutely forbidden
1. **Manual return**: will result in the death of a thread card, UTL full but MEM low
2. **Sync in Conditional Branch**: Could result in a dead lock
3. **Consequencing conditions in cycle**: would result in a dead lock
4. **shared memory distribution in the Conditional Branch**: would result in undefined behaviour

### ✅ must be followed
1. **Use internal attribute functions**: `T.reduce_sum()`, `T.reduce_max()`, etc.
2. **To ensure that all threads are synchronized**: `T.sync_threads()` must be on all threads to execute the path
3. **Record thread index with T. Parallel**: replace `T.get_thread_binding()`
4. **Use of built-in memory operation**: `T.copy()`, `T.clear()`, etc.

### Debug Recommendations
- **UTL High but MEM Low**: Check for death of threads as a result of manual return
- **kernel does not return**: check for synchronisation in conditional branches resulting in death locks
- **Result error**: Check if the sync point is correct and if the data is not used after the sync
