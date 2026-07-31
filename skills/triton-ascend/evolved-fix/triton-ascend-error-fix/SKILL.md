---
name: triton-ascend-error-fix
description: triton-ascend common error fixation after failure to verify, compile or run: UB/CBUF spill, BiShengir compilation failure, syntax restriction violation, numerical correctness, multi-dimensional index breakdown error, tensor continuity
category: fix
version: "1.0.0"
metadata:
  case_type: fix
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3, Atlas A5"
---

## 1. UB / CBF Spill

- **Systems**: `cbuf overflow`
- **Gin**: Partition parameter (BLONK_M/N/K) is too large to exceed Ascend UB capacity
- **Refurbishment**: Reduced segment parameters, usually BLONK_M=64, BLONK_N=128 is the safe starting point

```python
# Error: Too large the UB overflow
BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

# Rehabilitation: Safety threshold
# CUBE (matmul fp16): BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
# CUBE (matmul fp32): BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
# VEC (elementwise):  BLOCK_SIZE=1024~2048
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
```

4D tensor matrix multiplication to use the UB dimension extra space, it is proposed to extend the bat to the grid dimension rather than the inner circle.

## 2. BiShengir/ HiVM compilation failed

- **Systems**: `hivm.hir.vsel: Unsupported op for finding the root alloc`, `Failed to run BiShengHIR pipeline`
- **Gin**: compilerZ1XQ does not support complex mask combination or pointer mode

### 2a. Overcomplicated calculation of inline addresses

```python
# Error
tl.store(c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn, acc, mask=mask)

# Repair: Split into intermediate variables
c_ptrs = c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
tl.store(c_ptrs, acc, mask=mask)
```

### 2b. `tl.where` + complex mask leads to vsel error

```python
# Error: Mask + tl. where to trigger hivm.hir.vsel error
a_tri_mask = a_offsets_k[None, :] >= a_offsets_m[:, None]
a_valid_mask = a_mask_m & a_mask_k
a = tl.where(a_tri_mask & a_valid_mask, a, 0.0)

# Fix: Replace tl.where with a multiplication (multiply after Bool mask to float)
a_tri_mask = (a_offsets_k[None, :] >= a_offsets_m[:, None]).to(tl.float16)
a_valid_mask = (a_mask_m).to(tl.float16) * (a_mask_k).to(tl.float16)
a = a * a_tri_mask * a_valid_mask
```

## 3. Triton's Grammar Restrictions Violation

### 3a. Prohibition of `continue` / `break` / `return`

```python
# Error: unsuppleted AST node type: Continue
for i in range(N):
    if condition:
        continue
    do_work()

# Restoration: packaged in if-else
for i in range(N):
    if not condition:
        do_work()
```

### 3b. index error

```python
# Error: ValueError ('unsuppleted tensor index: constexpr [0]')
result = tl.sum(data, axis=0)
tl.atomic_add(out_ptr, result[0])

# Repair: tl.sum returned scalar, directly used
result = tl.sum(data, axis=0)
tl.atomic_add(out_ptr, result)
```

### 3c. tensor.cast type not compatible

```python
# Error:cast incompatible images
result = tl.dot(a_fp16, b_fp16)

# Restoration: Visible designation of fp32 loader
result = tl.dot(a_fp16, b_fp16, acc=tl.zeros([M, N], dtype=tl.float32))
```

### 3d. Post-failure tracking

| Misreporting/incident | Common Roots | Modify Direction |
|----------|----------|----------|
| `unsupported AST node type: Continue` | `continue` / `break` / `return` | Change to `if-else` Package Valid Branch |
| Dynamic `while` loop | Ascend backend does not support dynamic `while` | Use a statically bounded `for` loop with an `if` guard |
| `unsupported tensor index` | Python slice or replay scalar `[0]` | Use `tl.extract_slice`/ Use scalar directly |
| `hivm.hir.vsel` | `tl.where` Select Complex Mask, Pointer or Ofset | Split the static branch, or multiply the data after turning mask dtype |
| `cast incompatible` | Implicit dtype/shape extrapolation failed | Visible accumulator dtype and `.to(dtype)` |

## 4. Numerical correctness issue

- **Systems**: `AsserviceError: output inconsistent, err_cnt=XXXX '

### 4a. Triangular matrix mask error

```python
# Upper Triangle: Ensure that col > = row area is not zero
row_idx = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
col_idx = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
tri_mask = col_idx[None, :] >= row_idx[:, None]
a = tl.load(a_ptr + ..., mask=tri_mask & bounds_mask, other=0.0)
```

### 4b. 4D tensor dimension decomposition error

```python
# Right-watch dimension decomposition
pid = tl.program_id(0)
batch_idx = pid // num_blocks_per_batch
block_idx = pid % num_blocks_per_batch
b0 = batch_idx // dim1
b1 = batch_idx % dim1
```

### 4c. Reduction accuracy lost

```python
# fp32 Thrusters to avoid fp16 accuracy's loss
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(...)  # fp16
    b = tl.load(...)  # fp16
    acc += tl.dot(a, b)  # fp32 Gradient
result = acc.to(tl.float16)
```

## 5. MultiDirect index decomposition error

- **System feature**: Incorrect calculation (no translation misreported)
- **Ghen**: Index decomposition/restoration logic error after multidimensional questions have been spread to one dimension. Common in Norm, Poling, multiple bat operator
- **Rehabilitated**: using clear loop nesting + visible dimension decompose to avoid a fragile one-dimensional stretch map

```python
# Easier to be wrong: complex 1-D mapping
total_tasks = N * G
for task_idx in range(pid, total_tasks, CORE_NUM):
    n_idx = task_idx // G
    g_idx = task_idx % G
    c_local = offsets // S       # It's not very clear. It's easy to write wrong.
    hw = offsets - c_local * S

# Recommendations: Visible multilayered + clear local index
for n in range(pid, N, CORE_NUM):
    for g_idx in range(num_groups):
        for i in range(0, group_elems, BLOCK_SIZE):
            local_idx = i + tl.arange(0, BLOCK_SIZE)
            c_local = local_idx // hw_size
            spatial_idx = local_idx % hw_size
```

## 6. tensor continuity

Force `.contiguous()` at the Kernel wrapper entrance:

```python
if not x.is_contiguous():
    x = x.contiguous()
```

---
## Quick Checklist

1. **Compiled failed+ "ub overflow" / "cbuf overflow"** →Zoom OutBLOCKDimensions§1)
2. **Compiled Failed + "hivm.hir"/ "root alloc"**→ Simplified mark / Split pointer calculation (§2)
3. **Compiled failed+ "unsupported AST"** →Check for forbidden syntax tables (%2)§3)
4. **Certification failed + "err_cnt"**→ check mark orientation, index calculation, accuracy (§4)
5. **Incorrect but no error**→ Check multi-dimensional index decomposition logic (§5)
6. **ResultNaNOr a silent error.** →Check continuity (%)§6)
