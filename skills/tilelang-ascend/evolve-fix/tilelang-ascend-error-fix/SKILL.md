---
name: tilelang-ascend-error-fix
description: "TileLang-Ascend operator coding is a common error and fixes a guide. It covers errors in translation (RAM distribution failed, dimensions not matched, API parameter error, GEMM divided by zero, Autotune problem), runtime error (incorrect results, accuracy problem)."
category: fix
version: "1.0.0"
metadata:
  case_type: fix
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang-Ascend operator coding common errors and fixes

---

## Error compiling

### 1. Memory distribution failed

**error message**:
```
TVMError: Memory allocation failed for: buffer_name required: XXXX, new memory available: YYYY
```

**Reason: UB space is inadequate and all buffer sizes exceed limit

**Solution**:
1. Decrease the size of the fraction:
   ```python
   block_M, block_N = 64, 128
   ```
2. Enable automatic memory planning to reuse the buffer:
   ```python
   pass_configs = {
       tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
   }
   ```
3. Reduce the number of intermediate buffer and reuse it to the extent possible

### 2. The dimensions do not match

**error message**:
```
error: Source and Dest dimension must match.
```

**Reason**: source and targetshape of Broadcast operation does not meet requirements

**Solution**:
Ensure that the source's Shape is `[M, 1]` or `[1, N]`. Target is `[M, N]`:

```python
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
T.tile.broadcast(max_2d_ub, max_ub)
```

### 3. API parameter error

**error message**:
```
error: max() takes 3 positional arguments but 4 were given
```

**Reason**: API call parameters are incorrect

**Solution**:
View the API guide to confirm the correct parameter signature:

```python
T.tile.max(dst, src0, src1)
```

### 4. GEMM Uncode Error

**error message**:
```
InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero
 --> ...py:65:18  bx = cid // n_num
```

**Reason: `n_num = N // block_N = 0` (when `block_N > N`) resulting in `cid // 0`.

**Solution**: Ensure that M, N ≥ block size is reached before calling GEMM. If `M < block_M` or `N < block_N`, zero-pading arrays to block multiples call GEMM, then shear and finish.

```python
M_pad = ((M + block_M - 1) // block_M) * block_M
N_pad = ((N + block_N - 1) // block_N) * block_N
K_pad = ((K + block_K - 1) // block_K) * block_K

if M_pad > M or K_pad > K:
    kernel_padded = torch.zeros(M_pad, K_pad, ...)
    kernel_padded[:M, :K] = kernel_flat

output = output[:M, :N]
```

**Key constraints**: `M // block_M = 0` (when M < block_M) does not padding will cause a zero block start (out of all output) or a zero-coding collapse.

### 5. Autotune supply_prog IndexError

**error message**:
```
An error occurred while testing config {...}
```

**Reason: `params` in `supply_prog(params)` only contains input tensor description (excluding output) and `params[2]` access crosses the border.

**Solution**: extraction of dimensions from `params[0].shape` and `params[1].shape`:
```python
def supply_prog(params):
    M_val, K_val = int(params[0].shape[0]), int(params[0].shape[1])
    _, N_val = int(params[1].shape[0]), int(params[1].shape[1])
    return [torch.randn(M_val, K_val).half().npu(), torch.randn(K_val, N_val).half().npu()]
```

### 6. Autotune get_configs parameter format error

**error message**:
```
TypeError: get_configs() missing 1 required positional argument: 'K'
```

**Reason: Autotunner called `get_configs` with `(key_args_tuple, key_kwargs_tuple)`, `((M,N,K), ())`. Direct statement that `get_configs(M, N, K)` will receive tuple instead of three ints.

**Solution**: signed as `get_configs(key_args, _key_kwargs=None)`, unpacking M, N, K from `key_args`. Transferable references (`configs=get_configs`) when calling instead of calling results (`configs=get_configs()`).

### 7. L0C SpillSegfault

**System**: autotune compiled but benchmark process directly rash (Segfault), no Python anomaly.

**One possible reason**: When `block_M * block_N * sizeof(accum_dtype) > L0C_capacity` is used, this may result in the use of a piece of buffer exceeding the hardware limit. For example, A2/A3 device L0C is 128KB, the number of float32 accum elements should not exceed 32768.

**Block: in the solution**: autotone's `get_configs` filtering super big block:
```python
block_M = [bs for bs in [64, 128] if bs <= M]
```

**Autotune filter rule**(must be implemented in `get_configs`):
1. Filter an invalid combination of `block > dimension` (avoiding zero-coding errors)
2. Filters a combination of `block_M * block_N * sizeof(accum_dtype) > L0C_capacity` (avoiding L0C spills). `block_M * block_N ≤ 32768` while A2/A3 device L0C = 128KB, float32acum

```python
def get_configs(key_args, _key_kwargs=None):
    M, N, K = key_args
    configs = []
    for bm in [64, 128]:
        for bn in [64, 128]:
            for bk in [32, 64]:
                if bm > M or bn > N or bk > K:
                    continue
                if bm * bn * 4 > 131072:  # L0C 128KB limit for float32 accum
                    continue
                configs.append({"block_M": bm, "block_N": bn, "block_K": bk})
    return configs
```

### 8. `InternalError: Duplicate buffer name found: tmp_ub`

**Reason: TileLang compiler internal pass automatically creates temporary buffer. These temporary buffer may have a fixed name (e.g. `tmp_ub`) and conflict arises if the same name is customised.

**Common error code**:
```python
@T.prim_func
def main(...):
    with T.Kernel(N, is_npu=True):
        tmp_ub = T.alloc_ub([block_size], "float32")  # ❌ Custom tmp_ub
        T.tile.broadcast(tmp_2d, mean_ub)              # compilerIt'll also be created. tmp_ub
```

**Solution**: Avoid using the following names as user-defined buffer names:
- `tmp_ub`,`tmp`,`tmp_buf`
- `broadcast_workspace_*`
- Other names that may conflict with the compiler internal pass

---

## runtime error

### 1. It didn't work right.

**Possible cause**:
1. Synchronising folder
2. Formula realization error
3. data type Question

**Solution**:

1. Process sync according to programming mode. Expert manual mode requires visible sync; no additional manual barrier is inserted after Devloper / Mixed mode opens `TL_ASCEND_AUTO_SYNC`.
   ```python
   pass_configs = {
       tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
       tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
   }
   ```

2. Check if data type matches

### 2. accuracy Question

#### There's a slight difference between output and reference implementation

**Reason**: lower float16accuracy, cumulative error

**Solution**: cumulative calculation using float32

#### Kernel is running normal, but the output is all 0.

**Possible cause**:
1. scalar and vector did not operate with T.tile API. The algorithm between scalar and vector cannot be written directly by the `+ - * /` operator, and must be done with the `T.tile` series API.
2. `out_idx=[...]` was specified when Kernel compiled, but return value was discarded on call and written as `kernel_func(*tensors)`. When you enter out_idx, the output of kernel returns only through return value and will not be written in situ, so the output is a predefined empty value.

**Solution**:
1. Change `T.tile` API to scalar-vector
2. If kernel compiles with `out_idx=[...]` specified, return value: `outputs = kernel_func(*inputs)` must be received on call.

#### Only the first 64 (float32)/128 (float16) elements were correct, followed by all errors, and Kernel used `T.tile.select`

**Reason: `selMode` for `T.tile.select` is wrongly set to `VSEL_CMPMASK_SPR`. AscendC Selact supports three modes:

| Mode | Enumeration values | Mask Consumption Mode | It's valid for every one of them. | Apply scene |
|------|--------|-------------|--------------|---------|
| VSEL_CMPMASK_SPR | 0 | Repeat the same mask for each traverse | 64 bit (8 bytes) | Mask All Same or Fixed |
| VSEL_TENSOR_SCALAR_MODE | 1 | Continuous consumption per traverse | Unlimited | tensor vs scalar selection |
| VSEL_TENSOR_TENSOR_MODE | 2 | Continuous consumption per traverse | Unlimited | tensor vs tensor selection |

`VSEL_CMPMASK_SPR` uses the same fixed pre-64 bit mask to process all data - if your mask is the result of `compare(x > 0)` (different position bit), this mode leads to a complete error in the selection of the element 64+.

**Solution**: Replace `VSEL_CMPMASK_SPR` with `VSEL_TENSOR_SCALAR_MODE` or `VSEL_TENSOR_TENSOR_MODE`
