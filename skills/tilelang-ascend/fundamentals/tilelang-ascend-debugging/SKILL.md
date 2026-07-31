---
name: tilelang-ascend-debugging
description: "TileLang-Ascend operator coding problem."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang-Ascend operator coding problem

## Checklist

After generating code, check item by item:

### Basic inspection

| # | Checkpoint |
|---|--------|
| 1 | `out_idx` corresponds to the position of output parameter in the function signature |
| 2 | `block_M // VEC_NUM` used consistently in the Buffer Allocation and Index |
| 3 | Shape product of all `T.alloc_ub`s does not exceed UB capacity |
| 4 | Expert mode with `T.Scope("V")` and `T.barrier_all()` |
| 5 | Devloper mode with corresponding `pass_configs` |
| 6 | Test contains at least 2 configurations (small + typical size) |
| 7 | The Golden function is performed using the PyTorch standard |

### Combining operator check

| # | Checkpoint | Annotations |
|---|--------|------|
| 8 | **workspace_idx corresponds to function signature** | Workspace parameter position correct |
| 9 | **AUTO_CV_COMBINE / AUTO_CV_SYNC Configuration** | Devloper mode to open |
| 10 | **Cube → workspace → Victor data stream is correct** | T.copy handle path complete |
| 11 | **Nuclear separation matches pass_configs** | Devloper mode does not need to be visible T. Scope |

## 1. How to deal with dynamic Shape?

Use `T.symbolic`:
```python
N = T.symbolic('N', 'int32')
```

## 2. How do you achieve operator with parameters?

Use function parameters to pass:
```python
def my_op(M, N, block_M, param1=0.1, dtype="float"):
    @T.prim_func
    def main(...):
        T.tile.add(a_ub, a_ub, param1)
```

## 3. How are non-2D data treated?

Adjust index and segment policy:
```python
@T.prim_func
def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):

@T.prim_func
def main(A: T.Tensor((B, M, N), dtype), ...):
```

## 4. How can memory use be optimized?

1. Enable automatic memory planning
2. Restart Middlebuffer
3. Avoid unnecessary buffer distribution

## 5. scalar and vector must be operated using T. t.tile API, scalar can only be placed in the second operating number, and some API does not support scalar

The arithmetic between scalar and vector does not allow the operation of an operator such as `+ - * /`, and must use an API series of `T.tile`.

Correct practice:

| Expressions to write | Means of implementation |
|------------|---------|
| `1.0 - x` | `T.tile.mul(x, x, -1.0)`, again, `T.tile.add(x, x, 1.0)`. |
| `2.0 / x` | `T.tile.div(dst, T.broadcast(2.0, shape), x)` (**significant**: `T.tile.reciprocal` accuracy inadequate, prohibition on use) |
| `x + 1.0` | `T.tile.add(x, x, 1.0)` |

## 6. T. Kernel (n_num) and T. Serial (n_num) do not mix

- `T.Kernel(n_num, is_npu=True) as (cid, vid)` decides how many blocks to run in parallel, and each block runs again, Kernel body
- `for by in T.serial(n_num):` is a single block internal serial cycle for a nuclear that requires multiple processing of multiple pieces of data

The semantics are independent and do not use `for by in T.serial(n_num)` in the body of `T.Kernel(n_num)`:

```python
# Error: n_num controls block numbers while controlling serial loops, semantic repetition
with T.Kernel(n_num, is_npu=True) as (cid, vid):
    for by in T.serial(n_num): ...  # I don't need this cycle, every one. (cid, vid) Just deal with yourself. partition
```

## 7. Prohibit the use of Python built-in functions in Kernel

The TVM `Expr` in TileLang Kernel is a symbolic expression and cannot be operated using Python 's built-in functions (e. g. `min()`, `max()`, `and`, `or`, `not`).

```python
# Error ❌ - Ban Python min()
hw_end = min(hw_start + block_HW, H * W)       # ❌ Not supported Python min

# Correct ✅ - Use T.min
hw_end = T.min(hw_start + block_HW, H * W)
```