---
name: pypto-api
description: "PyPTO All API Signature and Constraint Quick Check"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
---

# PyPTO API quick check

## Kernel Decorator

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": _PYPTO_RUN_MODE},
    debug_options={"runtime_debug_mode": _PYPTO_RUNTIME_DEBUG_MODE},
)
def kernel(x: pypto.Tensor(shape_tuple, dtype)) -> pypto.Tensor(shape_tuple, dtype):
    ...
```

## tensor

| API | Purpose | Example: |
|-----|------|------|
| `pypto.Tensor(shape, dtype)` | Type of input/output label | `x: pypto.Tensor((m, k), pypto.DT_FP32)` |
| `pypto.tensor(shape_list, dtype)` | Create output in kernel | `output = pypto.tensor([m, n], pypto.DT_FP32)` |
| `pypto.zeros(shape_list, dtype=)` | Zero Initialization tensor (cumulator) | `acc = pypto.zeros([1], dtype=pypto.DT_FP32)` |
| `pypto.full(shape, val, dtype, valid_shape=)` | Constant Fill tensor | `ones = pypto.full(s, 1.0, pypto.DT_FP32, valid_shape=s)` |

data type: `pypto.DT_FP32`, `pypto.DT_INT32`, `pypto.DT_INT64` (INT64 is used only for input labelling).

## Tile Configuration

| API | Constraints |
|-----|------|
| `pypto.set_vec_tile_shapes(*shapes)` | Number of arguments = operated tensor rank |
| `pypto.set_cube_tile_shapes(m, k, n, l1, split_k)` | Fixed 5 Parameters |

**tie double bound**:
1. `prod(tile_shape)` ≤ 16384
2. `auto_tiles = prod (shape[i]/tile[i] per dimension)) `≤ 2048 (each)

- You must call before any computing operation. A kernel can switch files several times.
- If `auto_tiles > 2048`, priority is changed to `loop + view/assemble` segment.
- **vec file recommended**: `(8192)` (1D), `(1, 16384)` (2D), `(1, 1, 16384)` (3D)
- **cube file recommendation**: `set_cube_tile_shapes([128, 128], [32, 128], [256, 256], True, False)`

## Segments

| API | Purpose | Constraints |
|-----|------|------|
| `pypto.loop(start, end, step, name=, idx_name=)` | Compiler Cycle | It's not embedded. It's not used to the minimum. |
| `pypto.view(tensor, shape, offset)` | **Slice extraction**(Equivalent `tensor[a:b, c:d]`) | Shape Ds are constant, rank is unchanged, ≤ input corresponding dimensions per D |
| `pypto.assemble(chunk, offset, output)` | Write back the chunks (Equivalent Slice) | None |

`pypto.view`**is not reshape**. It is an API equivalent of `tensor[offset[0]:offset[0]+shape[0], ...]`. The dimensions cannot be changed, and the dimensions' layout cannot be changed. All reshapes must be done with torch in forward.

## Calculator

**Operator rule**: `+` `*` supports scalar at any location; `-` `/` requires tensor on the left side (`1.0 - x` rash).
**Function calls**: `pypto.add`/`sub`/`mul`/`div` the first parameter shall be Tensor.
One dollar to reverse `-x`: Not supported, with `pypto.mul(x, -1.0)` or `x * (-1.0)`.
Slice value: `output[:] = expr`

## Math Functions

| Functions | Annotations |
|------|------|
| `pypto.exp(x)` | Index |
| `pypto.log(x)` | logour |
| `pypto.sqrt(x)` | Square root |
| `pypto.abs(x)` | Absolute value [u] |
| `pypto.sigmoid(x)` | sigmoid |
| `pypto.softmax(x, dim=)` | softmax |
| `pypto.maximum(a, b)` | Maximum by element, b could be scalar:`pypto.maximum(x, 0.0)` |
| `pypto.minimum(a, b)` | Minimum by element, b could be scalar:`pypto.minimum(x, 0.0)` |

When a built-in function is available, the manual equivalent formula is prohibited by direct call.

**Disables API**: `pypto.where` (with bugs), `pypto.clamp` (unstable). Conditional logic is achieved using a combination of `maximum`/`minimum`.

## Return

```python
pypto.sum(x, dim=int, keepdim=bool)
pypto.amax(x, dim=int, keepdim=bool)
pypto.amin(x, dim=int, keepdim=bool)
```

**No `pypto.mean` API.**`mean` semantics please use `sum * (1.0 / count)` to accomplish this.
**`dim` accepts only a single `int` and does not support `list`.**Multi-axis returns are subject to repeated calls.
`dim` should be compiled as a closed-pack constant in practice; static missions should not have multiple `dim` runtime branches in the same Kernel.

## matrix multiplication

```python
pypto.matmul(a, b, out_dtype, a_trans=False, b_trans=False)
```

- 2D:`[M,K] @ [K,N] → [M,N]`
- 3D watched: `[B,M,K] @ [B,K,N] → [B,M,N]` (both sides must be together with rank watch,**do not support radio**)
- `a_trans` / `b_trans` supports the conversion.
- **Restriction**: each input of the last dimension ≤ 65535. Replace it with `sum(a * b_broadcast, dim=)` when it exceeds.
- **matmul almost has to match the loop**(M-axis segment) because the mattmul is complex inside, not loop easily time out. Except: M may not be loop when it is small (≤128).

## Type Conversions and Indexes

| API | Example: |
|-----|------|
| `pypto.cast(tensor, dtype)` | `pypto.cast(x, pypto.DT_INT32)` |
| `pypto.unsqueeze(tensor, dim)` | `pypto.unsqueeze(x, 1)` |
| `pypto.gather(tensor, dim, index)` | `pypto.gather(log_probs, dim=1, index=idx)` |
| `pypto.expand_clone(tensor, shape)` | One-axis broadcasts, only one at a time. |
