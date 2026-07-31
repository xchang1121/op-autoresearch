---
name: pypto-case-matmul-2d
description: "Example for mode B: 2D matrix multiplication + M-V loop segment + tail processing"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "matmul,loop,linear,bias_add"
---

# Mode B: Matmul + Loop (including tail processing)

```python
def ceil_div(a, b):
    return (a + b - 1) // b

def create_matmul_kernel(m, k, n):
    # Select first in loop_count space, then reverse BASIC_BATCH
    # Default 16/32 to try first when the loop range is around 1 ~128
    TARGET_LOOP_COUNT = 16
    BASIC_BATCH = ceil_div(m, TARGET_LOOP_COUNT)

    full_iterations = m // BASIC_BATCH
    tail = m % BASIC_BATCH
    tail_offset = full_iterations * BASIC_BATCH

    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        a: pypto.Tensor((m, k), pypto.DT_FP32),
        b: pypto.Tensor((k, n), pypto.DT_FP32),
    ) -> pypto.Tensor((m, n), pypto.DT_FP32):
        pypto.set_cube_tile_shapes([128, 128], [32, 128], [256, 256], True, False)
        c = pypto.tensor([m, n], pypto.DT_FP32)

        for idx in pypto.loop(0, full_iterations, 1, name="LOOP_M", idx_name="idx"):
            offset = idx * BASIC_BATCH
            a_chunk = pypto.view(a, [BASIC_BATCH, k], [offset, 0])
            c_chunk = pypto.matmul(a_chunk, b, pypto.DT_FP32)
            pypto.assemble(c_chunk, [offset, 0], c)

        if tail > 0:
            a_tail = pypto.view(a, [tail, k], [tail_offset, 0])
            c_tail = pypto.matmul(a_tail, b, pypto.DT_FP32)
            pypto.assemble(c_tail, [tail_offset, 0], c)

        return c
    return kernel
```

Forward: assert → contigous → read the page → to Kernel

**3D input +2D B**:forward calculates `nm = N * M`, `A.reshape(nm, K)` → to transfer `nm` to the plant function (do not pass N-M separately):
```python
def forward(self, A, B):
    N, M, K = A.shape
    nm = N * M
    A_2d = A.reshape(nm, K)
    result_2d = create_matmul_kernel(nm, K, L)(A_2d, B)
    return result_2d.reshape(N, M, L)
```

## Matmul + Bias (Linear)

`linear = matmul + bias` does not plug `add` directly into the cube phase. `matmul` is cube op, `add/expand_clone` is vec op, and the tile must be changed in a visible way.

```python
def create_linear_kernel(m, k, n):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        x: pypto.Tensor((m, k), pypto.DT_FP32),
        w: pypto.Tensor((k, n), pypto.DT_FP32),
        b_row: pypto.Tensor((1, n), pypto.DT_FP32),   # forward in b.reshape(1, -1)
    ) -> pypto.Tensor((m, n), pypto.DT_FP32):
        # Phase 1: cube matmul
        pypto.set_cube_tile_shapes([128, 128], [32, 128], [256, 256], True, False)
        mm = pypto.tensor([m, n], pypto.DT_FP32)
        for idx in pypto.loop(0, full_iterations, 1, name="LOOP_M", idx_name="idx"):
            off = idx * BASIC_BATCH
            x_chunk = pypto.view(x, [BASIC_BATCH, k], [off, 0])
            y_chunk = pypto.matmul(x_chunk, w, pypto.DT_FP32)
            pypto.assemble(y_chunk, [off, 0], mm)

        # Phase 2: vec bias add
        pypto.set_vec_tile_shapes(1, n)
        b_full = pypto.expand_clone(b_row, [m, n])   # Single-axis broadcasts
        out = pypto.add(mm, b_full)
        return out
    return kernel
```

## Points
- Ban `BASIC_BATCH` as a fixed answer; first `loop_count`, then `BASIC_BATCH`.
- When the range of `loop_count` is about `1~128` and the candidate changes by a double step, the mid-term priority is to try `16/32` (in the middle of the logarithmic scale, not the mid-point of arithmetic).
- Example: `loop=16/32` corresponds to `BASIC_BATCH=1024/512` at `m=16384`; `loop=8/64` is extended to `2048/256`.
- Avoids two extremes: neither blindly pursues `loop_count=1` nor defaults to use a minimum bat to get `loop_count` close to the maximum.
- **view Shape must be a compilation constant**: `BASIC_BATCH`, `tail` are all closed constants
- **Ban**`min(BASIC_BATCH, m - offset)` as View Shape (offset with loop variable = runtime value)
- `a_trans=True` / `b_trans=True` supports conversion with no change in structure
- Triangular/symmetric matrix: direct standard matmul
- M ≤ 128 can not loop: `c[:] = pypto.matmul(a, b, ...)`
- When `matmul + elementwise` is mixed, use two stages tile: `set_cube_tile_shapes(...)`, before the vec phase and `set_vec_tile_shapes(...)`.
