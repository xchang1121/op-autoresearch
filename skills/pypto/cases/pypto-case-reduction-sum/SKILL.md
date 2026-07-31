---
name: pypto-case-reduction-sum
description: "Example of one-axis alignment: 3D Sum description - maintain original dimensions, simplest kernel"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "reduction"
---

# Single axle: Sum Reduction (3D)

Simplest mode -- no loop/view/assemble, kernel only 3 rows.

```python
def create_sum_reduction_kernel(in_shape, out_shape):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        x: pypto.Tensor(in_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(out_shape, pypto.DT_FP32):
        output = pypto.tensor(list(out_shape), pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 16, 256)
        output[:] = pypto.sum(x, dim=1, keepdim=True)
        return output
    return kernel
```

Forward:**Maintain original dimensions without decline**.

```python
def forward(self, x):
    assert x.dim() == 3
    assert tuple(x.shape) == (16, 256, 256)
    assert self.dim == 1
    x = x.contiguous()
    batch, _, dim2 = x.shape
    return create_sum_reduction_kernel(
        tuple(x.shape), (batch, 1, dim2)
    )(x)
```

## Elements of a model
- **Maintenance of input original dimensions**, no reshape → file arguments = input rank
- Kernel Extreme: `set_tile + sum + return`, no loop/view/assemble
- `pypto.amin` / `pypto.amax` Same, change API only
- = `sum * (1.0 / size)` (no built-in means API)
- For a 3D single-axis contract for `(16, 256, 256), dim=1`, the default preference starts with `(1, 16, 256)` and is compared to `32/64`.
- Do not apply the mid-point of the loop to the file; this fixed shape is achieved directly by `(1, 16, 256)` as default.
- This fixed shape, `(1, 32, 256)` and `(1, 64, 256)` are not used as default templates, but only as cross-check candidates.

## Rediction_over_a_disarmament series
- `get_init_inputs()` return value for this series title is**the current fixed parameter**(e.g. `dim=1`).
- The `Example, change to desired dimension` in the note is a description of the library, not the current target.
- When generating the code:
  - Retain `ModelNew.__init__(dim)` signatures;
  - `assert self.dim = = < fixed value > for `forward`;
  - kernel uses a fixed constant `dim = a fixed value ' and does not write to the `if dim == ...` branch;
  - Fixed dim scene `create_*_kernel` no longer receives `dim` runtime parameters.

In retrospect:
```python
def create_xxx_kernel(in_shape, out_shape, dim):
    ...
    output[:] = pypto.amin(x, dim=dim, keepdim=True)
```

Normal (fixed dim written to die):
```python
FIXED_DIM = 1
def create_xxx_kernel(in_shape, out_shape):
    ...
    output[:] = pypto.amin(x, dim=FIXED_DIM, keepdim=True)
```
