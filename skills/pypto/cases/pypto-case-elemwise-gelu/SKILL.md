---
name: pypto-case-elemwise-gelu
description: "Example A: 1D elementwise - GELU activates, displays a handwritten formula, operator when equal, untangible"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "elementwise,activation"
---

# Mode A:1D Elementwise — GELU

```python
def create_gelu_kernel(flat_size):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def gelu_kernel(
        x: pypto.Tensor((flat_size,), pypto.DT_FP32),
    ) -> pypto.Tensor((flat_size,), pypto.DT_FP32):
        output = pypto.tensor([flat_size], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8192)
        x_cubed = x * x * x
        inner = x + x_cubed * 0.044715
        tanh_arg = inner * 0.7978845608028654
        exp_pos = pypto.exp(tanh_arg * 2.0)
        tanh_val = (exp_pos - 1.0) / (exp_pos + 1.0)
        output[:] = x * 0.5 * (1.0 + tanh_val)
        return output
    return gelu_kernel
```

forward:`reshape(-1)` → kernel → `reshape(x.shape)`

## Elements of a model
- `assert dim + shape`, `reshape(-1)` parsing to 1D in Forward
- `set_vec_tile_shapes(8192)` — 1D requires only one argument
- There's no insider than → `(exp(2x)-1)/(exp(2x)+1)` -- note that in `exp_pos - 1.0` Tensor is legal on the left.
- All `Tensor op scalar` valid; rewrite if `scalar op Tensor` is required
