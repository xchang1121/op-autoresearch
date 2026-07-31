---
name: pypto-case-matvec
description: "Matrix-vector Multiplication: K > elementwiseul + sum instead of matmul"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "matrix_vector,matvec,large_k"
---

# Matrix-Vector Multiplication (K > 65535)

A: (256, 131072), B: (131072, 1) -> C: (256, 1)

K = 131072 exceeding the `pypto.matmul` limit (last dimension < = 65535) and replaced by `sum(a * b_row, dim=1)`.

```python
def create_matvec_sum_kernel(a_shape, b_shape):
    out_shape = (a_shape[0], 1)

    @pypto.frontend.jit(...)
    def matvec_sum_kernel(
            a: pypto.Tensor(a_shape, pypto.DT_FP32),
            b_row: pypto.Tensor(b_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(out_shape, pypto.DT_FP32):
        output = pypto.tensor(list(out_shape), pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 8192)
        output[:] = pypto.sum(a * b_row, dim=1, keepdim=True)
        return output
    return matvec_sum_kernel

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        assert A.dim() == 2
        assert tuple(A.shape) == (256, 131072)
        assert B.dim() == 2
        assert tuple(B.shape) == (131072, 1)
        A = A.contiguous()
        # B: (K, 1) - > (1,K) Multiplication for broadcasting
        B_row = B.contiguous().reshape(1, -1)
        return create_matvec_sum_kernel(tuple(A.shape), tuple(B_row.shape))(A, B_row)
```

Key points: `B.reshape(1, -1)` in Forward converts column vector into line vector to make `a * b_row` broadcastable.
