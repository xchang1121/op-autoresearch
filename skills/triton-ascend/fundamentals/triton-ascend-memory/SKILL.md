---
name: triton-ascend-memory
description: "Ascend NPU memory access optimization strategy, including UB (Uniform Buffer Zone), data layout optimization, combined access memory and prefeeding techniques. Application to memory bandwidth restricted, need to optimize data handling efficiency, or kernel code performance optimization scenario for processing large-scale data"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Memory access optimization

## Block size selection rationale

The size of the block needs to be balanced by the type of operator and the hardware storage level:

- **VEC class operator**(election-wise, reduce, softmax, etc.): data need to be placed in UB (192KB/VEC), `BLOCK_SIZE * sizeof(dtype)` need to be less than UB available capacity, taking into account the parallelity of calculation. Too small parallels are insufficient and too large a spill UB
- **CUBE class operator**(matmul, attent etc.): Left matrix L0A (* KB), Right matrix L0B (* KB), result L0C (* KB), specific reference hardware information document:
  - `m0 * k0 * sizeof(A.dtype) ≤ * KB`(L0A)
  - `k0 * n0 * sizeof(B.dtype) ≤ * KB`(L0B)
  - `m0 * n0 * sizeof(C.dtype) ≤ * KB`(L0C)
- All data transfers are aligned with**256 Bytes**, BLONK_SIZE is 32 times best

## 2D data: priority tl.make_block_ptr

```python
A_block_ptr = tl.make_block_ptr(
    base=A_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
)
a = tl.load(A_block_ptr, boundary_check=(0, 1))
# Move Pointer
A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
```

## Continuous memory: one-dimensional access

Uncontinuing tensor first converts `.contiguous()`, then accesss with one-dimensional ptr + offsets:

```python
class ModelNew(torch.nn.Module):
    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        kernel[grid](x, out, n, BLOCK_SIZE=1024)
        return out
```

A 1-D visit is more efficient than a three-dimensional visit and recommended for priority use.

If the host side does not convert the non-continuous input to `.contiguous()`, Kernel cannot assume that the one-dimensional `ptr + offsets` corresponds to the order of the logical elements; then it has to go into and use the stride in a visible fashion, or first at the host side, in a continuous tensor.

Common security writing:

- If you use `x.numel()` + 1-D offsets, put the input first on the side of `.contiguous()`.
- matmul / transpose / broadcast /Non-last dimensionreduceIf it's not materialized,tensor  Gotta put  `stride()`Pass it.kernel, and calculated by the logical dimensionoffset.
- If the output is written on a one-dimensional continuous basis, the output is created using either `torch.empty_like()` or `torch.empty(..., device=x.device, dtype=x.dtype)`.

## Alignment Requirements
- Ascend 256B alignment: elect-wise/redance operator
- Ascend 512B alignment: MatMul split
- Data removal bandwidth cap of approximately 256*256B to design removal policy

## Points
- Priority `.contiguous()` + 1-D access
- Continuous memory access is much more efficient than the cost of calculation for the distance
