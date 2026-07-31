---
name: tilelang-ascend-example-matmul
description: "The standard matrix multiplication TileLang Ascend Express model achieves an example. Cube core program: L1/L0C active memory allocation, T.gemm_v0 call, T. Scape (\"C\") nuclear separation, T.barrier_all() sync, K-D-Rype add. The code structure of this example can be consulted when generating matmul type operator."
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "matmul"
---

# matrix multiplication - TileLang Ascend Achievement Example (Expert Mode)

**Programming mode**: Express (manual management of L1/L0C memory level)

**Key technical points**:
- `T.alloc_L1` / `T.alloc_L0C` Visible Memory Allocation
- `T.gemm_v0(A_L1, B_L1, C_L0C, init=(k==0))` Cube Matrix Multiplication
- `T.Scope("C")` Cube Nuclear Separation
- `T.barrier_all()` Nuclear Synchronization
- K-Direct Cyclops, first iterative `init=True` Zero L0C

```python
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

**Called**:

```python
func = matmul(M, N, K, 128, 256, 64)
c = func(a, b)
```

**Constraint**: M, N must be an integer multiple of block_M, block_N. The non-integrated scene needs to be trimmed after zero-padding on the Python level, or using the Devloper mode + `T.ceildiv`.
