---
name: tilelang-ascend-example-elementwise
description: "An example of TileLang Ascend from Broadcast (row extension). Displays a pure Vector core program: T.alloc_ub UB RAM distribution, T.tile.broadcast vector radio, T. Scope (\" V\") Vector domain, T. Barrier_all() nuclear synchronise. The code structure of this example can be consulted when generating editionwise-type operator."
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise"
---

# Broadcast (line extension) - TileLang Ascend

**Programming mode**: Expert (pure Victor nuclei, manual UB memory level)

**Key technical points**:
- `T.alloc_ub` UB Memory Allocation (Vector Special)
- `T.tile.broadcast` vector Broadcasting (extension of 1×N to sub_block_M×N)
- `T.Scope("V")` Victor domain label
- `T.barrier_all()` kernel Cube/Vector Sync
- `vid` sub-item vector ID, split block_M into 2 Victor nuclear parallel processing

```python
import tilelang
from tilelang import language as T


@tilelang.jit(out_idx=[1])
def broadcast(M, N, block_M, dtype="float"):
    m_num = M // block_M
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([1, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1, N), dtype)
            b_ub = T.alloc_ub((sub_block_M, N), dtype)

            row_base = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                T.copy(A[0, :], a_ub)

                T.barrier_all()
                T.tile.broadcast(b_ub, a_ub)
                T.barrier_all()

                T.copy(b_ub, B[row_base : row_base + sub_block_M, :])

    return main
```

**general mode of**elementwise class operator**:
1. Data from Global → UB (`T.copy`)
2. Perform element-by-item calculations on UB (`T.tile.add/mul/sub/div/exp` or `T.Parallel` loop)
3. From UB → Global (`T.copy`)
4. No `T.Scope("C")`/ `T.gemm_v0` for pure Victor operation
