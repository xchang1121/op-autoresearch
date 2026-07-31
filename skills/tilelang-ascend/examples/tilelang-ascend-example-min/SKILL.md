---
name: tilelang-ascend-example-reduction
description: "The TileLang Ascend Express model achieves the example. The code structure of this example and the double buffering pipeline model can be consulted when you generate the operator category of reduce."
category: example
version: "2.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduction"
---

# Reduce_min - TileLang Ascend Pipeline (Expert mode)

**Programming mode**: Express (manual `T.Scope("V")` + `T.barrier_all()` + double buffering pipeline)

**Key technical points**:
- `T.alloc_ub` UB Memory Allocation (double buffering with `stages` dimensions)
- `T.reduce_min` Return Operation (same as `T.reduce_max` / `T.reduce_sum`)
- `T.Scope("V")` Victor Nuclear Separation
- `T.barrier_all()` Nuclear Synchronization
- `VEC_NUM = 2` Double nucleus parallel, `sub_M` block slice
- `stages = 2` double buffering pipeline: Current block calculation parallels the next piece of data removal

```python
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[1], target="ascendc")
def reduce_min_pipeline(M, N, block_M, block_N, sub_M, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    stages = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            vec_proc = block_M // sub_M

            a_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((stages, sub_M // VEC_NUM), dtype)

            with T.Scope("V"):
                T.barrier_all()

                T.copy(A[bx * block_M + vid * sub_M // VEC_NUM + 0 * sub_M, by * block_N], a_ub[0, :, :])
                T.barrier_all()

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc - 1:
                        T.barrier_all()
                        T.copy(A[bx * block_M + vid * sub_M // VEC_NUM + (mm + 1) * sub_M, by * block_N],
                               a_ub[nxt, :, :])
                        T.barrier_all()

                    T.barrier_all()

                    T.reduce_min(a_ub[cur, :, :], b_ub[cur, :], dim=-1)

                    T.barrier_all()

                    T.copy(b_ub[cur, :], B[bx * block_M + vid * sub_M // VEC_NUM + mm * sub_M])
                    T.barrier_all()

    return main
```

**Called**:

```python
func = reduce_min_pipeline(M, N, block_M, block_N, sub_M)
c = func(a)
```

**Constraint**: M must be the integer multiple of block_M and N must be the integer multiple of block_N. The non-integrated scene needs to padding or using the Devloper mode + `T.ceildiv`.