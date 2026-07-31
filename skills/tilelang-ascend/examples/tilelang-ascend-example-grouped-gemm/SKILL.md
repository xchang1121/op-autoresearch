---
name: tilelang-ascend-example-grouped-gemm
description: "The TileLang Ascend Express mode of grouping/dynamic batches achieves the example. This example is referenced when operator with grouping/dynamic batches is generated."
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Group matrix multiplication - TileLang Ascend Implementation Example (Expert Mode)

**Programming mode**: Express (manual management of L1/L0C memory level)

**Key technical points**:
- **block_metadata projection**: replace 3-D Kernel with the `[batch_idx, m_start, valid_rows]` table describing the attribution of each block
- **D1 Kernel + manual index breakdown**: `T.Kernel(total_m_blocks * n_num)` + `cid // n_num` / `cid % n_num`
- **Static Cycle Boundary**: `T.ceildiv(K, block_K)` Alternative Dynamic Boundary (TileLang Ascend does not support the number of cycles dependent on tensor value)

## Host side: block_metadata projected

```python
def construct_inputs(batch_sizes_list, K, N, block_M, device, dtype):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)

    A = torch.randn(batch_sum, K, device=device, dtype=dtype)
    B = torch.randn(batch_count, K, N, device=device, dtype=dtype)

    metadata_list = []
    current_global_offset = 0

    for batch_idx, size in enumerate(batch_sizes_list):
        num_blocks = (size + block_M - 1) // block_M
        for i in range(num_blocks):
            local_start = i * block_M
            m_start_global = current_global_offset + local_start
            valid_m = min(block_M, size - local_start)
            metadata_list.append([batch_idx, m_start_global, valid_m])
        current_global_offset += size

    block_metadata = torch.tensor(metadata_list, device=device, dtype=torch.int32)
    return A, B, block_metadata
```

## Kernel: Group GEMM

```python
@tilelang.jit(out_idx=[2])
def grouped_gemm(batch_sizes_list, K, N, block_M, block_N, block_K, dtype="float16"):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = "float32"
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)
    n_num = (N + block_N - 1) // block_N

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),
        B: T.Tensor([batch_count, K, N], dtype),
        C: T.Tensor([batch_sum, N], dtype),
        # Metadata table: [batch_idx, m_start_offset, valid_rows]
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),
    ):
        with T.Kernel(total_m_blocks * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            cur_batch_idx = block_metadata[bx, 0]
            m_start = block_metadata[bx, 1]
            # Partial memory movement (tail handling) is not yet supported;
            # this variable is currently unused.
            _actual_rows = block_metadata[bx, 2]

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    # Copyin
                    T.copy(
                        A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K],
                        A_L1,
                    )
                    T.copy(
                        B[
                            cur_batch_idx,
                            k * block_K : (k + 1) * block_K,
                            by * block_N : (by + 1) * block_N,
                        ],
                        B_L1,
                    )
                    T.barrier_all()

                    # Compute
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                # Copyout
                T.copy(
                    C_L0,
                    C[
                        m_start : m_start + block_M,
                        by * block_N : by * block_N + block_N,
                    ],
                )

    return kernel
```

**Called**:

```python
func = grouped_gemm(tuple([64, 128, 256]), 8192, 8192, 64, 64, 64)
A, B, block_metadata = construct_inputs([64, 128, 256], 8192, 8192, 64, device, dtype)
out = func(A, B, block_metadata)
```

**Design elements**:
- Static cycle boundary + condition judgement (replacement dynamic boundary): `batch_sizes_list` flows to `@jit` layer by tuple and expands to specific values during the compilation period to avoid dynamic cycle boundary
- Expected scale (replacement of 3D Kernel): `block_metadata` is projected from host to pass in Kernel, instead of 3D block number of `T.Kernel`
- `m_start` read from metadata table to achieve different starting offsets between groups
- Kernel: `T.Kernel(total_blocks)` + manual index breakdown
