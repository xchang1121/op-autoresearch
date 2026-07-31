---
name: tilelang-ascend-matmul
description: "TileLang Ascend matrix multiplication operator Encoding Guide. The GEM code model that covers two models: Express/Developoper, K-Direct Rygold, transferred-GEMM, GEMM's non-integrated dimension processing. Reference is made to this guide when generating matmul / GEMM / Linear Layer operator."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "matmul"
---

# TileLang Ascend matrix multiplication Coding Guide

---

## Decision tree: matmul operator path

**Important**: `T.reduce_sum/max/min` and `T.tile.*` can be used in both Devloper and Express modes**. Model selection depends on whether manual memory levels and synchronisation are required, rather than which API is used.

```
Ham matmul / @ / Matrix Multiplication
├─ only matmul → Pure Cube
│   Mode: Expert(Manual management) L0)
│   API(Ascend Special): T.gemm_v0(A_L1, B_L1, C_L0C, transpose_A, init)
│   Memory (%1)Expert): T.alloc_L1 → T.alloc_L0C
│   Memory (%1)Developer): T.alloc_shared → T.alloc_fragment
│   Sync: T.barrier_all() + T.Scope("C")
│   Kernel: T.Kernel(One-dimensional., is_npu=True) as (cid, _)
│
└─ matmul + element-wise Reprocessing → Mixed (integration)operator)
    Mode: Developer + AutoSync (Recommended) or Expert + Synchronise Manually
    API: T.gemm_v0 + T.tile.* / T.Parallel + workspace
    Memory: GM→L1→L0A/L0B→L0C→workspace→UB→GM
    workspace: Number/shape/dtype Auto extrapolation, in GM
    pass_configs: AUTO_CV_COMBINE:True + AUTO_CV_SYNC:True + AUTO_SYNC:True
    Sync: Auto (%1)AUTO_CV_SYNC) or manual ()T.set_cross_flag / T.wait_cross_flag)
```

