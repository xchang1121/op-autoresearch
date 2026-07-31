---
name: tilelang-ascend-example-atomic-add
description: "atomic_add of TileLang Ascend DeveloperMode achieves the example. When you need to generateall-reduceoperatorOr multi-nucleus to add to the same output area.reduceCategoryoperatoryou can refer to this example."
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduction"
---

# Tomic_add (more block atoms) - TileLang Ascend (Developer mode)

**Programming mode**: Devloper (`pass_configs` AutoSync + Memory Planning)

**Key technical points**:
- `T.tile.atomic_add(dst_gm, src_local)` - Multiple block/core atoms added to the same GM area
- Zero GM output before calling (`torch.zero_()` or kernel `T.tile.fill` + `T.copy` zero)
- UB → GM Path: Victor nuclei returns GM from UB atoms
- L0C → GM Path: Cube Nuclear GEMM Result Back GM from L0C Atom
- `pass_configs` Open `AUTO_SYNC` + `MEMORY_PLANNING` without handwritten `T.Scope("V")` or `T.barrier_all()`

## Scene Description

When more than one block/core needs to add its own partial result to the same GM output area, normal `T.copy` will cover each other and `T.tile.atomic_add` must be used to guarantee atom accumulation.

Typical scenario:
- **Spit-K GEMM**: multiple blocks along K-dimensional splits, calculating partial matrix multipliers, atoms added to the same GM output
- **Deportation is not integral**: more blocks do reduce between different lines of the same line, atoms cumulative result to GM
- **Full reduce**: all blocks contribute partial values to the same region, atoms aggregated

## Example I: UB → GM Atomic Plus (1D)

Multiple blocks fly 1.0 each to UB, and then atoms add up to the same GM output. Finally, GM values = num_blocks × VEC_NUM.

```python
import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

num_blocks = 4
tile_n = 32
dtype = "float32"

@tilelang.jit(pass_configs=pass_configs)
def atomic_add_1d(num_blocks, tile_n, dtype):
    @T.prim_func
    def main(C: T.Tensor((tile_n,), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_n,), dtype)

            T.tile.fill(src_ub, 1.0)
            T.tile.atomic_add(C[0], src_ub)

    return main
```

## Example two: UB → GM Atomic Plus (2D region)

Multiple blocks each fill 1.0 to 2D UB, then atoms are added to the same 2D GM area.

```python
tile_m = 4
tile_n = 32
dtype = "float32"

@tilelang.jit(pass_configs=pass_configs)
def atomic_add_2d(num_blocks, tile_m, tile_n, dtype):
    @T.prim_func
    def main(C: T.Tensor((tile_m, tile_n), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_m, tile_n), dtype)

            T.tile.fill(src_ub, 1.0)
            T.tile.atomic_add(C[0, 0], src_ub)

    return main
```