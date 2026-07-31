---
name: catlass-api-basics
description: "CATLASS 5 Layer API and Gemm assembly paradigms: Device/Kernel/Block/Tile, GemmType/GemmShape/DispatchPolicy, standard header file and Device call process. This applies to the type aliases of.asc/.h that modify the catlass_op in the autoresearch task."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc_catlass
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# CATLASS API Foundation

CATLASS is the matrix-type operator template library on board. AR Tasks,**really change the location of template-type aliases**usually:

- `catlass_op/kernel/catlass_kernel.asc` — Kernel side type with `using` alias
- `catlass_op/include/catlass_kernel.h` — Statement consistent with `.asc`

`ModelNew` in `kernel.py` only for `load_library` and `torch.ops.catlass.*` calls,**do not**rewrite calculation logic in Python.

Typical `task.yaml` path and editable list:

```yaml
catlass:
  root: /path/to/catlass          # CATLASS Repository root (also available for environment variables) CATLASS_ROOT)
  op_dir: catlass_op              # Relative task_dir of pybind Project Directory (folders)
editable_files:
  - kernel.py
  - catlass_op/kernel/catlass_kernel.asc
  - catlass_op/include/catlass_kernel.h
  - catlass_op/src/catlass_torch.cpp
  - catlass_op/CMakeLists.txt
```

`catlass_torch.cpp` is responsible for TORCH_LIBRARY registration, tensor pre-processing and lanch; if operator such as volume shows a higher share of Transdata, it is often necessary to change this document rather than just `.asc`.

## Five-tier structure (high to low)

| Level | Typical entrance. | Duties |
|------|----------|------|
| Device | `Gemm::Device::DeviceGemm<Kernel>` | Host entrance, verification of parameters, lanch |
| Kernel | `BasicMatmul` / `MatmulEpilogue`, etc. | Block + subnucleic + Sync |
| Block | `BlockMmad` | Single K private cycle (MMAD + double buffering) |
| Tile | `TileCopy` / `TileMmad` | L1/L0 Removal and Microkernel |
| Basic | `AscendC::Mmad` / `DataCopy` | Command Level Envelope |

## Standard Gemm assembly order

```cpp
// 1. BlockMmad
using ArchTag = Arch::AtlasA2;  // and objectives NPU Intergenerational consistency
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

// 2. Follow-up (unordered void)
using BlockEpilogue = void;

// 3. Subnucle Swizzle
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// 4. Kernel
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

// 5. Device
using Matmul = Gemm::Device::DeviceGemm<MatmulKernel>;
```

When moving to PyTorch, the**include list should be consistent with the source example**and do not add or delete `catlass/...` headers by guess.

## Common header files (optional, not all copies)

```cpp
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/arch.hpp"
```

Add `matmul_epilogue.hpp`, `epilogue/block/...`, etc. to Epilogue (see `catlass-epilogue-composition`).

## Core type

### GemmShape

```cpp
using L1TileShape = GemmShape<M, N, K>;
using L0TileShape = GemmShape<M, N, K>;
```

- `M/N/K` usually requires a multiple of**16**
- Common habits: `L0.M == L1.M`, `L0.N == L1.N`, `L0.K == L1.K / 4`

### GemmType

```cpp
using AType = Gemm::GemmType<ElementA, LayoutA>;
```

The element type determines the handling and MMAD behaviour in conjunction with `Layout` (`RowMajor` / `ColumnMajor`, etc.).

### DispatchPolicy

| Policy | Characteristics | Application |
|------|------|------|
| `MmadAtlasA2Pingpong<unitFlag>` | L1 double buffering | Universal baseline |
| `MmadAtlasA2Preload<unitFlag, shuffleK>` | Prefetch + shuffleK | Big Shape, bandwidth sensitive |
| TLA Pingpong | TLA Model | Match with TLA Block |

### Layout and Swizzle Direction (Experience)

- `RowMajor + ColumnMajor`: Common, regular with `L1TileShape<128,256,256>`
- Double `RowMajor`: `M>N` may be different from Swizzle orientation when `M<N` is used (see matmul tone skill)

## Device, call the skeleton.

```cpp
Matmul matmulOp;
typename MatmulKernel::Arguments args{problemShape, deviceA, deviceB, deviceC};
matmulOp.CanImplement(args);
size_t wsSize = matmulOp.GetWorkspaceSize(args);
matmulOp.Initialize(args, deviceWorkspace);
matmulOp(stream, aicCoreNum);
```

kernel (e. g. Split-K) requiring workspace must be distributed by `GetWorkspaceSize` on the host side. See the workspace rule in the migration regulation.

## AR Environment Alarm

- Compile `ASCEND_HOME_PATH`; `CATLASS_ROOT` from `task.yaml catlass.root` or environment variable; `copytree` entire `catlass_op/` folders at Verify, cake pass `-DCATLASS_ROOT=`, `-DNPU_ARCH=`, `-DCATLASS_ARCH=`
- AR process with**triton_ascend**Path: `profiler_npu` → `op_statistic.csv` → `generation_profile_result.json` (non msprof CLI)
- `.asc` is changed and each round of eval will be in the verify directory**recake & make**based on static assertions and link results
