---
name: catlass-epilogue-composition
description: "CATLASS Aftercare: MatmulEpilogue, EVG TreVisitor, UB workspace; Add/Bias/ReLU integration and corresponding Kernel selection. This applies to matmul+Element-by-Element integration operator."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc_catlass
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "matmul, fused"
---

# CATLASS tailing (Epilogue)

`D = matmul (A, B) (+/-) Element-by-Element ` needs to be mounted on Gemm**BlockEpilogue**or**EVG**instead of `F.relu(matmul(...))` in `kernel.py` (which deviates from Catlass optimization and may trigger static inspections).

## Path Contrast

| Path | Kernel | Annotations |
|------|--------|------|
| No tailings | `BasicMatmul<..., void, Scheduler>` | GEMM pure |
| Standard Epilogue | `MatmulEpilogue` | MMAD+Element by Element within AIC |
| EVG + GM | `BasicMatmulTlaVisitor` | MMAD results are integrated through workspace, AIV |
| EVG + UB | `BasicMatmulTlaUbVisitor` | UB, reduce GM round-trip |

## Standard Epilogue (example Add)

Example of additional header file:

```cpp
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
```

Elements of assembly:

```cpp
using BlockMmad = /* The same BasicMatmul */;

using XType = Gemm::GemmType<half, layout::RowMajor>;  // Consistency with semantics of integration
using DType = CType;
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseOneSource;
constexpr uint32_t computeLength = 16384;  // Press tile Adjustments to structure
using TileElemWise = Epilogue::Tile::TileElemWiseAdd<ArchTag, CType, computeLength>;
using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, CType, XType, DType, TileElemWise, EpilogueTileCopy>;

using MatmulKernel = Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
```

`MatmulEpilogue` parameter semantics contain**bias/ add-up matrix X**and output D; aligned to `03_matmul_add` type example.

## Customised Element by Element (e. g. Add+ReLU)

`TileElemWise*`:

- **Custom Tile Epilogue**header file with `.asc` to `BlockEpilogue`
- AR Tasks that do not include the header file in `editable_files` can only be selected in the existing Tile group

## Summary of EVG (TreeVisitor)

Fits to a more complex integration map (multi-input, multi-operator chain):

```cpp
#include "catlass/gemm/kernel/basic_matmul_tla_visitor.hpp"
#include "catlass/epilogue/block/block_epilogue_visitor.hpp"
#include "catlass/epilogue/fusion/tree_visitor.hpp"
```

A combination of `VisitorAccLoad`, `VisitorAuxLoad`, `VisitorCompute<Op>`, `VisitorAuxStore`, etc. with `TreeVisitor`; `Arguments` requires additional `EVG::Arguments`.

UB version replaces `EpilogueVisitor<false>` only with `<true>`, Kernel with `BasicMatmulTlaUbVisitor`.

## Integration model quick check.

| Objective | Component orientation |
|------|----------|
| D = C + X | `TileElemWiseAdd` / `VisitorCompute<Add>` |
| GEMM only | `BlockEpilogue = void` |
| ReLU / GELU / SiLU | EVG `VisitorCompute<Relu>` etc. |
| GEMM with bias | Some templates add Bias type parameters to `BlockMmad` |

## Kernel Selection

| Requirements | Kernel |
|------|--------|
| No integration | `BasicMatmul` |
| Simple by element, AIC completed | `MatmulEpilogue` |
| Complex integration, AIV implementation | `BasicMatmulTlaVisitor` |
| Complex integration + small GM round-trip | `BasicMatmulTlaUbVisitor` |

## AR Alignment with Python

- `Model.forward` for `reference.py` defines semantics (e. g. `relu(A@B+X)`)
- `ModelNew` of `kernel.py` should call `torch.ops.catlass.*` of the same word**
- Integration logic should be achieved on**catlass**(`.asc`/ customize epilogue header / `catlass_torch.cpp`); do not recalculate the catlas results with a pure torch in Python
