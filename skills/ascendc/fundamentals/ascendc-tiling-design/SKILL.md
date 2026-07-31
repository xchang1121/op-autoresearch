---
name: ascendc-tiling-design
description: Design guide for Ascend C operator Tiling. Covers the scene route, algorithm selection, UB split formula, tilling field engagement for the three types of operator Reducation / Elementwise / Broadcast. Trigger: operator design phase, planned multi-nuclei / UB cut, Buffer allocation, and a review of operator's Tiling methodology.
---

# Ascend C operator Tiling

## operator Clan scene route

Each family reads the `patterns.md` first for the scene determination and then, by conclusion, enters the specific algorithm document.

| Group | Typical operator | Scene Path |
|---|---|---|
| **Reduction** | ReduceSum, Softmax, LayerNorm, ArgMax, RMSNorm | [reduction/patterns.md](references/reduction/patterns.md) |
| **Elementwise** | Sin, Cos, Abs, Exp, sigmoid, mish, gelu | [elewise/patterns.md](references/elewise/patterns.md) → [elewise/tiling.md](references/elewise/tiling.md) |
| **Broadcast** | Add, Mul, Sub includes broadcast syntax | [broadcast/patterns.md](references/broadcast/patterns.md) |

## Reduction Library

| Algebra | Application | Details |
|---|---|---|
| **FullLoad (direct)** | Data block at UB, CopyIn once | (Inline in patterns.md) |
| **TwoPass** | FullLoad next two var/std/layer_norms | (ibid.) |
| **Welford Online** | Split, two related statistics in one-way flow | [alg-welford.md](references/reduction/alg-welford.md) |
| **Welford + Group(8)** | error when it's big | (Same as alg-welford.md) |
| **Group Reduce** | Cross-core return. | [alg-group-reduce.md](references/reduction/alg-group-reduce.md) |
| **Partly cumulative / Dichotomy** | Sum accuracy is sensitive. | [alg-dichotomy.md](references/reduction/alg-dichotomy.md) |
| **Indexed return (Agg Max/Min)** | Return + Index Tracking | [with-index.md](references/reduction/with-index.md) |

AR / ARA mode (ordered after axis):

| Mode | Trigger | Sub Mode |
|---|---|---|
| AR-FullLoad | A0 = 1, UB installs the next line | [ar-fullload.md](references/reduction/ar-fullload.md) |
| AR-ColSplit | A0 = 1, UB cannot load the whole line | [ar-colsplit.md](references/reduction/ar-colsplit.md) |
| ARA-FullLoad | A0 > 1, UB can load R × fileA0 | [ara-fullload.md](references/reduction/ara-fullload.md) |
| ARA-RowSplit | A0>1, UB can't load | [ara-rowsplit.md](references/reduction/ara-rowsplit.md) |

Other supporting documents: [algorithms.md] (references/reduction/algorithms.md) (routing matrix), [multi-axis-transform.md] (references/reduction/multi-axis-transform.md) (multi-axis returned Shape three-step variation), [multi-output-buffer.md] (references/reduction/multi-output-buffer.md) (multi-output Buffer planning), [tiling-fields.md] (references/reduction/tiling-fields.md) (tilling structure field agreement).

## Broadcast Subprogram

| Programme | Application | Details |
|---|---|---|
| One-dimensional broadcasting. | Shape, easy and smooth. | [onedim.md](references/broadcast/onedim.md) |
| UB Internal Broadcasting (PadBC) | Radio tensor small, UB can load | [ub-broadcast.md](references/broadcast/ub-broadcast.md) |
| Dynamic UB Broadcasting | Radio. Sharpe runtime confirmed. | [dynamic-ub-broadcast.md](references/broadcast/dynamic-ub-broadcast.md) |
| NDDMA Broadcasting | Big Shape, go-hardware to move the radio. | [nddma-broadcast.md](references/broadcast/nddma-broadcast.md) |

## Generic Tiling Design Elements

Any operator must consider:

1. **Polynuclear cut**: load balance, data locality, particle size (≥ 4KB per core)
2. **UB cut**: UB capacity limit (910B/B3 = 192KB, 910B4 = 128KB, 950 = 248KB), single process amount, split chunk formula
3. **Buffer Planning**: input/output Queue, CalcBuf Temporary, DoubleBuffer × BUFER_NUM
4. **Branch coverage**: dtype (FP32/FP16/BF16/INT8), Shape size, alignment, boundary values

Detailed UB byte book is [[ascendc-ub-budget]], and the LocalTensor subview rules on UB are [[ascendc-localtensor-subviews].
