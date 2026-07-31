---
name: ascendc-perf-optimize
description: Ascend C operator performance optimizes the knowledge base. Based on profiling/ simulations, type of base (VEC / access / Scalar / no base), category loads the corresponding optimization policy; spatial analysis with parameters. Trigger: operator performance optimization, current analysis, sound diagnosis, tiling parameters.
---

# Ascend C operator Performance Optimizing Knowledge

## Bound type and optimisation strategy

Press the bottleneck type route for profiling/ simulation:

| Bound | Conditions for determination | Documentation |
|---|---|---|
| **VEC bound** | The Veter unit is highly utilized and vector commands are time-consuming | [vec.md](references/single-core-pipeline/vec.md) |
| **visited** | MTE2/AIC bandwidth close to peak, Vector, etc. | [memory.md](references/single-core-pipeline/memory.md) |
| **Scalar bound** | scalar time-consuming, controlled stream-intensive, or very small Shape | [scalar.md](references/single-core-pipeline/scalar.md) |
| **No found** | The utilization of the various units is low and further oil extraction is required | [no-bound.md](references/single-core-pipeline/no-bound.md) |

Each document covers: determination of conditions, severity classification, simulation track features, specific strategies (integration instructions, DoubleBuffer, Cast Optimization, L2 reuse, etc.), and Tiling amendments.

If a single base is not visible, or only a few samples are very slow, read the "Sampback and narrow path" and "Refrequently reusable optimisation paradigm" of `ascendc-profiling-optimization` first. Such questions do not normally continue to adjust a numerical parameter, but rather require split paths according to dtype, rank, Broadcast, reduce axis, index continuity or special value models.

## Tiling Parameter Spatial Analysis

Design to optimize search space to read: [parameter-analysis.md] (references/tiling/parameter-analysis.md) - a four-stage approach (the full set of kernel parameters) → Type/Constraint retroactive → Algorithms enable the determination of →'s candidate space).
