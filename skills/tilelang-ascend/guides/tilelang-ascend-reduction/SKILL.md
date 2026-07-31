---
name: tilelang-ascend-reduction
description: "TileLang Ascend is a contract-class operator code guide, which is consulted when generating operator, the equivalent of which contains the dimension of attribution."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduction"
---

# TileLang Ascend Conscript operator Encoding Guide

---

## Decision tree: Consumes operator path

**Important**: `T.reduce_sum/max/min` and `T.tile.*` can be used in both Devloper and Express modes**. Model selection depends on whether manual memory levels and synchronisation are required, rather than which API is used.

```
Inclusion in the Convention (incl.reduce_sum / reduce_max / reduce_min)
    The dimensions of restitution must be correctly determined: the first axis of the contract (recession); the last axis of the contract (recession); all axes of the contract; and the jumper's return
    Select the right API: T.reduce_sum / T.reduce_max / T.reduce_min / T.atomic_add
    Memory: T.alloc_shared → UB
```

