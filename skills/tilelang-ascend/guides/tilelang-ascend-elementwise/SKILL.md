---
name: tilelang-ascend-elementwise
description: "The TileLang Ascend element by element operator coding guide. It covers the T. Parallel symbols API and T.tile.xxx expands the original language by two programming paradigms, broadcast mode, column cut. Reference is made to this guide when generating the element by element category operator."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise"
---

# TileLang Ascend Element by Element operator Encoding Guide

---

## Decision tree: element-wise operator path

**Important**: `T.tile.*` can be used in Devloper and Express**. Model selection depends on whether the memory hierarchy and synchronisation need to be manually controlled, rather than which API is used.

```
Pure element-wise(Element-by-Element)
├─ One-step calculation → Developer Mode
│   API: T.Parallel + Numerical Symbols
│   Memory: T.alloc_shared(compilerMap to UB)
│
└─ Multistep Operations
    ├─ Precise buffer Control → Expert Mode
    └─ There's no need for precision control. → Developer Mode
```

