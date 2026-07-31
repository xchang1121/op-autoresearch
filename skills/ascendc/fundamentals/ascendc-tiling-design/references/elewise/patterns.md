# EleWise-type operator scene route

> This document is used for**scene determination**and**policy selection**. Once the scene is determined, you enter the corresponding detailed document by link.

---

## scene determination process

EleWise (Elementwise): The input output Shape is the same, it is calculated on an element-by-element basis, without cross-element dependency. It does not distinguish between one dollar and two dollars, either Sin, Cos, Abs, Add, Mul, etc.

```
Organisation: N Individual Inputs shape + M Output shape

Step 1 — Shape Decision:
  All Input Output shape Exactly the same?
    ├─ YES → EleWise..and Ping is... dim0,1D Linear processing → Enter Step 2
    └─ NO  → Broadcast → [../broadcast/patterns.md]

Step 2 — dtype × (Decisions) Compute Path):
  Operating as Add/Sub / A cumulative link, mostly plus or minus AND dtype ∈ {FP16, BF16} AND spec Not stated"Enter the same level"?
    ├─ YES → High-accuracy branch: cast → FP32 compute → cast, with an extra FP32 intermediate buffer
    └─ NO  → Original dtype Direct Calculating Branch
  Two branches. UB Budget and ubFormer I'll see you at the formula. [tiling.md](tiling.md)
  Multiplication/Other scenarios, such as division, are not yet covered. dtype Direct branch)
```

**Step 2 policy statement**: the problem of half-accuracy "Most Eat Decimals" needs to be raised to accuracy circumvent, and K values and Cast writings are details of API realization. Tiling-design is responsible for macro slices (branch determination, UB budget, ubFormer), and API realization details are determined by the fall of the Compute phase.

---

## General rules

- **Multiple check**: multiple of elements aligned to 512 (ELEM_ALIGN_FACTOR), at least 4KB data per nuclei
- **UB Alignment**: Align with 256B to ensure the effectiveness of the Vector Command
- **Distinguishing head/tail block**: tail block data may be smaller than the first block, cycle number and tail size

The constant definition and formulae are specified in [tiling.md] (tiling.md).

---

## Cross-Scene Reference

| Theme | Documentation |
|------|------|
| EleWise Tiling Detailed Calculation (continent, formulae, templates, dtype branch UB budget) | [tiling.md](tiling.md) |
| Broadcast scene route (input Shape at different times) | [../broadcast/patterns.md](../broadcast/patterns.md) |
