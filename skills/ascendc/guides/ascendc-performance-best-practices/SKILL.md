---
name: ascendc-performance-best-practices
description: Ascend Coperator performance optimizes the best practice library. Design documents (softmax / elementwise /broadcast / scalar / common) according to the operator organization are used to query reference implementation during the implementation phase. Trigger: a reference code/ design template is required for an optimisation.
---

# Ascend C operator performance optimized best practice

Optimize knowledge by**operator (operator family)**.

## operator family design documents

| operator | Typical operator | Design Document |
|---|---|---|
| **Reduction / Softmax** | Softmax, log_softmax, FlashAttention embedded | [online_softmax_design.md](reference/softmax/online_softmax_design.md) (FlashAttention-style running max+sum), [state_resident_design.md](reference/softmax/state_resident_design.md) |
| **Elementwise** | Sin, Cos, Abs, Exp, sigmoid, tanh | [double_buffer_design.md](reference/elementwise/double_buffer_design.md), [vector_efficiency_design.md](reference/elementwise/vector_efficiency_design.md) |
| **Broadcast** | Add, Mul, Sub includes broadcast syntax | [broadcast_mask_design.md](reference/broadcast/broadcast_mask_design.md) |
| **Scalar Code** | ScalarBund class (control flow/ index intensity) | [guide.md](reference/scalar/guide.md), [coding_principles.md](reference/scalar/coding_principles.md) |

## Cross the operator generic mode

| Optimization Type | Apply scene | Documentation |
|---|---|---|
| **End block processing** | The amount of data cannot be divided by file size | [tail_block_design.md](reference/common/tail_block_design.md) |
| **DataCopy Optimization** | Multiple loads of non-matched, small quantities | [datacopy_optimization_design.md](reference/common/datacopy_optimization_design.md) |
| **UB / TBuf Resident & Bank Conflict Circumvention** | A large number of files/loads repeat the same data from GM | [ub_resident_design.md](reference/common/ub_resident_design.md) |

## Each design.md chapter specifies

- Optimization of targets (quantified benefits)
- Structure Overview (storage level / data stream / event sync)
- Key parameters (hostside calculations and fields)
- Core Calculator Cycle (removed and transposed)
- Key Change Point Table
- Optional: restraining / stepping pit / selecting decision-making
