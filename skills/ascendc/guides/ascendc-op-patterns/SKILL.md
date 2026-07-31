---
name: ascendc-op-patterns
description: "AscendC's common operator family implementation template: elementwise, Broadcast, reduction, softmax-like, index/gather, and matmul emilogue. Use to quickly select the initial Kernel structure while running."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "elementwise,broadcast,reduction,indexed,matmul-like,fused"
---

# AscendC Common operator Template

Use this skill to achieve the skeleton before changing the mathematical logic of Kernel. It is recommended to get the correct direct-invoke kernel and then optimize the performance with the profiling document.

## 1. Template Selection Table

| operator form | Initial skeleton | Main risks |
|---|---|---|
| unary/binary elementwise | flatten + vector tile | Dype Cast and tail mask |
| broadcast elementwise | Linear output index map to input | Integer index costs |
| row reduction | One or more rows per nuclear process | accuracy |
| large reduction | Sub-contracting + Phase II merger | Workspace or atomic cost |
| softmax/logsumexp | max pass + exp/sum pass + normalize | Spill and workspace |
| gather/scatter/index | Contigous Quick Path + General Path | Crossing borders and writing conflicts |
| matmul + epilogue | Keep Cube main path and try to integrate UB/L0C epilogue | Cube/Vector load balance |

## 2. Elementwise Bones

Gradually equalize the output of a contigouous to one-dimensional elements:

```text
for each core:
  CopyIn each input tile
  Compute vector expression
  CopyOut output tile
  handle tail
```

Rules:

- tile length meets both vector block size and 32B moving particle sizes.
- Preprocessing of scalar parameters is placed in host tilling.
- Output dtype does not calculate dtype at the same time, but only at the end.
- The `where`-type operator predicate is generated in UB to avoid intermediate results writing back to GM.

## 3. Broadcast skeleton

Break speed path for common broadcast:

- I'm sorry, I'm sorry, I'm sorry, I'm sorry.
- trailing-dim broadcast: transformation along the last dimension of vector.
- Scalar input: Each file loads only once.
- General Broadcast: Use indexMapping fallback.

Don't let all Shape go general index happening; it usually becomes scalar-born.

## 4. Reduction skeleton

Define before encoding:

```text
outer  = product(dims before reduced axis)
reduce = product(reduced dims)
inner  = product(dims after reduced axis)
```

Select order:

1. `inner == 1` gives priority to consecutive block returns.
2. When `reduce` is smaller, a core handles one or more rows.
3. When `reduce` is large, break into multiple core sub-contracts and merge them with workspace or atomic.
4. fp16/bf16 for sensitive fate using fp32 accumulator as required by reference accuracy.

tail rule:

- The last description file must be valid.
- max/min uses correct effect value.
- sum/prod visible initialization, without UB default content.

## 5. Softmax-Like skeleton

Numerical stabilization path:

```text
row_max = reduce_max(x)
tmp = exp(x - row_max)
row_sum = reduce_sum(tmp)
y = tmp / row_sum
```

Rules:

- When rows can be placed in UB, first preserve the intermediate result in UB.
- Long row uses more pass tilling and workspace.
- epsilon can only be introduced when the syntax allows.
- The output Shape and dtype must match reference implementation.

## 6. Indexed operator

For gather, scatter, nonzero, index-put, etc.:

- Verify index dtype and bases when feasible on the host side.
- Separates the contigous fast path from the generic path.
- Writing a conflict must be defined in a visible way: semantics such as tomic/add/last-write cannot be confused.
- Do not rephrase the silence of indexed semantics into dense elementwise.

## 7. Matmul Epilogue

If the mission is a matmul-like plus bias, action, scale:

- Keep Cube Tiling on the right main path first.
- Simple epilogue try to merge before writing back to GM.
- fp32 accumulation is used for semantic reference requirements.
- Quant/dequant scale 's playout writing tilling data, without any hidden assumptions.

## 8. Select the fast path from the sample form

Multishape operator does not select a generic skeleton by operator only. Enter the sample into a small semantic mode, then break down the fast path for a high frequency or high-time mode, and eventually keep the complete gerneric fallback.

Common bins:

| Drums | Conditions for determination | Bones recommended. |
|---|---|---|
| same-shape contiguous elementwise | All input sape is the same and contigouous | flatten + bulk DataCopy + vector tile |
| scalar broadcast | Some input numel=1 | Only one scalar or host scalar for each file |
| last-dim broadcast | `(outer, D)` and `(D,)` or golf | Line-by-line processing, reusing Broadcast input |
| small-row reduction |   Reduce dim small, row many   | Multi-line authorization, scalar Statute if necessary |
| single-tile row reduction | One line completes the UB | CopyIn, UB complete all passs at once |
| large-row reduction | Single Line Over UB | Subtile + Phase 2 Merge or Workspace |
| contiguous indexed segment | index forms a continuous segment | DataCopy continuous block to avoid element by element GetValue |
| identity / single segment | Special semantics for scatter/segment review | Direct copy, sum or localaccumulation |
| special value | All-zero, all-NaN, constant input | Semantic Fast Path Filling or Skip Calculator |

host side recommends the generation of mode fields, and the Kernel side does only light branching:

```cpp
enum class PatternMode : int32_t {
  kGeneric = 0,
  kSameShape = 1,
  kLastDimBroadcast = 2,
  kSmallReduction = 3,
  kSingleTileReduction = 4,
};

PatternMode Classify(const at::Tensor& x, const at::Tensor& y, int64_t axis) {
  if (x.sizes() == y.sizes() && x.is_contiguous() && y.is_contiguous()) {
    return PatternMode::kSameShape;
  }
  if (x.dim() >= 2 && y.dim() == 1 &&
      y.size(0) == x.size(x.dim() - 1)) {
    return PatternMode::kLastDimBroadcast;
  }
  if (axis == x.dim() - 1 && x.size(axis) <= 32) {
    return PatternMode::kSmallReduction;
  }
  return PatternMode::kGeneric;
}
```

The device end does not drag all modes into a complex inner circle. The mode branch should determine, as far as possible, at the `Process` outer level:

```cpp
void Process() {
  if (mode_ == kSameShape) {
    ProcessSameShape();
  } else if (mode_ == kLastDimBroadcast) {
    ProcessLastDimBroadcast();
  } else {
    ProcessGeneric();
  }
}
```

## 9. Family-level realization details

### 9.1 Elementwise / Activation

- Removes constant `Adds(x, 0)` and unnecessary fp32 round-trip.
- Reference onlyaccuracyRaised on requestaccuracy;fp16/bf16 nativePath to be independently verifiederror.
- Exp, log, tanh, rsqrt link prioritizes the intermediate integration phase, avoiding a calc buffer per step.
- All-zero, all-NaN, constant input can have semantic speed paths, but the dtype overlay of normal input cannot be changed.

### 9.2 Broadcast Elementwise

- Same-shape, scalar, last-dim broadcast must precede the general index happening.
- `div/mod` in general Broadcast shall use the stide and increments in the host side axis, kernel, as far as possible.
- If the broadcast input is small, you can load the entire section to the UB and repeat it in a file.

### 9.3 Reduction / Softmax-like

- small D: Multi-line approvals, or scalar reduction synchronisation.
- Mediam D: A movement of UB, max/sum/normalize stringed in UB.
- large D: Split tile cumulative, save partial, then combine in two stages.
- Softmax/logsumexp to maintain numerical stability: `max -> exp(x-max) -> sum -> normalize`, do not destroy spill protection for less pass.

### 9.4 Indexed / Geometry

- Priority is given to identifying continuous periods, dim0/dim1 common axes, intensity reduce, single security.
- Change `GetValue/SetValue` to UB + Continuous DataCopy, which is the primary optimisation direction for operator in the Gate/scatter/resize class.
- Geometrics of operator are projected at the beginning of the bat, with only incremental advancements in the inner layer.
