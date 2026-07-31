---
name: ascendc-batch-playbook
description: "AscendC 24-op batch development/scrambling playbook: operator classification, template reuse, failure layer, priority, log interpretation and one assumption per round."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC Batch Development and Batch Playbook

This skill is used when the round requires multiple AscendC operators at the same time. The goal is to reduce repetition errors so that the conclusions of different ops are comparable and reusable.

## 1. Single operator archive

Each op first record the following information and then start changing the code:

```text
op name:
family: elementwise | broadcast | reduction | indexed | matmul-like | fused
input ranks:
output shape rule:
dtypes:
special cases:
expected tolerance:
initial skeleton:
first failing shape:
```

`initial skeleton` selects from `ascendc-op-patterns`. Do not write Kernel without classification.

## 2. Batch Priority

Recommended advance order:

1. Let all the operators configure, build, load.
2. Make simple 32B alignment Shape correct.
3. Let tail, unmatched, unincorporated Shape correct.
4. Make dtype the variant right.
5. Optimizing only the operator that has passed through correctness verification.
6. Do the slowest and correct operator profling-driving turning.

There is still an unexplained accuracy error in Kernel not entering performance optimization.

## 3. Frustrated bins

| Type of failure | Priority Skill |
|---|---|
| CMake/configure/build failed | `ascendc-direct-invoke`,`ascendc-crash-debug` |
| `.so` missing or op namespace error | `ascendc-direct-invoke` |
| timeout,hang,aic error | `ascendc-crash-debug` |
| Output All 0, Random Value, Err_cnt | `ascendc-precision-debug` |
| Only tail Shape failed | `ascendc-hardware-tiling` |
| The results were correct, but the performance was slow. | `ascendc-profiling-optimization` |
| Initial Kernel Structure Uncertain | `ascendc-op-patterns` |

## 4. Batch Not Variable

- `kernel.py` remains stable in grapper form: lazy load `.so`, calling `torch.ops.npu.<op>`.
- The name of CMake target, the name of the `.so` file, the name of the registered op are consistent.
- Most tiling formulas are synchronized with kernel tiling scripts.
- bueld/load fixes are submitted separately from mathematical logic fixes or on a rotational basis.
- Only one activity saving wave per round.
- Not silently shrink dtype, playout or Shape over.

## 5. When to use abstractally

At least two operators have proven sharing the same mode before abstracting the public template:

- elementwise flatten skeleton
- broadcast fast-path skeleton
- row-reduction skeleton
- two-stage reduction skeleton
- softmax-like row skeleton
- indexed fallback skeleton

Do not introduce universal helper in advance until ABI, Tiling fields and output rules are stabilized.

## 6. Minimum report per round

For each round of records:

```text
hypothesis:
changed files:
tested shape(s):
result:
next action:
```

If the same error still exists after two blind changes, stop changing the code and revert to logs, tilling values, DumpTensor or the smallest share.
