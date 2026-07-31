---
name: pypto-pitfalls
description: "PyPTO common first-time generation error and correct writing"
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "all"
---

# PyPTO Common Trap

## 0. kernel output: return comment is prohibited, exit parameters (hard constraint, highest priority)

PyPTO Kernel**is not a normal Python function**: the output tensor must be imported as a kernel and written in kernel**.**The return of `-> ...` is forbidden, the output of `return` is forbidden, or the error is reported directly at the resolution stage:

- `NotImplementedError: Return annotation is not allowed` — Write `def kernel(...) -> pypto.Tensor(...):`
- `ParserError: Return statements are not allowed` — `return out` in Kernel

Correct: The output is also included, and the three ways to write are chosen.

```python
@pypto.frontend.jit(...)
def kernel(
    x: pypto.Tensor((flat_size,), pypto.DT_FP32),
    out: pypto.Tensor((flat_size,), pypto.DT_FP32),   # OutputtensorIt's also stylish. Don't. return
):
    pypto.set_vec_tile_shapes(8192)
    out[:] = pypto.exp(x)                              # 1) Entire Overwrite
    # 2) Loop segment by offset: pypto.assemble (chunk_result, [off], out)
    # 3) Visible in-place submission: out.move(src) (common end after loop+assemble)
```

## 1. Operator rule (highest frequency error)

`+` `*`: scalar at any position. `-` `/`:tensor must be left. Function calls: First parameter must Tensor.

```python
1.0 + x            # OK(__radd__)
1.0 - x            # CRASH(__rsub__ Not achieved)
1.0 / x            # CRASH(__rtruediv__ Not achieved)
pypto.add(1.0, x)  # CRASH(Functions call)scalarBefore)

# 1 - x Write correctly
x * (-1.0) + 1.0   # Recommendations

# Python between scalar
neg_delta = -delta  # OK(delta It's closed. float)
```

## 2. Clamp / min(x,d) Achieved

`pypto.clamp` is not available. `min(x, d)` uses double reverse: `-max(-x, -d)`. `pypto.minimum(x, 0.0)` is available.

```python
# min (abs_diff, delta) — delta is closed
neg_abs = pypto.mul(abs_diff, -1.0)
clipped = pypto.mul(pypto.maximum(neg_abs, -delta), -1.0)  # = min(abs_diff, delta)
```

**Huber Los complete mode**(must use clamp, not simplified):
```python
diff = predictions - targets
abs_diff = pypto.abs(diff)
neg_abs = pypto.mul(abs_diff, -1.0)
clipped = pypto.mul(pypto.maximum(neg_abs, -delta), -1.0)  # min(|d|, delta)
half_sq = clipped * clipped * 0.5
loss = half_sq + abs_diff - clipped  # Full Huber Formula
total = pypto.sum(loss, dim=0, keepdim=True)
output[:] = total / flat_size
```

## 3. Plant Functions

scalar parameters (eps, slope, margin, etc.)**must**be passed into Kernel in a closed package as a plant function parameter.

When 3D+2D matmul, forward**spreads the dimension nm=N*M**to the plant, not separate N-M.

## 4. matmul K > 65535

Replaces it with an element-by-element multiplication + `pypto.sum(a * b_broadcast, dim=1)`. Forward, `B.reshape(1, -1)` makes it broadcastable.

## 5. Distance measurement must sqrt

`sum(diff*diff)` is**square distance**, not L2. Triplet Marginloss etc. must `pypto.sqrt(sum_sq + eps)`.

## 6. tile rank = tensor rank

The number of `set_vec_tile_shapes` parameters must be equal to the rank of the operated tensor. 2D tensor uses 2D file.

## 6.1 Blind copy tile constant (especially 16384)

The example of `16384`/`8192` is an empirical candidate, not a fixed answer. It must be recalculated according to the current shape and attribution.

- Common error: Enter `(128, 4096)` while writing `set_vec_tile_shapes(1, 16384)`.
- A more reasonable candidate: `set_vec_tile_shapes(4, 4096)` (the axis of return is not wasted and the bat is even higher).

Key points:
- Priority is given to avoiding the apparent "budget waste" of `tile[i] > shape[i]`.
- The example code can only borrow structure and cannot copy the shape/tile number.

## 6.2 Incorrect reading of "less segments" as "the larger the axis of return"

The "continuous removal at the threshold before reverting to the contract axis" is the secondary objective, but not "the larger the axis of return".

- Common error: Directly write `tile_hidden = hidden`, seeking one-time coverage of the axis of contract.
- Typical consequence: UB/OoOSchedule error (even if the semantics are correct).

Correct practice:
- Meet `prod(tile_shape) <= 16384` first with `auto_tiles <= 2048`.
- In the event of UB/OOOScheduule error, drop priority: `16384 -> 8192 -> 4096`.
- If `auto_tiles > 2048`, change priority to loop segment, do not push more radical files.
- First, the continuous movement reaches the approximate `1KB` (experimental threshold) and then a comparison of the attribute axial desserts in the candidate for compliance (often tried `16/32/64`).

## 6.3 Incorrect reading of "removal priority" as "as long as it is more continuous"

Continuous removal is a first step, not an infinity one. Once an efficient area is reached, continuing to give the budget to non-prescriptive axes usually yields little, and instead crowds out the axle file budget.

Correct practice:
- Start with an empirical threshold: `contiguous_bytes(tile) >= 1KB`.
- When the target is met, the remaining budget is used for the return axis dessert search, without the assumption of "larger and faster".
- If more than one candidate has met the target, preference is given to the measured dessert (e.g., the scene of `(1,16,256)` is better than `(1,64,256)`).
- Do not move the mid-stage start-up rule of the loop directly to the file; the proven preferred value should be preferred for the known fixed shape.

## 6.4 `shape` as a continuous removal length (HF miscalculation)

The continuous removal threshold must be calculated on the actual continuous period of**tie**and not on input `shape`.

Example of error:
- When `shape=(16,256,256), dim=1, tile=(1,256,64)` is written, "Consequent dimensions are 256, thus reaching 1KB".

Correct calculation:
- Vec Common estimate: `contiguous_tile_elems = tile[last_axis]`.
- The last instance of continuous removal shall be calculated as `tile[2]=64`: `64*4=256B` (FP32), which does not amount to 1KB.
- If the 1KB threshold is to be met first, the continuous dimensions of the file should be at least 256 (FP32) before comparing the subparagraphs of the Statute.

## 6.5 Combination of cube/vec operator with only one file (high frequency breakdown during compilation)

HF misuse:
- First `set_cube_tile_shapes(...)` for `matmul`, then directly for `add/mul/expand_clone`.
- Mistakes that cube file will automatically overwhelm vec operator.

Typical error reporting:
- `ASSERTION FAILED: vecTile.valid()`
- `op [ADD] tile shape not set`

Correct practice:
- Dismantling the kernel phase: first the cume phase (matmul) and then the vec phase (elementwise/broadcast).
- `pypto.set_vec_tile_shapes(...)` before entering the vec phase.
- Linear layer `y = x @ w + b` recommended: `b` first `reshape(1, -1)`, kernel broadcast with `expand_clone` and then `add` in Forward.

## 7. Prefer Internal Functions

`pypto.sigmoid`, `pypto.softmax`, `pypto.abs`, `pypto.exp`, `pypto.log`, `pypto.sqrt` etc. have built-in direct use and are prohibited from manual realization.

## 8. Not required assert flat_size %file_size=0

Auto-tile automatically handles the remaining number. Only the loop+view mode needs to ensure partial division.

## 9. Two paragraphs of per-sample loss

2DEnteredper-sample loss:Phase 1 `(4, 4096)`Fine.per-sample → Phase 2 `(128, 1)`CrossbatchReturn.

## 10. Conditional branch with maximm + minim

`pypto.where` not available.**`pypto.minimum` available!**

```python
# ELU
output[:] = pypto.maximum(x, 0.0) + (pypto.exp(pypto.minimum(x, 0.0)) - 1.0) * alpha
# LeakyReLU
output[:] = pypto.maximum(x, 0.0) + pypto.minimum(x, 0.0) * slope
```

`maximum(x, f(x))` is not allowed to make a condition selection (the result is wrong on the positive half-axis f(x)>x).

## 11. Square formula

`var = sq_sum * inv_count - mean * mean` (E [X²]-E[x]²). The symbol reverses NAN.

## 12. ModelNew._init__ Signature

Must be consistent with the original Model. Shape fetchs in forward, not added to the `__init__` parameter.

## 12.1 Static missions are written as "All-powerful Kernel" (multi-dim branch)

HF misuse:
- Write `if dim == 0/1/2` in a Kernel and try to cover all the L. A. at once.
- In order to reuse the branch, the `keepdim=True` goes back to the Forward `squeeze`.

Why not recommend:
- The single task of benchmark `shape/dim` comes from fixed `get_inputs/get_init_inputs` and is essentially a static contract.
- Multiple branches introduce search space unrelated to the subject and increase the probability of error (tile is also easier to write as irrelevant).
- `Example, change to desired ...` such a comment is a description of the repository, not a current running requirement; the extension of multiple dims accordingly belongs to the proposed thematic text.

Correct practice:
- Only fixed parameter paths (e.g. `dim=1`) are achieved for the current task.
- The output semantics are aligned directly to the baseline (if `keepdim` is retained by the baseline) and do not make a " reset before " bypass.
- When fixing `dim`, do not transmit `dim` as a kernel runtime parameter; use the fixed constant `dim = `fixed value' directly in kernel.

## 13. Module name `pypto`

Not `pyto`, `pytorch`, `pto`. All transfers to `pypto.` start.

## 14. baseline semantic error (HF and hidden)

The most dangerous error was not grammar, but a "syntax contract" error: the code could be compiled, or even PASS, but not the same thing.

High-haired error source:
- Use the variable name as semantic (e.g. `input/target/predictions`).
- API names are used as mathematical directions (especially asymmetric targets).
- Seeing that verify is correct by default (the tolerance gap may hide the error in the direction).

Error prevention process (first semantic, later achieved):
1. Write mathematical styles from baseline `forward`.
2. Separately check API contracts: Parameter Meaning, Login Entry, Statute Definition.
3. Quest the semantics of the Statute: `sum`, `batchmean`, or `mean` syntax = `sum/count`, and specify the axes and output of shape.
4. Whether the mark is asymmetrical; the asymmetrically specified "reference entry" and "comparison entry".
5. Use 1 group to explain the sample for semantic self-examination (prior asymmetric distribution).

Example (KLDiv, only examples):
- `F.kl_div(torch.log(pred), target, reduction='batchmean')`
- If it is written as `pred * (log(pred) - log(target))` by template, the direction is the reverse.

## 15. Non-consolidation of successive axes of statutes, resulting in intermediate costs of multiple returns

When the objective of the Statute is essentially a "continuous multiaxis joint statute", the direct chain `sum(dim=...)` regular sessions produce the intermediate tensor and additional movements.

Candidatures:
- Forward consolidates successive axes of statutes (e.g. `H,W -> HW`, or `(B,H) -> (B*H)`).
- Kernel completes the main contract with a single statute and then makes the final semantic statute (e.g. `batchmean` except `B`).

HF error zone:
- The adoption of the "per-sample two-stage contract" as a default template has resulted in two paragraphs of a continuing statute that could have been merged.

The correct determination:
- Only if the intermediate result has to be reused by another operator/output will two stages be retained.
- If the intermediate result is used only for the continuation of the Statute, priority is given to the consolidation of successive axes of the Statute.
