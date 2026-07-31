---
name: pypto-optimization
description: "PyPTO performance optimization rules and reference sequences. This applies to scenarios that require the optimization of tile/loop/renunciation performance, comparison of tile scenarios, interpretation of the same operator for different tile performance differences (especially softmax/logsoftmax/reduction/norm/loss)"
category: method
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "softmax,logsoftmax,reduction,norm,loss,tile,performance"
---

# PyPTO performance optimization (continuous replenishment)

## Rule 1: Continuous removal first reaches the threshold and then finds dessert at the axis of return

The following priority is given to operator (softmax, logsoftmax, sum/amax/amin, and `mean=sum/count` semantics) containing the contract of return:
- For a fixed type `shape=(16,256,256), dim=1`, the default template is directly with `set_vec_tile_shapes(1, 16, 256)`.
- First meet hard constraints: `prod(tile_shape) <= 16384` and `auto_tiles <= 2048`.
- If `auto_tiles > 2048`, introduce the `loop + view/assemble` segment first, then fine-tune the file.
- If a UB or OoOSchedule-related error is compiled/verified, priority is given to the reduction of `prod(tile_shape)` (often from `16384 -> 8192 -> 4096`).
- **Empirical threshold for continuous movement**: `contiguous_bytes(tile) >= 1KB` (Empirical value, first performance threshold).
- Under the same compileable bounds, the candidate that did not reach 1KB will be phased out by default,**not**directly selected simply because of the "axis of the Statute".
- Only when the candidates can be compiled less than 1KB will the axes of the Statute be compared with the measurements in the `<1KB` candidate.
- When the threshold is reached, do not default on the "Axes of the Statute" to be bigger and faster; and make a candidate test for the axis of the Statute (the default order `16 -> 32 -> 64`) and select desserts accordingly.
- **tile is not wasted**: priority is given to each dimension `tile[i] <= shape[i]`. `tile[i]` is much greater than the corresponding dimension, which usually does not increase effective parallels, but wastes the resource budget and raises auto-tiling costs.
- **Prohibition of misreading**: It is not "the better the better" or "the less the axle of the contract." When the criteria are met, the objective is to reduce the number of subparagraphs of the Statute instead of continuing to zoom in the non-axis of the Statute.

Of which (key, prohibition of miscalculation):
- `contiguous_bytes(tile) = contiguous_tile_elems * dtype_bytes`
- `contiguous_tile_elems` means**the number of tile elements in a continuous moving segment**, not the original number of `shape` elements.
- The Vec scenario is by default estimated at the last dimension: `contiguous_tile_elems = tile[last_axis]` (without conversion/replacement).
- Prioritize `tile[last_axis] <= shape[last_axis]`; do not "convert " through `tile > shape` before a continuous removal threshold is determined.
- FP32 Common threshold: `contiguous_tile_elems >= 256` (approximately 1KB)
- FP16/BF16 Common threshold: `contiguous_tile_elems >= 512` (approximately 1KB)
- In retrospect: For `shape=(16,256,256), dim=1, tile=(1,256,64)`, consecutive handling is based on `tile[2]=64`, only `64*4=256B`,**not reaching 1KB**.

### Reason

- In the absence of continuous handling, access to debris and overstepping costs will be the first bottlenecks.
- Once a continuous move has reached a high-efficiency zone, the marginal returns are usually further amplified.
- When the axis of return is too big, the monotask may be too fat (local return tree is heavier and storage/flow pressure is higher).
- The return axle tile is over the hour and the segment and consolidation costs will rise.
- It is therefore common for non-uniform relationships (U-types) to find dessert in the candidate for compliance, rather than to be singled out.

### Example A: Softmax `(16, 16384), dim=1`

- `set_vec_tile_shapes(1, 8192)`: 2 paragraphs per row (priority)
- `set_vec_tile_shapes(2, 4096)`: 4 paragraphs per row
- `set_vec_tile_shapes(4, 4096)`: 4 paragraphs per line with more budget allocated to non-return axes

Empirically, `(1, 8192)` is usually better than `(2, 4096)` and `(4, 4096)`.

### Example B: Reduction `(16, 64, 256, 256), dim=1`

- `set_vec_tile_shapes(1, 16, 1, 256)` / `(1, 32, 1, 256)` / `(1, 64, 1, 256)`: Both satisfy continuous handling compliance and require a dessert comparison.
- `set_vec_tile_shapes(1, 1, 16, 256)`: Although the standards were continuously moved, the axis of the contract of return was not used effectively and was usually a poor candidate.

This example reflects a hierarchy of rules: successive removal leads to compliance, followed by a return-axis dessert search.

### Example C: TripletMarginlos Phase-1 `(128, 4096), dim=1`

- `set_vec_tile_shapes(4, 4096)`: Complete coverage of the attribute axis and higher parallelness of the bat axis (priority)
- `set_vec_tile_shapes(1, 16384)`: A second-dimensional file clearly exceeds real dimensions `4096`, has a file budget waste

Empirically, `(4, 4096)` is usually better than `(1, 16384)`.

### Example D: 3D Max/Reduction `(16, 256, 256), dim=1`

- Candidate `set_vec_tile_shapes(1, 256, 64)`:
  - The axis of the contract is not subdivided, but only `64*4=256B` (FP32) is moved continuously and does not meet the 1KB threshold.
- Candidates for compliance:
  - `set_vec_tile_shapes(1, 16, 256)`: Continuously moving the standard and returning to the axis segment 16 (this shape default priority).
  - `set_vec_tile_shapes(1, 32, 256)`: Continuously move the standard and assign the axial fraction 8 (optional).
  - `set_vec_tile_shapes(1, 64, 256)`: Continuously move the standard and assign the axial fraction 4 (not recommended for default).

This example shows that the continuous removal threshold check must be based on `tile`. For the fixed shape, the default direct use of `(1,16,256)` does not replace the empirical conclusion with a "middle start ".

### Example E: Refusal to "convert to meet " `(16, 256, 256), dim=1`

- Candidate `set_vec_tile_shapes(1, 32, 512)`:
  - It can be compiled, but `tile[2]=512 > shape[2]=256`, which is a budget waste candidate, should not be justified by "effictive = 256".
- Conclusions
  - Threshold judgement is based on the actual continuous period of `tile`; first selects `tile[last_axis] <= shape[last_axis]`'s candidacy, then makes `16/32/64`'s dessert comparison.

### Boundary application

- This is the performance priority rule, not the semantic rule, and ultimately the test.
- Exceptions may exist when the non-return axis is large and backend achieves speciality.
- The continuous removal threshold `1KB` is the starting point of experience and not a hard constraint; it can be fine-tuned by profile results.
- Threshold values judge the actual duration of `tile` without the full-dimensional length of `shape`.
- Benchmark for stationary Shape, avoids the "converted to the standard" for `tile[last_axis] > shape[last_axis]`.
- For two tiles, pipeline (e.g., the per-sample phase of CrossEntropy and the Batch reintegration phase), this rule is applied separately in stages.
- For RMSNornm/BatchNomm such 3D contracts, if the axis of the contract is `C` and `C` is medium (e.g. 32/64/128), priority is given to the last dimension of continuous movement up to 1KB and then to the `C` axis file comparison for dessert.
- `tile[i] > shape[i]` is not a grammatical error, but it is usually a performance signal, unless there is a clear and realistic measurement of the proceeds.
- Cannot and does not need to calculate UB occupancy manually; based on `prod(tile_shape)` grade down + compilation.

## Rule 2: `loop count` for mid-sweet, not extreme

The scenes that apply to the fixed total volume of work, which are achieved through sections along the outer axis of `loop + view/assemble` (matmul, norm, reduction, loss) are common.

- When the total is fixed, it usually meets: `total_batch = loop_count * main_batch`.
- `loop_count` is in essence a modulation of "task particle": the larger the `loop_count`, the smaller the single task; the smaller the `loop_count`, the larger the single task.
- **Empirical patterns are usually of the U-type**: the best performance with intermediate desserts; both ends are likely to slow down.

### Reason (general)

- `loop_count` is too big (taggered):
  - Increased number and dependency of teams and increased costs of ready-queue/mobilization/aware.
  - Common: `Wait Schedule Time`, `Wait Predecessor Time` significantly increased.
- `loop_count` is too small (overweight mission):
  - The single task internal submap is heavy and the single execution time is significantly higher.
  - The parallel particle size becomes thicker, the load balance changes, and the tail core slows the total time.

### Selective method (common reusable)

1. Brush candidate: `loop_count ∈ {8, 16, 32, 64, 128}` (or extension by scale of problem, etc.).
2. Record harmonized indicators under the same process:
   - `gen_time_us` (main target)
   - `avg Wait Schedule Time / core`
   - `avg Execute task num / core`
   - `sum_dur` for main time-consuming PSG and `avg_dur`
3. Decision logic:
   - If `loop_count` is increased by `gen_time_us` and the water indicator is increased sharply, then enters the Permuted Area.
   - If `loop_count` is reduced, `gen_time_us` rises and the main stage `avg_dur`/`sum_dur` lifts up, enters the "overweight".
4. The best advantage is to choose between the "over-crash" and "over-fat" areas, with minor fine-tuning.

### Common error zone

- Error A: "`loop_count` is smaller and faster".
- Error B: "`loop_count` as soon as possible (in parallel with more).
- Error C (Matmul HF): "`BASIC_BATCH=128` is the fixed best value".
- Correct: Scanning for dessert, without a monotonous assumption.

### Matmul Reminder (HF)

- For matmul for `m` dimensional loop, fixation of `BASIC_BATCH` constant (e.g. 128/256) is prohibited.
- Select the candidate in the space of `loop_count` and then reverse `BASIC_BATCH = ceil_div(m, loop_count)`.
- The default candidate order is recommended: `16/32 -> 8/64 -> 1/max` (stop early as required).
- Example:`m=16384` when,`loop=16/32`Correspond`BASIC_BATCH=1024/512`;`loop=8/64`Correspond`2048/256`.
- In the event of a conflict with any illustrative constant, this article prevails.

### Lazy Start (Recommends Default)

- When `loop_count` is feasible for a wider range (e.g. `1 ~ 128`), do not sweep from both ends in sequence.
- You can start directly with the mid-sweet candidate (often try first `16` or `32`).
- The word "intermediate" here is**between logarithmic**, not the midpoint of arithmetic:
  - The loop candidate usually changes by two steps (`1, 2, 4, ..., 128`).
  - At this level, the middle of `1~128` is given priority to `16/32`, which avoids both the "overweight" and "overbreak" ends.
- Quick process:
  1. Check the mid-point candidate (`16/32`).
  2. One more leap to each side (e.g. `8`, `64`).
  3. Only fills the endpoints (`1`, `128`) as needed.
- Purpose: First, avoid extremes at both ends quickly, give priority to the dessert area and then fine-tune it at a small scale.

### Non-uniform examples (evidence only, not fixed answers)

A survey of case35 (Groupnorm) shows that:
- `loop=16`: `630.30us` (More)
- `loop=8`: `649.58us`
- `loop=32`: `649.42us`
- `loop=64`: `804.96us`
- `loop=128`: `936.66us`

This example shows that `16` is better than `8/32`, and that `64/128` is clearly entering the "scattered area".

## Rule 3: Consolidation of successive axes of the Statute in preference (in semantic equivalent)

PyPTO's attribution API (e. g. `sum/amax/amin`) supports only one `dim` at a time; `mean` semantics need to be achieved with `sum/count`.
When the semantic of baseline is a "general statute for multiple continuum axes" and does not depend on the intermediate axis results, priority is given to the consolidation of these continuum axes by `reshape` in favour of a single axis.

### Why?

- Repeatedly, `sum(dim=...)` will introduce the middle tensor and additional movement costs.
- The consolidation of successive axes of statutes, followed by a single statute, would normally reduce intermediate writing and synchronization.

### Common approaches

First, the following three articles must be satisfied at the same time:
1. The axle of the Statute is a contiguous continuous axis in the current tensor layout (which can be combined directly through reshape and does not require conversion/replacement).
2. The intermediate statute results are not reused by other operators (only for subsequent Statutes, not input/output as branches).
3. The merged semantic contract is fully consistent (statistically defined, zoomed in such a way as `sum`/`batchmean`/`sum/count`, output shape).

When three articles are satisfied:
1. Consolidation of successive Statutes in Forward (reshape only, no replacement).
2. In Kernel read a single Statute + the original text of the final Statute.

### Example:

- Space fate of Norm `(N, C, H, W)`:
  - If the goal is a joint statute for `H,W`, it can be transformed into a single-axis statute for `(N, C, H*W)`.

### When should we not merge?

- Intermediate axis results are required (e.g., the intermediate tensor would also be involved in other operators or as an observable output).
- The axis of the Statute are not continuous, or the merger changes semantics (not purely reshape equivalent).
- The merger resulted in a worse combination of `tile/loop` and no gain was measured.

## Quick Check List

- [ ] Whether operator contains the Axis of the Convention
- [ ] Whether continuous removal first meets the empirical threshold (approximately 1KB, calculated as `tile` consecutive period)
- [ ] Whether to avoid using `tile[last_axis] > shape[last_axis]` as "effictive to meet standards"
- [ ] After continuous removal, press`16 -> 32 -> 64`Reciprocal Axis Candidates (Ben)shapeDefault Preferred`16`)
- [ ] Whether or not to make a comparison of axial desserts (e.g. `16/32/64`) in the candidate for compliance
- [ ] Whether to avoid visible `tile[i] > shape[i]` (no gain waste)
- [ ] Satisfactory for `prod(tile_shape) <= 16384` and `auto_tiles <= 2048`
- [ ] If `auto_tiles > 2048`, change to loop segment before tile
- [ ] If UB/OOOScheduule is wrong, whether to drop `prod(tile_shape)` from `16384` to `8192/4096` and try again
- [ ] Whether at least two tile candidates have been measured
- [ ] Whether `loop_count` Brush has been done (at least 3 points, 5 points recommended)
- [ ] Whether the end of "overclosed (movement waiting)" and "overweight (single task overweight)" were checked simultaneously
- [ ] Whether the matmul is first inverted by `loop_count`'s mid-point candidate `BASIC_BATCH` (rather than copying the constant)
- [ ] Whether or not the multiaxis statutes have assessed the achievement of the equivalent of the "Continued Axis Together and One Axis Statutes"

## To be completed

- Rule 4: Case split guidelines for two paragraphs tile
- Rule 5: Impact of operator integration and numerical stability on performance
