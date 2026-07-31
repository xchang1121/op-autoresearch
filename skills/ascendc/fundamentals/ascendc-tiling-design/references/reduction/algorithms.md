# Reduction class operator core algorithm route

> **⚠ ️**: First look at the algorithm to determine the applicable algorithm, then read the corresponding detailed document by link.

---

## Algorithm Select Contrast

| Conditions | Recommended algorithm | Reason | Typical operator | Detailed documents |
|------|---------|------|---------|---------|
| FullLoad | Direct order of calculation | UB, CopyIn once | reduce_sum/max, softmax_v2 | — |
| FullLoad + Double-order return | TwoPass | First request A, second request A, request B | reduce_var/std, layer_norm | — |
| Split + Relevant Conventions | Welford Online | One-way, one-way, one-round IO. | reduce_var/std | [alg-welford.md](alg-welford.md) |
| Split + Stream Cumulative error Large | Welford + Group(8) | Merge anti-error accumulations per 8 block | reduce_var/std | [alg-welford.md](alg-welford.md) |
| The split + mononucle processing is not complete R and A small | Group Reduce | R, workspace sync | arg_max, reduce_var | [alg-group-reduce.md](alg-group-reduce.md) |
| Sum accuracy sensitive | Half plus | Add Quantities to the Quantities first. | reduce_sum, reduce_var | [alg-dichotomy.md](alg-dichotomy.md) |
| Split + Return Belt Index Tracking | Split Merge | Part by piece + cross-section update global index | arg_max_v2 | [with-index.md](with-index.md) |

---

## Algorithm Summary

### Welford Online (one time online)

Split mode downflow calculates two related statistical volumes. A single scan, an incremental update, supports grouping in parallel. Details are provided in [alg-welford.md](alg-welford.md).

### Group Reduce (cross-border return)

R is too big for mononuclear processing, and A is too small for multiple nuclear parallels. Split R to multiple nuclears, workspaces are synchronized after individual nuclear returns. See [alg-group-reduce.md] (alg-group-reduce.md) for more details.

### Half-plus / Half-Interval

Sum is assigned to accuracy optimization. The digonal tree folds in order to add a similar scale first. See [alg-dichotomy.md] (alg-dichotomy.md) for further details.

---
