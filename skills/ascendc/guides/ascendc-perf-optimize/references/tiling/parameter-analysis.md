# Parameter space and algorithm enabling analysis

> Candidates are listed as pre-steps. Failure to complete this analysis is invalid.

---

## Order of implementation

Four-stage staggering, not skipping or merging:

| Phase | Objective | Doors are forbidden. |
|------|------|------|
| §1 | Determines the tiling full set of parameters actually received by kernel | List is not empty |
| §2 | Categorization of parameters, binding sources retroactively | Classify for each parameter |
| §3 | Determines the algorithms available under current Shape/dtype | ≥1 algorithm hit |
| §4 | Build a candidate space for each goal algorithm | Every algorithm has a generation rule. |

Outputs are written in `{working directory}/parameter spatial analysis.md '.

---

## §1 Kernel full set of parameters

1. Reads the demo code and finds the data structure of the host to transfer tiling parameters to kernel, listing all fields
2. Distinction: Which fields are transmitted to Kernel and which are used only on the host side. The latter are not modulated and excluded
3. Analyse how these fields are used internally in Kernel, extracting implied restraints (separation conditions, scope checks, etc.)

Output: List of Parameters + The use and binding of each parameter in Kernel.

---

## §2 parameter classification and binding retroactive

Do two things for each parameter in §1:

### Type determination

| Type | Decision | Search Behaviour |
|------|------|---------|
| Fixed Input | The scale of the problem directly from the user-specified | Do Not Search |
| It's independent. | There is a search/selection logic, multiple legitimate values | Search dimensions |
| Absent. | Calculated by formulae for other parameters | Autocalculate with independent variables |

### Retroactivity of binding sources

Limit the range of values to be taken from each parameter and define the source retroactively:

- From chip specification configuration (buffer capacity, number of cores, alignment) →**Hardware constraint**, not relaxed
- Text size from hard-coding →**Software experience value**, available for search

Method of determination: The value chain goes back to the original definition — the platform configuration field or constant.

Output: Catalogue + Compressed Source Table + List of Parameters Interrupted by Software (mark cut-off values and hardware caps).

---

## §3 algorithms enable determination

The algorithm decision tree of tiling-flow.md, which loads the operator family, checks the doorbar. For the algorithm of each fate, extracts the difference between its parameters and the default baseline (additional constraints, fixed constants, adjustable parameters increase or decrease).

Output: `A algorithm → hit / missed (cause) + adjustable parameters + special constraints '.

### Sample drums

Multishape / Multiple dtype operator cannot construct only one global search space.

| Half-barrel dimensions | Examples | Candidates for influence |
|---|---|---|
| dtype | fp32,fp16,bf16,int8,int64 | tile bytes, whether Cast scratch is needed, can count native |
| rank/layout | rank2 contigouous, rank5 contigouous, non-contigouous | Whether to flatten, whether to inexmap |
| broadcast | same-shape,scalar,last-dim,general | Whether or not to take the route of Broadcast |
| reduction axis | Last-dim, Little D, Inconsistent Axes, Large D | scalar Statute, Single tile Statute, Phase II Statute |
| index pattern | Continuous period, dim0/dim1, random index | DataCopy block or scalar fallback |
| Special semantics | all-zero,all-NaN,single segment,identity reduce | Fill/copy/sum Quick Path |

The output needs to include the number of samples per barrel, the total time-consuming ratio and the slowest sample.

Example:

```text
bucket large_same_shape_int8:
  samples: 1
  total time share: 42%
  enabled algorithms:
    - native integer vector path
    - framework bypass upper-bound
  tunables:
    - tileLength: 4096, 8192, 12288
    - bufferNum: 1, 2

bucket last_dim_broadcast_half:
  samples: 3
  total time share: 12%
  enabled algorithms:
    - row-wise broadcast reuse
    - generic index mapping
  tunables:
    - rowsPerTile
    - broadcast cache length
```

---

## §4 candidate space

For each hit algorithm:

1. Independent adjustable parameters to take the full range of their**hardware binding**, with a hardware alignment particle size and not subject to software cut-off values
2. The derivative parameter is calculated using the §2 formula. If the software is cut, the cut-off value and the hardware ceiling are maintained as a candidate dimension
3. Final verification of all candidates with hardware constraints, or exclusion if not satisfied

Output: Independent range of variables per algorithm, hardware constraints, derivative formulae, extension dimensions, projected candidates.

Candidate space should include not only numerical parameters, but also structural candidates:

| Structure Candidates | When is it included? |
|---|---|
| queue depth 1/2/3 | MTE does not overlap with VEC and UB account book allows |
| scalar small-D path | The cost of synchronizing the Statute is obvious. |
| bulk CopyOut | More lowercase per line/element |
| native dtype path | Cast high ratio and accuracy allowed |
| direct copy/fill path | Semantic value or clarity |
| host dispatch bypass | General Kernel rewrite high risk, a semantic drum is clearly degraded |

Each structural candidate should have clear conditions for commissioning and regression, and not just one switch name.
