---
name: triton-ascend-fused-operator-optimization
description: The depth optimization methodology for the integration of operator on Ascend NPU. The coverage ceiling analysis of framework, the multi-Pass consolidation policy, the re-engineering of data access modes, the decision-making phase, the NPU native operator assessment methodology. It applies to scenarios such as elementwise integration, sub-integrated integration, softmax+topk integration, and matmul+action integration.
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
---

# Integration of operator depth optimization methodology

## Before optimization: Performance ceiling analysis framework

Analysis of the**bottleneck type of operator before optimization,**choosing the right direction for optimization and avoiding waste of time near physical limits:

| Bottleneck Type | Method of judgement | Optimizing direction | Typical ceiling |
|---------|---------|---------|----------|
| Memory bandwidth restricted | Less calculated, more data porter | Reduce HBM reading and writing times | ~1.5-2x |
| Repeatedly. | The same data is read 3+ times | Multiple pass merge into single pass | ~3-4x |
| Data access mode | Inconsistent/strided access | Reconstruct as continuous access | ~5-20x |
| Computation dominance | matmul weight matrix large | It's almost non-existent. | ~1.0x |

### The ceiling calculation method

**Theoretical acceleration ratio = baseline total HBM access / optimized total HBM access**

Example: For `y = f(x) * z` type integration:
- Baseline (2 PyTorch op): read x → writing f(x) → reading f(x) + z → writing y =**4 times**
- Triton Integration: reading x + z → writing y =**2 times**
- Theory Upper = 4/2 = 2x

**Reason for lower actual ceiling**: lower number of HBM visits due to the equivalent of the midbaseline tensor hit L2 Cache.

### Matmul's judgement on dominant integration

## Optimizing method 1: Multiple Pass Merge

### Conditions of application
operatorMultiple separate visits to the same data (e.g.softmax of max→exp_sum→normalize,or topkThe multiple scans.

### Methodology
Merge all passs into a single cross-chronology and complete all calculations in the register:

```python
# Sub-prefecture: Multiple visits
max_val = pass_find_max(data)          # Walking through 1
exp_sum = pass_compute_exp(data)       # Walking through 2
topk = pass_find_topk(data)            # Walking through 3+

# RECOMMENDED: A single trip
data = tl.load(...)
max_val = tl.max(data, axis=0)
exp_vals = tl.math.exp(data - max_val)
exp_sum = tl.sum(exp_vals, axis=0)
probs = exp_vals / exp_sum
# Topk done directly within the same block
first_val = tl.max(probs, axis=0)
```

### Key constraints
The best effect is when the dimension of the engagement is placed in a single BLONK; when the dimension is spent, it is divided and the proceeds diminish.

## Optimizing method 2: Reconstructing data access mode

### Conditions of application
operator refers to mode of continuous / non-continuous access (e.g. a pairing of adjacent elements needs to be visited).

### Methodology
From pairing to semantic grouping of elements in the same block:

```python
# Sub-optimal: processing by flat_idx after display, with cross-stride access to pairing elements
for block_id in range(pid, total_elements // BLOCK_SIZE, CORE_NUM):
    flat_idx = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    d = flat_idx % D
    d_pair = d ^ 1  # Cross stride Random access

# RECOMMENDED: Grouped by semantic dimension, continuous inner layer loading
for group_idx in range(pid, total_groups, CORE_NUM):
    # Calculate Group Coordinates
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        # The pairing element is naturally in the same block
        vals = tl.load(ptr + base + d_offsets)
```

### Why does it work?
Ascend hardware has a significant performance penalty for non-continuous access. The recombinant load reduces the load operation by several times to several dozen times the performance gap.

## Optimization methodology 3: Two-stage decision-making

### Conditions of application
LayerNorm / RMSNorm / GroupNormWhen you need statistics, you need them.operator.

### Conclusions: two phases (2-pass) superior to single Pass

| Programme | Strengths | Disadvantages |
|------|------|------|
| 2-pass (statistical → integration) | compiler current is efficient and UB pressure is controllable | The data goes through twice. |
| Single pass (online statistics) | The data goes through it once. | Cyclical active tensor more, UB pressure high, compiler current optimization limited |

### Reason analysis
Single pass, although reduced once, maintains both means/var and raw data pointers in the cycle, resulting in:
- UB active tensor increased and available space reduced
- Reduced efficiency of compiler for multi-level streaming of complex cyclings
- The actual utilization of bandwidth has declined.

### Recommended

```python
# Pass 1: Statistics
mean_acc = 0.0
var_acc = 0.0
for n_start in range(0, N, BLOCK_SIZE_N):
    data = tl.load(...)
    mean_acc += tl.sum(data, axis=0)
    var_acc += tl.sum(data * data, axis=0)
mean_val = mean_acc / N
std_val = tl.sqrt(var_acc / N - mean_val * mean_val + eps)

# Pass 2: Normalization
for n_start in range(0, N, BLOCK_SIZE_N):
    data = tl.load(...)  # Reload
    normalized = (data - mean_val) / std_val
    tl.store(out_ptr + ..., normalized * weight + bias)
```

## Optimizing Method 4: BLONK_SIZE Optimizing Policy

### Elementwise class

| scene | Recommended BLONK_SIZE | Reason |
|------|----------------|------|
| Large data volume (>1M element) | 4096 | Make full use of UB to reduce recycling costs |
| Small amount of data (<100K element) | 1024 | Reduce UB Pressure |
| Multiple Intermediate Variables (>4 Active Tensor) | 2048 | Prevent UB overflow |

### Auto-optimise with Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['n_elements'],
)
```

## NPU Primary operator (toch_npu) assessment methodology

Before handwritten, Triton Kernel should be evaluated for NPU raw integration with operator, but attention should be paid to common traps:

### Evaluation Checklist

- [ ] **Input format requirements**: Does a specific continuous tensor layout be required? Does the cost of data fusion/reset offset the gain?
- [ ] **accuracy Consistency**: internal cumulative order, rounding pattern consistent with baseline? diff values to be measured
- [ ] **Compiled costs**: Is there a larger compilation of latency for first call?

### common issue mode

| Problem | Performance | Solutions |
|------|------|---------|
| Enter layout does not match | Concat/reshape pre-treatment, copy > Kernel gain | Use only when entering natural format requirements |
| accuracy inconsistent | Validate diff exceeding limit (often fp16) | Change to a step-by-step PyTorch operation |
| Approximate Mode Difference | Activate function accuracy with large deviations | Manually achieve precision formulae |

### torch.compile / torch.jit.script

On Ascend NPU, for simple integration**there is usually no significant gain**:
- Compile expensive, steady-state performance close to manual fusion
- Figure optimization is limited by NPU backend support

## Don't overcomposed.

Plug all operations into a Kernel that could result in:
- UB Spill → forced to shrink BLONK_SIZE → as a whole
- compiler pipeline ' s Optimization lapsed. → 's actual bandwidth utilization rate dropped.
- It's slower than breaking into 2-3 simple Kernels.

**Criterion**: Distribution should be considered when the kernel active tensor number × BLONK_SIZE × sizeof (dtype) × multi_buffer coefficient (2-3) is close to UB capacity.
