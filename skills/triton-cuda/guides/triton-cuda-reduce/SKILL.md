---
name: triton-cuda-reduce
description: "Approximate operator (reduce) optimisation policy, which includes skills such as sum/mean/max/min, softmax, playernorm, logsoftmax. This applies to kernel code generation scenarios that need to be calculated for arbitrary dimensions of the CUDA GPU, regularize layers or focus fractions"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "reduce"
  algorithms: "sum, mean, max, min, softmax, layernorm, logsoftmax"
---

# Reduce operator Optimization

> Applies to contract operations that require the aggregation of multiple values

## Apply operator

**Base contract**: sum, mean, max, min, prod
**: softmax, logsoftmax, playnorm, watchnorm
**Statistics**: varance, std

## Universal Return Strategy

### 1. Binary + Atomic Operations

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Loading data
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Internal Convention
    block_sum = tl.sum(data, axis=0)

    # Atomic Operations Return global memory
    tl.atomic_add(output_ptr, block_sum)
```

### 2. Numerical stability processing

**Key**: For operations involving exp (softmax, logsoftmax), the maximum value must be subtracted to prevent spills.

```python
# Error: Direct exp may spill
exp_val = tl.exp(x)

# Correct: minus maximum
max_val = tl.max(x, axis=0)
exp_val = tl.exp(x - max_val)
```

## Specific operator Optimization

### Softmax

**Standard Softmax**: `output = exp(x - max(x)) / sum(exp(x - max(x)))`

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # Get the line of the current program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        # Calculating the starting pointer for the current line
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # Create column offset
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        # Load data, use mask to process boundaries
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # Numerical stability: minus maximum
        row_minus_max = row - tl.max(row, axis=0)

        # Calculating Index (CUDA backend directly using tl.exp)
        numerator = tl.exp(row_minus_max)

        # Calculated denominator (consolidation factor)
        denominator = tl.sum(numerator, axis=0)

        # Calculating softmax
        softmax_output = numerator / denominator

        # Storage Results
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

**Key points**:
- Maximum value must be subtracted (value stability)
- CUDA backend Directly with `tl.exp` (no need to use `tl.math.exp2` like Ascend)
- Multi-line processing using `tl.range` (grid mode loop)

### LayerNorm

**Standard Layer Norm**: `output = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias`

```python
@triton.jit
def layer_norm_kernel(
    X, Y, W, B, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the line of the current program
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # Round one: Calculate the average
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # Second time: calculate the difference
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Save mean and rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Third: Normalize and apply linear transformations
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
```

**Key points**:
- Multiple scans: calculation of averages, deviations, homogenization (for N > BLOCK_SIZE)
- Use float32 for intermediate calculations (even if input is fp16)
- Save mean and rstd for backpropagation

### LogSoftmax

**Standard LogSoftmax**: `output = x - max(x) - log(sum(exp(x - max(x))))`

```python
@triton.jit
def logsoftmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_ptr = input_ptr + row_start
    x = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf'))

    # Numeric stabilization
    max_val = tl.max(x, axis=0)
    x_stable = x - max_val

    # Calculate log(sum(exp(x))
    exp_x = tl.exp(x_stable)
    sum_exp = tl.sum(exp_x, axis=0)
    log_sum_exp = tl.log(sum_exp)

    # LogSoftmax
    output = x_stable - log_sum_exp

    output_ptr_row = output_ptr + row_start
    tl.store(output_ptr_row + col_offsets, output, mask=mask)
```

## Full example: Softmax

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        y = torch.empty_like(x)

        num_programs = min(32, n_rows)  # Limiting the number of procedures

        softmax_kernel[(num_programs, 1, 1)](
            y, x,
            x.stride(0), y.stride(0),
            n_rows, n_cols,
            BLOCK_SIZE
        )
        return y
```

## Performance optimization recommendation

### 1. accuracy Upgrade
Use float32 to make an intermediate calculation, even if input is float16/bfloat16:

```python
# Convert to float32 when loading
x = tl.load(input_ptr + offsets, mask=mask)
x = x.to(tl.float32)

# Calculating...

# Turn back to original accuracy before storage
result = result.to(tl.float16)
tl.store(output_ptr + offsets, result, mask=mask)
```

### 2. Line-by-line processing

For 2D data, it is usually done on a line-by-line basis (in the last dimension):
- Grid: `(n_rows,)` or `(min(n_rows, max_programs),)` Each application handles one or more rows
- Benefits: Each line is independent and easily parallel

### 3. BLONK_SIZE Selection

- **Recommended**: `triton.next_power_of_2(n_cols)` to take up to 2
- Reason**: Alignment to 2 times, compiler optimized better
- **Large column**: use multiple scans (for loops) when n_cols are large

### 4. Grid Stride Loop

When the number of lines is large, use the grid side load instead of assigning one block to each line:

```python
row_start = tl.program_id(0)
row_step = tl.num_programs(0)
for row_idx in tl.range(row_start, n_rows, row_step):
    # Deal with row row_idx
```

## Numerical stability check list

- [ ] Softmax/ LogSoftmax minus the maximum value?
- [ ] Whether the variance calculation is protected by `tl.maximum(var, 0.0)` or eps?
- [ ] Does the division add an eps to prevent the exclusion of zero?
- [ ] Do you want to use float32 for intermediate accumulation?
- [ ] Exp, is it possible to spill?

## Common Errors

1. **Forget to minus maximum value**: Softmax directly exp causes spills
2. **accuracy is not enough**: Full use of fp16 resulting in cumulative error
3. **Except zero error**: variance or zero without pss
4. **The variance is negative**: the value error causes the difference to be slightly less than zero
5. **Boundaries processing**: other parameters are inappropriate (softmax should use `-inf`, sum should use `0.0`)
