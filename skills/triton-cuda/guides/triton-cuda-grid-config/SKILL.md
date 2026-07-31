---
name: triton-cuda-grid-config
description: "Grid/Block Configuration Policy, including thread block size selection, SM occupancy optimization and large sape operator processing scheme. This applies to kernel generation scenarios that need to be used to determine CUDA start parameters, optimize GPU co-efficiency, or process mega-data"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# Grid Configuration Policy

Grid configuration is the key to Triton Kernel startup. This document provides Grid configuration policy for Triton CUDA and a large Shape processing scheme.

---

## 1. Grid Settings

### Dimension Format
- **Grid must be a tuple type**, maximum 3D
- Supported format: `(x,)`, `(x, y)`, `(x, y, z)`

```python
# Correct.
grid = (100,)
grid = (100, 200)
grid = (100, 200, 50)

# Error
grid = 100  # It must be. tuple
grid = [100, 200]  # It must be. tupleIt can't be. list
```

### Use lambda (autotune scene)

When using autotune, grid must use lmbda:

```python
# Autotune must use lmbda
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

# Can be calculated directly when not autotune
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
```

---

## 2. 1D Grid Configuration

### Element-wise operator

The most common configuration method: each block handles BLONK_SIZE elements.

```python
n_elements = input_tensor.numel()
BLOCK_SIZE = 1024
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
```

### Line-by-line (Reduce Class operator)

Each block handles one or more rows:

```python
n_rows, n_cols = x.shape
BLOCK_SIZE = triton.next_power_of_2(n_cols)

# Method 1: one block per line
grid = (n_rows,)

# Mode 2: Limit parallelity (grid parallel loop)
num_programs = min(n_rows, 65535)
grid = (num_programs,)
```

---

## 3. 2D Grid Configuration

### MatMul Class operator

Two-way parallels using 2D Grid:

```python
BLOCK_M, BLOCK_N = 128, 256
grid_m = triton.cdiv(M, BLOCK_M)
grid_n = triton.cdiv(N, BLOCK_N)

# Method 1: 2D Grid
grid = (grid_m, grid_n)

# Mode 2:1D Grid (more flexible, supported Grouping)
grid = (grid_m * grid_n,)
```

### 1D vs 2D Grid

| Features | 1D Grid | 2D Grid |
|------|---------|---------|
| Flexibility | High (Supported Grouping) | Low |
| Code Complexity | Manual calculation required pid_m, pid_n | Direct Access |
| L2 Cache Optimization | It's easy to achieve. | Not easy to achieve. |
| Recommended scene | MattMul (Cache optimization required) | Simple 2D operator |

**Recommended**: For the MatMul category operator, use 1D Grid + Grouped Ordering.

---

## 4. Great Shape Process: Grid Stride Loop

### Description of the problem

CUDA GPUYeah.gridSize is limited (normally)2^31 - 1 per dimensionBut more importantly, too big.gridThis will result in:
- Increased start-up costs
- Waste of resources (only a small amount of data per block)

### Grid Stride Loop

Each block handles multiple data blocks by recycling:

```python
@triton.jit
def grid_stride_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # Grid stride loop
    for block_start in range(pid * BLOCK_SIZE, n_elements, num_pids * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)

# Limit Grid Size
MAX_GRID = 65535
num_blocks = min(triton.cdiv(n_elements, BLOCK_SIZE), MAX_GRID)
grid = (num_blocks,)
grid_stride_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=1024)
```

### Grid Stride Loop from Softmax

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    # Multiple rows per block
    for row_idx in tl.range(row_start, n_rows, row_step):
        # Deal with row row_idx...
        pass

# Limiting the number of procedures
num_programs = min(32, n_rows)
softmax_kernel[(num_programs, 1, 1)](...)
```

---

## 5. Batch dimensions processing

### 3D Grid

For a watch operation, a third dimension can be used:

```python
# batch matmul
batch_size = Q.shape[0]
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), batch_size)

@triton.jit
def batch_kernel(...):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_idx = tl.program_id(2)
```

### Flatten Batch

Scale the size of the bat to 1D:

```python
# & Draw (batch, M, N)
x_flat = x.view(-1, N)  # (batch * M, N)
total_rows = batch_size * M
grid = (total_rows,)
```

---

## 6. Summary of best practice

### Element-wise operator
1. 1D Grid:`(triton.cdiv(n_elements, BLOCK_SIZE),)`
2. BLOCK_SIZE = 1024
3. Large size use grid mode loop

### Reduce operator
1. 1D Grid: `(n_rows,)` or `(min(n_rows, max_programs),)`
2. BLOCK_SIZE = triton.next_power_of_2(n_cols)
3. Grid mode loop processing multiple rows

### Matt Mul operator.
1. 1D Grid + Grouped Ordering
2. Or 2D Grid.
3. Search for optimal configuration using autotune

### Attention operator
1. 1D Grid: Parallel by Query Location
2. or 3D Grid: (seq_len / BLONK, heads, watch)

---

## 7. Common errors and solutions

### Error 1: Grid parameter type error
```python
# Error
grid = 1024  # It must be. tuple
grid = [1024]  # It must be. tuple

# Correct.
grid = (1024,)
```

### Error 2: Autotune without lambda
```python
# Error: autotune calculates grid
grid = (triton.cdiv(M, BLOCK_M),)  # BLOCK_M Unknown

# Correct: use lmbda
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
```

### Error 3: Grid is too large to cause performance to decline
```python
# Not recommended: one block per element
grid = (n_elements,)

# Recommended: Rational block size
grid = (triton.cdiv(n_elements, 1024),)
```

---

## 8. Performance Modified Recommendations

### Grid Size Selection
1. **Large enough**: take full advantage of all SM (A100 has 108 SM)
2. **Not large**: avoid unnecessary start-up costs
3. **Empirical value**: Grid usually is 2-4 times more than SM size

### Load Balance
1. Ensure that each block handles a similar workload
2. Grid Sride loop natural load balance
3. Grouped Ordering of MatMul requires attention to the size of the boundary group.

### Cooperate with Autotune
1. Grid uses lmbda to quote autotune parameters
2. Different BLONK_SIZE leads to different Grid sizes
3. Autotune will automatically search for optimal combinations.
