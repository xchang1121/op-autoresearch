---
name: triton-ascend-reduce
description: "Applicable to contract of return(reduce)CategoryoperatorAnd a combination of steps with a treaty.operatorOptimistic guidance (e.g., consolidated).operatorIncluding:sum, mean, max, min, prod, argmax, argmin, cumsum, cumprod, softmax, logsoftmax, layernorm, rmsnorm, groupnorm, instancenorm, batchnorm, l1norm, l2norm, frobeniusnorm, var, std, average_pooling, sum_poolingWait. It's particularly important: when the dimension is not the last dimension (e.g.dim=1Returnshape=[B,F,D1,D2]) needs to be properly addressed in the multi-dimensional index and in the two-stage contract.PyTorch normalized_shapeMultiaxis to One Semantic Description. Not applicable to pure element-by-fact calculations ormatrix multiplication.If operator is a loss function (element-by-element calculation before global conclusion), the guidance should be selected."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduce"
---

# Reduce operator Optimization

> Applies to contract operations that require the aggregation of multiple values

## Apply operator

**Base contract**: sum, mean, max, min, prod
**: softmax, logsoftmax, playnorm, rmsnorm, groupnorm, watchnorm
**Statistics**: varance, std

## Critical Optimization: Calculating Reorganization (latency Convention)

> **Ascend `tl.sum`/`tl.max`/`tl.min`, etc., has a high cost of contracting instructions**, and each of the iterative periods within the cycle is referred to dating as a performance bottleneck. Core thinking:**In the cycle, only element-by-element accumulation (`+=`) is done, and again after the cycle has been completed.**

### Counter Mode vs Correct Parameter

```python
# ❌ reverse mode: tl.sum is transferred every time in the cycle, resulting in N/BLONK_SIZE sub-register
total = 0.0
for offset in range(0, N, BLOCK_SIZE):
    block = tl.load(ptr + offset + tl.arange(0, BLOCK_SIZE), ...)
    total += tl.sum(block)  # Every one of them returns. → It's expensive.

# ✅ Correct: Element-by-Element in cycle, final sexual return
acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
for offset in range(0, N, BLOCK_SIZE):
    block = tl.load(ptr + offset + tl.arange(0, BLOCK_SIZE), ...)
    acc += block               # Element by Elements addIt's no cost.
total = tl.sum(acc)            # This is the only time I've returned.
```

### 2D scene (return along an axis)

```python
# ❌ inverse mode: every return along axis=0 in the cycle
acc_1d = tl.zeros((BLOCK_N,), dtype=tl.float32)
for m_start in range(0, M, BLOCK_M):
    tile = tl.load(...)  # [BLOCK_M, BLOCK_N]
    acc_1d += tl.sum(tile, axis=0)  # Every one of them returns.

# ✅ Correct: Maintain 2D loader, Last sexual return
acc_2d = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for m_start in range(0, M, BLOCK_M):
    tile = tl.load(...)  # [BLOCK_M, BLOCK_N]
    acc_2d += tile                      # Hold 2DNo return.
result = tl.sum(acc_2d, axis=0)         # Last return. → [BLOCK_N]
```

### Conditions of application

- **Combinable mode of operation**: sum (`+=`), prod (`*=`) etc.
- **Non-sum union (max/min) also applies**: circular inline with `tl.maximum`/`tl.minimum` to take extreme values by element, last `tl.max`/`tl.min`
- **UB volume trade-off**: 2D loader uses more UB (uniform buffer zone) to ensure that `BLOCK_M × BLOCK_N × dtype_size` does not exceed UB capacity. BLONK_SIZE is appropriately reduced when UB is insufficient
- **Mask Processing**: initialize the loader into the returned dollar(s)sum → 0,prod → 1,max → -inf,min → inf), with`other=Yen!`Dealing with borders

### Full example: Sum description over a summary

```python
@triton.jit
def sum_reduce_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    NUM_CORES: tl.constexpr = 20,
):
    """Input X[B, M, N] → Output Y[B, N]Go along. M Axial Summation"""
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_blocks = B * num_blocks_n

    for block_idx in range(pid, total_blocks, NUM_CORES):
        b_idx = block_idx // num_blocks_n
        n_start = (block_idx % num_blocks_n) * BLOCK_SIZE_N
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < N

        # 2D cumulator, latency
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for m_start in range(0, M, BLOCK_SIZE_M):
            m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
            m_mask = m_offsets < M
            x_offset = b_idx * M * N + m_offsets[:, None] * N + n_offsets[None, :]
            x_block = tl.load(x_ptr + x_offset, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
            acc += x_block  # Element-by-Element, no return.

        result = tl.sum(acc, axis=0)  # One-time return after cycle
        tl.store(y_ptr + b_idx * N + n_offsets, result, mask=n_mask)
```

## Universal Return Strategy

### 1. Binary + Atomic Operations

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(data, axis=0)
    tl.atomic_add(output_ptr, block_sum)
```

## Non-last dimension return (key difficult points)

When the entropy dimension is not the last dimension of the tensor (e.g. the consummation of the sape `[B, F, D1, D2]` along `dim=1`),**do not use permute + reshape pre-treatment**, which will result in significant expenses at the host end. The right approach is**to process directly through the dimensional index in Kernel**.

### Core thinking

In RMSNorom, for example, `[B, F, D1, D2]` follows `dim=1` (F dimension):
- **grid 0-D**: all-batch (B)
- **grid 1D**: equalize D1×D2 with each program for a D1D2 block
- **kernel inner cycle**: Part of the Entries F
- **Two phases**: first phase of cumulative statistics (e.g., squares), second phase of output with statistical aggregation

### Standard model: two-stage multi-dimensional return

```python
@triton.jit
def norm_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, F: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr, BLOCK_SIZE_D1D2: tl.constexpr,
):
    pid_b = tl.program_id(0)          # batch Dimensions
    pid_d1d2 = tl.program_id(1)       # D1*D2 Index to block after leveling

    total_d1d2 = D1 * D2
    d1d2_start = pid_d1d2 * BLOCK_SIZE_D1D2
    d1d2_offsets = d1d2_start + tl.arange(0, BLOCK_SIZE_D1D2)
    d1d2_mask = d1d2_offsets < total_d1d2

    # Phase 1: Accumulation of statistics along F dimensions (calculating reorganization: latency contract)
    accum = tl.zeros((BLOCK_SIZE_F, BLOCK_SIZE_D1D2), dtype=tl.float32)
    num_blocks_f = tl.cdiv(F, BLOCK_SIZE_F)

    for f_block in range(num_blocks_f):
        f_offsets = f_block * BLOCK_SIZE_F + tl.arange(0, BLOCK_SIZE_F)
        f_mask = f_offsets < F
        # Multi-dimensional index: x[b, f, d1, d2] → x_ptr + b*F*D1*D2 +f*D1*D2 +d1d2
        x_offsets = pid_b * F * total_d1d2 + f_offsets[:, None] * total_d1d2 + d1d2_offsets[None, :]
        load_mask = f_mask[:, None] & d1d2_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0)
        accum += x_tile * x_tile  # Hold 2D Plus, don't return to the circle.

    rms = tl.sqrt(tl.sum(accum, axis=0) / F + eps)  # One-time return after cycle

    # Phase 2: Normalized output (same cycle structure)
    for f_block in range(num_blocks_f):
        f_offsets = f_block * BLOCK_SIZE_F + tl.arange(0, BLOCK_SIZE_F)
        f_mask = f_offsets < F
        x_offsets = pid_b * F * total_d1d2 + f_offsets[:, None] * total_d1d2 + d1d2_offsets[None, :]
        load_mask = f_mask[:, None] & d1d2_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0)
        y_tile = x_tile / rms[None, :]
        tl.store(y_ptr + x_offsets, y_tile, mask=load_mask)
```

### Hostend Start

```python
def norm_forward(x, eps=1e-5):
    B, F, D1, D2 = x.shape
    y = torch.empty_like(x)
    total_d1d2 = D1 * D2
    BLOCK_SIZE_F = 16
    BLOCK_SIZE_D1D2 = 256
    grid = (B, triton.cdiv(total_d1d2, BLOCK_SIZE_D1D2))
    norm_kernel[grid](x, y, B, F, D1, D2, eps, BLOCK_SIZE_F, BLOCK_SIZE_D1D2)
    return y
```

### Key points

1. **Not permute/reshape**: making `permute → contiguous → view(N, D)` at the host end is extremely expensive for tensor
2. **multi-dimensional index formula**: `x[b, f, d1, d2]` offset in continuous memory = `b*F*D1*D2 + f*D1*D2 + d1*D2 + d2`, simplified to `b*F*total_d1d2 + f*total_d1d2 + d1d2` if D1D2 is displayed
3. **2D file load**: 2D offset matrix constructed with `[:, None]` and `[None, :]`, data obtained from `[BLOCK_F, BLOCK_D1D2]` at a load
4. **Calculating Reorganization**: `accum += x_tile * x_tile` within cycle keep 2D cumulation, once `tl.sum(accum, axis=0)` has been reconnected to avoid every time `tl.sum` is called
5. **grid size**: second dimension `cdiv(D1*D2, BLOCK_SIZE_D1D2)`, may exceed 65535 for larger D1*D2, need to note the grid cap limit for Ascend

## PyTorch Normalization/Register operator Semantics (important)

### Normalized_shape Multiaxis

`nn.LayerNorm(normalized_shape)` when `normalized_shape` is tuple, the integration range is**the product of the last `len(normalized_shape)` dimension**, not the individual dimension.

```python
# Example: input Shape = (B, F, D1, D2), normalized_shape = (F, D1, D2)
# Correct: Normalize F×D1 ×D2 = final 3 dimensions, N = F * D1 * D2
# Error: Normalize F dimension only

# kernel correctly achieved:
total_norm_size = F * D1 * D2  # normalized_shape Multiplication of dimensions
# Do with the last leg (normalized_shape) mean/var
```

### Contract of loss function

- `nn.MSELoss(reduction='mean')`: Average for all elements
- `nn.CrossEntropyLoss`: Enter logits `(N, C)` + figures `(N,)`, internal log_softmax + nll_loss
- loss functions are mostly**elementwise calculate + global reduce**, first press elementwise parsing, then combined with `tl.sum` or `tl.atomic_add`