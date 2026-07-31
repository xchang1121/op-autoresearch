---
name: triton-ascend-case-reduction-amin-atomic
description: "Amin (amin) optimization of atomic operation: the nonreduce axis maps reduce axis multiple cores at hours, provides both inner/outside atomic operating options (reduced storage vs less competition), increases performance by double-segregation + calculation, for M<N (e.g. 16×262144) extreme Shape scale scenario"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Amin Completing Atomic Optimization Case

## Task characteristics
- **Data size**(16,262144), nonreduce axis small, reduce axis large
- **Strategy**: mapping the reduce axis to the polynucleus to achieve the return of the cross-line elements through atomic operations

## Optimizing 1: Cut Policy Adjustments

```python
# Simple way: Nonreduce axis map multiple cores
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)

# Error: Optimization: reduce axis map multiple cores
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)

# Retract columns within Kernel
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
```

### Optimizing content
- Adjusting the splitting policy from the nonreduce axis map multi-nuclei to the reduce axis map multi-nuclei
- In order not to exceed the hardware cache, the rows in the kernel are split in a double-slit.

## Optimization 2: Calculate reorganization

```python
# Simple way: multiple returns within the cycle
row_min = float('inf')
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
  Error:curr_min = tl.min(data_block, 1)
    row_min = tl.minimum(curr_min, row_min)

# Correct: Optimistic approach: maintenance of matrix structure
curr_min = tl.full((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    curr_min = tl.minimum(data_block, curr_min)
row_min = tl.min(curr_min, 1)
```

### Optimizing content
- Maintain matrix structure and maintain intermediate results using curr_min
- Consolidation of multiple returns into one contract and reduction in the number of returns

## Optimization 3: Atomic Operations (two scenarios)

### Option I: Atomic operation in cycle

```python
for m_start in range(0, M, BLOCK_SIZE_M):
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    mmask = m_offsets < M

    curr_min = tl.full((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
    for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
        n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
        nmask = n_offsets < N
        mask = (mmask[:, None]) & (nmask[None, :])

        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        data_block = tl.load(block_ptrs, mask=mask, other=float('inf'))

        curr_min = tl.minimum(data_block, curr_min)
    row_min = tl.min(curr_min, 1)

    output_ptrs = out_ptr0 + m_offsets * out_stride0
    tl.atomic_min(output_ptrs, row_min, mask=mmask)  # Atoms per piece immediately
```

**Features**:
- Reduced intermediate storage
- But it's increasing the frequency of atomic operations.

### Option II: Atomic operations outside the cycle

```python
all_row_min = tl.full((M,), float('inf'), dtype=tl.float32)  # Full array pre-allocated

for m_start in range(0, M, BLOCK_SIZE_M):
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    mmask = m_offsets < M

    curr_min = tl.full((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
    for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
        n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
        nmask = n_offsets < N
        mask = (mmask[:, None]) & (nmask[None, :])

        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        data_block = tl.load(block_ptrs, mask=mask, other=float('inf'))

        curr_min = tl.minimum(data_block, curr_min)
    row_min = tl.min(curr_min, 1)
    curr_block_size_m = tl.minimum(BLOCK_SIZE_M, M - m_start)
    all_row_min = tl.insert_slice(all_row_min, row_min, [m_start], [curr_block_size_m], [1])  # Provisional intermediate result

output_ptrs = out_ptr0 + tl.arange(0, M) * out_stride0
tl.atomic_min(output_ptrs, all_row_min)  # Final unified atomic operation
```

**Features**:
- Reduced competition through centralized atomic operations
- Additional storage space is required for large-scale data

## Optimize 4: Configure

```python
@triton.autotune(
    configs=[
        # (AI core=40)
        # Grid = 32 < 40, UB full
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 1024}),  # MIt's a small cut.UBFull.
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 512}),  # MIt's bigger than that.16,NofSUBThrust to small.512Prevention of super.UB
    ],
    key=[...],
    restore_value=['out_ptr0'],  # autotune I have to. restore_value
)
```

### Optimizing content
- The split can be severed by the sape.
- Grid numbers are as large as possible but not exceeding AI core (BLONK_SIZE_N=8192, making grid=32)
- Make kernel cleaving adjustments when the UB is full

### Summary
1. When the nonreduce axis is small and the reduce axis is large, reduce axis is mapped to multiple cores and combined with atomic operations
2. Both atomic operating options have advantages and disadvantages: option one reduces storage but atom operations are frequent, option two concentrates atom operations but requires additional space
3. Once the number is determined, a secondary cut may be considered if the hardware cache is exceeded
4. Adjust the split and nucleus configuration to ensure as much as possible of the UB as possible without going beyond the UB
