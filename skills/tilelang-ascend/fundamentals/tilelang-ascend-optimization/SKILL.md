---
name: tilelang-ascend-optimization
description: "TileLang Ascend operator performance optimization technology. Provides tools such as nuclear internal optimization (Split-K, Double Buffer, MTE2 prep, Full-Load, command vectorization, integration of commands), inter-nuclear optimization (num_stages optimization, synchronization optimization), and the Fixed Core model."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
---

# TileLang Ascend Performance Optimization Guide

Select the optimisation method according to the operator type:

| Optimizing direction | Annotations | Typical means |
|---------|------|---------|
| pass_configs | Adjust compiler pass behavior | Close AutoSync, Close Memory Planning |
| kernel optimization | Raise single kernel command parallel | Double Buffer, L1 Permanent, Directive vector, Split-K Pipelined GEMM |
| Inter-nuclear optimization | Optimizing Cube/Vector nuclear collaboration | Num_stages Optimizing, Synchronizing, Fixed Core Mode |
| pipeline Optimization | Computation of overlap with interview memory | T. Pipelined (nuclear/inter-nuclear flow), T. Persistent (data block movement) |
| Fixed Core | By physical core, reduce redundancy initialization and device memory expansion | `T.Kernel(core_num, is_npu=True)`, Workspace by physical core |
| Command Integration | Reduced number of directives issued | AXPY Integration Directive, Broadcast vector |
| Rare access memory optimization | Efficient handling of discrete data | Double vector checkup, Gaber + consecutively move out, walk away |

**Programming mode selection**: priority is given to**Developer mode**(Automated Memory Planning, AutoSync, compiler AutoSegregation Cube/Vector) and, if performance requirements cannot be met, to**Expert mode**manual control (inflective designation of L1 /UB/L0 level, manual synchronization, fine particle size scheduling).

---

## I. Optimization of priorities corresponding to the operator type

Select the optimisation range according to the operator type (the operator type is judged by `IS_ASCEND_AIC`/ `IS_ASCEND_AIV` in `get_kernel_source()`):

| operator Type | Basis of judgement | Optimizing scope |
|---------|---------|---------|
| **Cube model** | Code contains `IS_ASCEND_AIC` | Cube kernel optimization + Fixed Core |
| **Vector type** | Code contains `IS_ASCEND_AIV` | Victor kernel optimisation + Fixed Core |
| **CV Integration type** | It's both. | Cube + Victor → and inter-nuclear optimization + Fixed Core |

> **Fixed Core model**applies to all operator types (nuclear/inter-nuclear energy enablers), see section 2.9.

Prefer the Devloper feature (autosynchronous, auto-RAM planning) and try to optimize it in the following order:

```
kernel optimization → Inter-nuclear optimization → pass_configs Modified (last resort)
```

---

## II. NUCLEAR EQUIPMENT

> **Optimized order**
> - **Cube operator**: Implementation of Cube kernel optimization (2.1 Split-K splitting policy, 2.2 Double Buffer, 2.3 MTE2 prep, 2.4 Full-Load, 2.5 Small Data Block combined) + 2.9 Fixed Core
> - **Vector-type operator**: Implementation of Victor's kernel optimization (2.2 Double Buffer Vector side, 2.6 Directive vectorization, 2.7 Directive Integration, 2.8 Duplicate storage optimization) + 2.9 Fixed Core
> - **CV Integration operator**: First Cube Nuclear Optimization, →, then Victor Nuclear Optimization, → Final Inter-nuclear Optimization (see chap. III) + 2.9 Fixed Core

### 2.1 Split-K cut-off strategy (Cube nuclear)

**Applicable scene**:
- Matrix multiplication K dimension large, single L1 → L0 move cannot accommodate all data
- The K dimension of GEMM is much greater than the L0 Buffer capacity
- There is a K dimension cycle in the code, but each cycle is waiting for the previous move to be completed

**Rationale**: Split K dimensions into smaller blocks, with Ping-Pong double buffering to duplicate the flow of water calculated by Cube. This is the pre-several strategy for subsequent Double Buffer optimization.

**Before optimization**(string removal and calculation):
```python
for k in T.serial(loop_k):
    T.copy(k_l1, l0b[:, :])
    T.mma(l0a[:, :], l0b[:, :], l0c[:, :])
```

**Optimized**(K axle slice + Ping-Pong double buffering):
```python
for k in T.serial(loop_k):
    side = k % 2
    T.wait_flag("M", "MTE1", SIG_L0AB + side)
    T.copy(k_l1, l0b[side, :, :])
    T.set_flag("MTE1", "M", SIG_L0AB + side)

    T.wait_flag("MTE1", "M", SIG_L0AB + side)
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])
    T.set_flag("M", "MTE1", SIG_L0AB + side)
```

### 2.2 Double Buffer (Cube / Victor nuclear utility)

**Applicable scene**:
- The cycle contains multiple serial operations (handling → calculations → writing back)
- Data blocks can be split into multiple pieces to support pipeline parallel
- Loop with `T.serial`

Note:
- Data blocks after cutting cannot be too small to be able to cover up the current:
  - Victor nuclei: the number of elements of each data block after the cut should be ≥ 128
  - Cube nuclei: the number of elements of each data block after the cut should be ≥ 256
- Achieved: Manual Double Buffer (manual distribution of double buffer, used alternately through `side = k % 2`)
- Synchronization method: Manual double buffering can be opened with `TL_ASCEND_AUTO_SYNC: True` to automatically insert compiler in sync, and then manually with `set_flag` / `wait_flag` if translation results do not match expectations

**Rationale**:
```
Serial Mode:
  Block0: [MTE2][VEC][MTE3]
  Block1:        ----------[MTE2][VEC][MTE3]

Double Buffer:
  Block0: [MTE2][VEC][MTE3]
  Block1:   [MTE2][VEC][MTE3]
```

**Cube nuclear example**:

**Before optimization**(serial execution):
```python
for k in T.serial(loop_k):
    T.copy(k_l1, l0b[:, :])
    T.mma(l0a[:, :], l0b[:, :], l0c[:, :])
```

**Optimized**(handwritten Ping-Pong double buffering, automatic sync enabled):
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# Allocation of double buffering
l0a = T.alloc_L0A([2, block_M, dim], dtype)
l0b = T.alloc_L0B([2, dim, block_N], dtype)
l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

for k in T.serial(loop_k):
    side = k % 2
    T.copy(k_l1, l0b[side, :, :])
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])
```

**Optimized**(handwritten Ping-Pong double buffering+ manually synchronized, automatically synchronized when not meeting expectations):
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

# Allocation of double buffering
l0a = T.alloc_L0A([2, block_M, dim], dtype)
l0b = T.alloc_L0B([2, dim, block_N], dtype)
l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

# Initialize the signal
T.set_flag("M", "MTE1", SIG_L0AB)
T.set_flag("M", "MTE1", SIG_L0AB + 1)
T.set_flag("FIX", "M", SIG_L0C)
T.set_flag("FIX", "M", SIG_L0C + 1)

for k in T.serial(loop_k):
    side = k % 2
    # MTE1 Swap and Cube Calculating Stream
    T.wait_flag("M", "MTE1", SIG_L0AB + side)
    T.copy(k_l1, l0b[side, :, :])
    T.set_flag("MTE1", "M", SIG_L0AB + side)

    T.wait_flag("MTE1", "M", SIG_L0AB + side)
    T.wait_flag("FIX", "M", SIG_L0C + side)
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])
    T.set_flag("M", "MTE1", SIG_L0AB + side)
    T.set_flag("M", "FIX", SIG_L0C + side)
```

**Vector Nuclear Example**:

**Before optimization**(serial execution):
```python
for k in T.serial(loop_k):
    T.copy(GM_data[k], ub_buf)
    T.tile.exp(result_buf, ub_buf)
    T.copy(result_buf, GM_out[k])
```

**Optimized**(handwritten Ping-Pong double buffering, automatic sync enabled):
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# Allocation of double buffering
ub_buf = T.alloc_ub([2, block_size], dtype)
result_buf = T.alloc_ub([2, block_size], dtype)

for k in T.serial(loop_k):
    side = k % 2
    T.copy(GM_data[k], ub_buf[side, :])
    T.tile.exp(result_buf[side, :], ub_buf[side, :])
    T.copy(result_buf[side, :], GM_out[k])
```

### 2.3 MTE2 Pre-Advanced Optimization (Cube Nuclear)

**Applicable scene**:
- Double Buffer is on, but each pipeline busy ≤ 70% (no base)
- K direction cut `kL1Iter ≥ 2`

**Rationale**: The main cycle is converted to a three-part structure for the "first round pre-take → formal cycle" to allow MTE2 to move into the next round of data ahead of schedule and eliminate running costs.

**Before optimization**(portation per wheel + calculation line):
```python
for k in T.serial(loop_k):
    T.copy(k_l1, l0b[side, :, :])  # MTE2 Move in
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])  # Cube Calculate
```

**Optimized**(first round prep + running water mask):
```python
# First Round Pre-PING PING
T.copy(k_l1_iter0, l0b[0, :, :])

for k in T.serial(1, loop_k):
    side = k % 2
    next_side = (k + 1) % 2
    # Prefetch the next round of data to PONG
    if k < loop_k - 1:
        T.copy(k_l1_next, l0b[next_side, :, :])
    # Consumption of the current round
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])
```

### 2.4 Reduction of duplicate loads / Full-Load (Cube nuclear)

**Applicable scene**:
- Smaller matrix on side (e.g. `baseM × K × dtype ≤ L1/2`)
- Number of side loops `T ≥ 2` (e. g. N multi-rotational)
- Small side matrix repeats from GM to L1 in each cycle

**Rationale**: one-time presence of small-side matrices in L1, elimination of repeat GM→L1 removal in side cycle, equivalent to compression of the MTE2 total byte bytes by `(T-1)/T`.

**Pre-optimization**(with small matrix A carried on each wheel):
```python
for n_iter in T.serial(T):
    for k in T.serial(loop_k):
        T.copy(A[bz, by, :, :], a_l1)  # Repeated loads per round
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
        T.gemm_v0(a_l1, k_l1, acc_l0c, transpose_B=True)
```

**Optimized**(A one-time presence L1):
```python
# Initialization: A one-time presence L1
T.copy(A[bz, by, :, :], a_l1)

for n_iter in T.serial(T):
    for k in T.serial(loop_k):
        # A is present, skip porter.
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
        T.gemm_v0(a_l1, k_l1, acc_l0c, transpose_B=True)
```

### 2.5 Integration of small data blocks (Cube nuclei)

**Applicable scene**:
- Existence of small block-by-line data (e. g. Scale, Bias, Lut, etc.), single load < 20 KB
- K cycle more frequently, small pieces of data being repeatedly moved
- Low utilization of MTE2 bandwidth (< 70%)

**Rationale**:will KA small piece of data in the direction to be shredded is combined into a big move.≥ 20 KBI'm sorry. I'm sorry.MTE2The cost of the launch head, please.bandwidthUtilization rate from50%–70%Pull back.80%+.

**Before optimisation**(with small scale on each wheel):
```python
for k in T.serial(loop_k):
    T.copy(scale[k * base_scale:(k + 1) * base_scale], scale_l1)  # Every time. 2 KB
    T.copy(data[k * block_N:(k + 1) * block_N, :], data_l1)
    T.gemm_v0(data_l1, scale_l1, acc_l0c)
```

**Optimized**(consolidation of multiple rounds scale one move):
```python
# Merge 8 wheel scale 1 move (2 KB × 8 = 16 KB)
for k in T.serial(loop_k):
    if k % 8 == 0:
        T.copy(scale[k * base_scale:(k + 8) * base_scale], scale_l1_merged)
    # Remove corresponds from merged buffer by offset
    T.copy(data[k * block_N:(k + 1) * block_N, :], data_l1)
    T.gemm_v0(data_l1, scale_l1_merged[k % 8], acc_l0c)
```

### 2.6 Instruction vector (Vector nuclei)

**Applicable scene**:
- Exists in code for multiple scalar operations in cycle (line-by-line/Element-by-Element)
- Use `range()` to do the same for multiple slices of tensor
- operator contains a large number of element-by-element mathematical operations (e. g. line-by-line alignment in Softmax)

**Note: vector adaptations must ensure that the logic of operations remains unchanged, particularly in the case of data dependence or cumulative operations, where the equivalence needs to be carefully verified.

**Before optimization**(multiple scalar operations in cycle):
```python
for h_i in range(block_M // 2):
    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
```

**Optimized**(single file operation):
```python
T.tile.broadcast(m_i_2d, m_i, tmp_ub)
T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
```

### 2.7 Directive Integration (Vector nuclei)

**Applicable scene**:
- Continuous operation according to specific mode (e.g. `y = a * x + y`)
- Need to reduce the number of instructions issued

**AXPY Integration**: `dst = scalar * src0 + dst`

**Before optimization**(two directives):
```python
T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
```

**Optimized**(using AXPY integration):
```python
T.tile.axpy(acc_s_ub, m_i_2d, sm_scale)
```

**Other integration directives**:
- `T.tile.leaky_relu(dst, src0, scalar)`: ReLU + Multiplication Integration (`dst = max(0, src0) if src0 >= 0 else src0 * scalar`)

**tip**: In addition to the integration commands mentioned above, active search should be made for the integrated mode of calculation in the code, and attempts should be made to replace the multistep base calculation with other composite operating instructions (e.g. `T.tile.select`, `T.tile.clamp`, `T.tile.compare`, etc.) provided by `T.tile`.

> **Note: User confirmation, description of integration programmes and expected benefits, prior to integration of implementing instructions, must be obtained and the code modified with the user ' s consent.

### 2.8 Rare access memory optimization (Vector nuclei)

**Applicable scene**:
- KV data is dispersed in Global Memoory (e. g. Paged Attention, Sparse Attention)
- Access KV data using index/page tables
- We need to use the discrete data first. Gather calculates for continuous blocks.

**Before optimisation**(Element-by-Element Gather+ Frequent Synchronization):
```python
# Single buffer, written immediately after each loop move and containing a large number of barriers
kv_ub = T.alloc_ub([D], dtype)
kv_tail_ub = T.alloc_ub([D_tail], dtype)

for bi_i in range(BI // 2):
    index_i = indices_ub_[bi_i + vid * BI // 2]
    T.barrier_all()
    if index_i > -1:
        block_idx = index_i // block_size
        block_i = block_table[b_i, block_idx]
        block_inter = index_i % block_size
        T.barrier_all()
        # Separate copies of elements by element
        T.copy(KV[block_i, block_inter, 0, :D], kv_ub)
        T.copy(KV[block_i, block_inter, 0, D:], kv_tail_ub)
    else:
        T.tile.fill(kv_ub, 0.0)
        T.tile.fill(kv_tail_ub, 0.0)
    T.barrier_all()
    # Write element by element to Workspace
    T.copy(kv_ub, workspace_1[cid, bi_i + vid * BI // 2, :])
    T.copy(kv_tail_ub, workspace_2[cid, bi_i + vid * BI // 2, :])
    T.barrier_all()
```

**Optimized**(two Buffer Garther + batch written):
```python
# Distribute double Buffer for Gather
kv_ub_gather = T.alloc_ub([BI // 2, D], dtype)
kv_tail_ub_gather = T.alloc_ub([BI // 2, D_tail], dtype)

for bi_i in range(BI // 2):
    index_i = indices_ub_[bi_i + vid * BI // 2]
    block_idx = index_i // block_size
    block_i = block_table[b_i, block_idx]
    block_inter = index_i % block_size
    # Dispersed data Gather to Double Buffer (reduce barrier)
    T.copy(KV[block_i, block_inter, 0, :D], kv_ub_gather[bi_i, :])
    T.copy(KV[block_i, block_inter, 0, D:], kv_tail_ub_gather[bi_i, :])

# Gather completed, one-time batch to Workspace
T.copy(kv_ub_gather, workspace_1[cid, vid * BI // 2 : (vid + 1) * BI // 2, :])
T.copy(kv_tail_ub_gather, workspace_2[cid, vid * BI // 2 : (vid + 1) * BI // 2, :])
```

**Key optimization points**:
- **Dispersed KV Gather**: first separate KV from the continuous area collected from GM to UB, then move out again
- **Double Buffer mechanism**: double Buffer replacement list with `[BI // 2, D]` to support Gather and subsequent calculation of flow cover
- **Reduced Synchronization**: remove `T.barrier_all()` and condition branches from the cycle and increase the efficiency of command delivery

### 2.9 Fixed Core mode (all operator types are common)

**Applicable scene**:
- Logical tasks are much larger than physical cores (e.g. block_num > > 24)
- Workspace device memory distribution increases linearly with block_num
- operator contains a large number of initializations for `alloc_buffer`, `annotate_address`, etc.

**Before optimization**(logical tasks lanch):
```python
with T.Kernel(block_num, is_npu=True) as (cid, vid):
    workspace = T.alloc_L1([block_M, block_N], dtype)
    T.copy(result, workspace[cid, :, :])
```

**Optimized**(by physical core launch, manually assigned tasks):
```python
with T.Kernel(core_num, is_npu=True) as (cid, vid):
    workspace = T.alloc_L1([block_M, block_N], dtype)
    single_core_load = T.ceildiv(block_num, core_num)
    for block_idx in T.serial(cid * single_core_load, (cid + 1) * single_core_load):
        ...
        T.copy(result, workspace[cid, :, :])  # workspace[cid] Reused
```

### 2.10 Pass_configs Modified (last resort)

> **Note: Changes in pass_configs settings equivalent to less-used Devloper features should be used after other optimisations have been tried. This optimization applies to all operator types (nuclear/nuclear).

#### Close AutoSync

**Applicable scene**:
- All the above-mentioned optimization tools have been tried and performance still falls short
- Use Expert mode and need precise control of sync timing
- Automatically inserted sync command leads to unnecessary waiting (confirmed by viewing the generated Ascend C code)

**Before optimization**:
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}
```

**Optimized**:
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}
# Manually insert T. Barrier_all() / T.set_flag / T.wait_flag
```

---

## III. Inter-nuclear optimization

> **Subject applicable**: Only**CV Integration operator**requires inter-nuclear optimization. Pure Cube or pure Victor operator skips this chapter.

### 3.1 Num_stages Modified

**Applicable scene**:
- Use `T.Pipelined` for inter-nuclear flow optimization
- More cycles (e.g. `loop_range ≥ 4`)
- The time-consuming differences between the Cube nuclear and the Vector nuclear are significant, with visible inter-nuclear waiting bubbles.

**Easing proposal**:
- **Constraint**: `num_stages ≥ 2` and `num_stages ≤ loop_range` (maximum not more than the number of cycles)
- Larger `num_stages` values are required when there are more cycles or CV time differences
- Starting with `num_stages=2`, gradually increasing, observing performance change selection best value
- Note that `num_stages` adds memory occupancy to the General Assembly. When `TL_ASCEND_MEMORY_PLANNING` is opened, the number of `num_stages`s should be reduced if the memory overwrites the error

### 3.2 Nuclear Synchronization Optimization

**Applicable scene**:
- CV multiple interactions, multiple cycles
- Retain only > 50% of the time taken for inter-nuclear synchronization after commenting on all computing and moving codes

**Easing proposal**:
- This operation reduces the parallelity between CVs and needs to be used with caution
- Synchronize parameters to 2 maximum
- Performance gains must be validated after implementation and, if not, reversed immediately

**Before optimization**(each task is synchronized):
```python
for i in range(n):
    process()
    T.set_cross_flag("FIX", SEM_ID)
    T.wait_cross_flag(SEM_ID)
```

**Optimized**(sync after multiple assignments):
```python
for i in range(n):
    process()
    if (i + 1) % cross_interval == 0 or i == n - 1:
        T.set_cross_flag("FIX", SEM_ID)
        T.wait_cross_flag(SEM_ID)
```

> **Nuclear Pipeline**: use `T.Pipelined` for inter-nuclear current water cover.

---

## iv. common issue

| The phenomenon | Possible causes | Solutions |
|------|----------|----------|
| C. Large nuclear bubbles | V nuclear time-consuming, `num_stages` too small | Increase `num_stages` |
| Memory Spill | `num_stages` Too big or buffer too big | Decreased block parameters or `num_stages` |
| Orders slow down. | scalar overoperated | Change to `T.tile` vector |
| GM bandwidth is not fully filled | Inefficiency of data removal | Open L1 Permanent, Double Buffer |
| Scalar base high | Too many syncs | Reduce sync frequency, use `cross_interval` |
