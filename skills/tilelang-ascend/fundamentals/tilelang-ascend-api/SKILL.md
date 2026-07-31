---
name: tilelang-ascend-api
description: "TileLang Ascend API reference manual. Provides a complete reference to API for structure overview, memory distribution, data handling, matrix calculations, attribution, element-level calculations, Tile Extensions, Synchronization, Schedulers, and others. Encoding paradigms and optimisation strategies can be found in a guide for operator type."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang Ascend API best practice

## API Speed Checklist

### Kernel definition

| API | Annotations |
|-----|------|
| `@T.prim_func` | Define Kernel function |
| `T.Tensor((M, N), dtype)` | Declaration of tensor parameters |
| `T.Kernel(block_num, is_npu=True) as (cid, vid)` | Kernel start context |
| `@jit(out_idx=[-1], pass_configs={...})` | JIT Compiler Decorator |
| `T.symbolic('K', 'int32')` | Dynamic Shape |

### Memory Allocation

| API | Annotations | Mode |
|-----|------|------|
| `T.alloc_shared(shape, dtype)` | Shared Level (compiler automatic determination L1/UB) | Developer |
| `T.alloc_fragment(shape, dtype)` | Fragment Level (compiler Auto-Judge L0A /B/C) | Developer |
| `T.alloc_var(dtype, init=...)` | scalar Variable | Developer |
| `T.alloc_ub / T.alloc_L1 / T.alloc_L0A/L0B/L0C` | Visible Assign Storage Levels | Expert |

### Data handling and calculation

| API | Annotations |
|-----|------|
| `T.copy(src, dst)` | Moving data between GM/L1/UB/L0 |
| `T.tile.atomic_add(dst_gm, src_local)` | Add local tensor atoms to GM; V1 supports local/ UB → GM |
| `T.gemm_v0(A, B, C, transpose_A, transpose_B, init)` | Standard GEMM |
| `T.mma(A, B, C, init)` | NPU MMA command |
| `T.reduce_sum/max/min(buffer, out, dim)` | Reunification by dimension |

### Looping and dispatching

| API | Annotations |
|-----|------|
| `T.serial(N)` / `T.unroll(N)` | Normal / Cycle Expand |
| `T.Parallel(ext0, ext1, ...)` | Parallel cycle at the element level |
| `T.Pipelined(range, num_stages=N)` | pipeline Parallel |
| `T.Persistent(domain, wave_size, index)` | Enduring schedule |

### Sync

| API | Annotations |
|-----|------|
| `T.set_flag / T.wait_flag` | Nuclear pipeline Synchronization |
| `T.barrier_all() / T.pipe_barrier(pipe)` | Tube barrier |
| `T.set_cross_flag / T.wait_cross_flag` | Nuclear Sync |
| `T.sync_all()` | Global Synchronization |

### Common pass_configs

| Configure Item | Annotations |
|-------|------|
| `TL_ASCEND_AUTO_SYNC: True` | AutoSync Insertion |
| `TL_ASCEND_MEMORY_PLANNING: True` | AutoMemory Planning |
| `TL_ASCEND_AUTO_CV_COMBINE: True` | Automatic CV separation (nuclear pipeline) |
| `tl.ascend_auto_cross_core_sync: True` | Automatic inter-nuclear synchronization (pipeline) |

---

## Calculating Original Language: GEMM, Conclude and Tile Extension

---

### 1. Matrix Calculations (GEMM)

#### T.gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False)

Block matrix multiplication to calculate C + = op(A) × op(B). A, B are at the Shared level and C is at the Fragment level.

**Parameters**:

- `A`: Left input matrix (shared level)
- `B`: Right input matrix (shared level)
- `C`: Result cumulative output matrix (fragmentation level)
- `transpose_A`: Whether to convert A (default False)
- `transpose_B`: Whether to convert B (default False)
- `init`: Whether C is Zero before calculation (default False). First iteration needs to be Zero and then cumulative.

```python
A_L1 = T.alloc_L1([block_M, block_K], dtype)
B_L1 = T.alloc_L1([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

for k in T.serial(loop_k):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)
    T.barrier_all()
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
    T.barrier_all()
T.copy(C_L0, C[bx * block_M, by * block_N])
```

**Usage with transfer**:

```python
T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
```

#### T.mma(A, B, C, init=False)

NPU-level matrix multiplied by cumulative commands, which are lower than `gemm_v0`. `transpose_A`/`transpose_B` is not supported. Usually used with `T.alloc_L0A`/ `T.alloc_L0B` and `T.annotate_layout`.

```python
A_L0 = T.alloc_L0A([block_M, block_K], dtype)
B_L0 = T.alloc_L0B([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)
T.annotate_layout({A_L1: make_zn_layout(A_L1), B_L1: make_zn_layout(B_L1)})
T.mma(A_L0, B_L0, C_L0, init=True)
```

---

### 2. Return Operation

#### T.reduce_sum(buffer, out, dim=-1, clear=True, real_shape=None)

#### T.reduce_max(buffer, out, dim=-1, clear=True, real_shape=None)

#### T.reduce_min(buffer, out, dim=-1, clear=True, real_shape=None)

Ascend fast-path reduction, mainly for the UB file / slice buffer scene.

**Parameters**:

- `buffer`: Enter a buffer or buffer slice
- `out`: Destination output buffer or buffer slice
- `dim`:reduce axis
- `clear`: Initialization output before calculation
- `real_shape`: logical validity range for 2D slice buffer; default use of physical buffer shape when not set

**Current range of support**:

- 1D buffer:`0 / -1`
- 2D buffer:`0 / 1 / -1 / -2`
- 3D Buffer: support only the triling-tile axis `0 / 1 / -1 / -2`

**`clear` syntax**:

- `clear=True`: Initialize the output and write the result
- `clear=False`: Reduce result merge to existing output
  - `reduce_sum`:`new_out = old_out + reduced_result`
  - `reduce_max`:`new_out = max(old_out, reduced_result)`
  - `reduce_min`:`new_out = min(old_out, reduced_result)`

**Output shape binding**(for example, 2D input `[M, N]`):

- `dim=-1`: The output can be `[M]` or `[M, 1]`
- `dim=0`: The output can be `[N]` or `[1, N]`
- 2D slice buffer with `real_shape` set, currently frontend compatible with the phsical-layout output, e. g. `[physical_cols]` or `[1, physical_cols]`

**Use of recommendations**:

- `clear` and `real_shape` also support key referencing and compatible representational modes
- Recommended preferred use of keywords to achieve clearer readability
- Illegal `dim`, illegal `real_shape`, illegal export Shape will report direct errors in frontend instead of silently entering backend

**Typical usage**:

```python
# Softmax / Attention scene
T.reduce_max(acc_s_ub, m_i, dim=-1)
T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)

# Clear=False merge
T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1, clear=False)

# slice buffer + real_shape
T.reduce_max(in_shared, out_shared, dim=-1, real_shape=[4, 4])
```

---

### 3. Element-wise (Developer Mode T. Parallel)

Use symbol API in the `T.Parallel` cycle, interplatform compatible.

```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
```

**Floating point line operation**:

| Operations | Algorithms |
|------|---------|
| Absolute value [u] | `T.abs(x)` |
| Index | `T.exp(x)` |
| logour | `T.log(x)` |
| Square | `T.sqrt(x)` |
| Square root countdown | `T.rsqrt(x)` |
| ReLU | `T.max(a, 0)` |

**Float-point double-eye operations**: `+`, `-`, `*`, `/`, `T.min(a, b)`, `T.max(a, b)`

**Integrative calculations**: `~` (bit non), `<<`, `>>`, `&` (bit and), `|` (bit or position)

**vector-scalar Operations and Broadcasting**:

```python
# vector-scalar
for j in T.Parallel(block_N):
    c_ub[j] = a_ub[j] + 1

# Lined
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i]  # b_ub.shape = (block_M // VEC_NUM,)

# The dimensions don't match the broadcast.
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = b_ub[j] + 5  # b_ub Yes. 1D,c_ub Yes. 2D
```

**Language split mode**:

```python
for i in range(block_M // VEC_NUM):  # Line Order
    for j in T.Parallel(block_N):    # Column Parallel
        c_ub[i, j] = a_ub[i, j] * b_ub[i, j]
```

---

### 4. Tile Extension Original (Expert / Mixed Mode T.tile.xx)

The `T.tile.xxx` series interface directly triggers the Ascend operation at the Tile level. They can be used either as a fully manual Expert mode or as a hybrid mode native under the Developer pass_configs.

#### 4.1 Basic arithmetic

| API | Functions | src1 type |
|-----|------|----------|
| `T.tile.add(dst, src0, src1)` | dst = src0 + src1 | Buffer or scalar |
| `T.tile.sub(dst, src0, src1)` | dst = src0 - src1 | Buffer or scalar |
| `T.tile.mul(dst, src0, src1)` | dst = src0 * src1 | Buffer or scalar |
| `T.tile.div(dst, src0, src1)` | dst = src0 / src1 | Buffer or scalar |
| `T.tile.max(dst, src0, src1)` | dst = max(src0, src1) | Buffer or scalar |
| `T.tile.min(dst, src0, src1)` | dst = min(src0, src1) | Buffer or scalar |

#### 4.2 Single-line calculations

| API | Functions |
|-----|------|
| `T.tile.exp(dst, src0)` | dst = exp(src0) |
| `T.tile.ln(dst, src0)` | dst = ln(src0) |
| `T.tile.abs(dst, src0)` | dst = abs(src0) |
| `T.tile.reciprocal(dst, src0)` | dst = 1/src0 |
| `T.tile.sqrt(dst, src0)` | dst = √src0 |
| `T.tile.rsqrt(dst, src0)` | dst = 1/√src0 |
| `T.tile.relu(dst, src0)` | dst = max(0, src0) |

#### 4.3 Operations requiring additional parameters

| API | Functions |
|-----|------|
| `T.tile.leaky_relu(dst, src0, scalar)` | Leaky ReLU, scalar is a negative tilt factor |
| `T.tile.axpy(dst, src0, scalar)` | dst = scalar * src0 + dst |
| `T.tile.sin(dst, src0)` | dst = sin(src0) |
| `T.tile.cos(dst, src0)` | dst = cos(src0) |

#### 4.4 Composite operations

| API | Functions |
|-----|------|
| `T.tile.mul_add_dst(dst, src0, src1)` | dst = src0 * src1 + dst |
| `T.tile.silu(dst, src0)` | dst = src0 * sigmoid(src0) (SiLU/Swish activated) |

**Annotations**:
- `mul_add_dst` perform integration multiplication, multiply src0 with src1 and add it to dst
- dst as input (cumulator) and output
- Support for half, float type (Atlas A2/A3)
- Support is also provided for int16_t, uint16_t, int32_t, uint32_t (Atlas 200I/500A2)

- `silu` Execute SiLU (Swish) Activation Function: x *sigmoid(x)
- Support for half, float type (Atlas A2/A3)

#### 4.5 Logical operations

| API | Functions |
|-----|------|
| `T.tile.bitwise_and(dst, src0, src1)` | dst = src0 & src1 |
| `T.tile.bitwise_or(dst, src0, src1)` | dst = src0 \| src1 |
| `T.tile.bitwise_not(dst, src0)` | dst = ~src0 |
| `T.tile.bitwise_xor(dst, src0, src1)` | dst = src0 ^ src1 |
| `T.tile.bitwise_lshift(dst, src0, scalar)` | Move Operation Left |
| `T.tile.bitwise_rshift(dst, src0, scalar)` | Move Operation Right |


#### 4.6 Comparative Operations

###### T.tile.compare(dst, src0, src1, mode)

For an element-by-element comparison, the result is bit mask (1=true, 0=false). src1 can be a buffer or scalar.

**mode values**: `"EQ"`, `"NE"`, `"GT"`, `"GE"`, `"LT"`, `"LE"`

```python
T.tile.compare(c_ub, a_ub, b_ub, "EQ")   # tensor vs tensor
T.tile.compare(c_ub, a_ub, 1.0, "GT")     # tensor vs scalar
```

#### 4.7 Select Operations

###### T.tile.select(dst, selMask, src0, src1, selMode)

Select the elements by the bits of selMask. Bit=1 selects src0, bit=0 selects src1.

**selMode extract**:

- `"VSEL_CMPMASK_SPR"`: Select by Compare Mask
- Select between `"VSEL_TENSOR_SCALAR_MODE"`:tensor and scalar
- `"VSEL_TENSOR_TENSOR_MODE"`: Select between two tensors

```python
T.tile.select(c_ub, selmask_ub, a_ub, b_ub, "VSEL_CMPMASK_SPR")
T.tile.select(c_ub, selmask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")
T.tile.select(c_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")
```

#### 4.8 gather_mask

###### T.tile.gather_mask(dst, src, src1Pattern)

Collect elements according to mask mode.

**Fixed mode**(src1Pattern is string):

- `"P0101"`: Indexed by even number `"P1010"`: Indexed by odd number
- `"P0001"/"P0010"/"P0100"/"P1000"`: one for every four
- `"P1111"`: Take all

**Custom mode**(src1Pattern is buffer): selected by index.

```python
T.tile.gather_mask(b_ub, a_ub, "P0101")
```

#### 4.9 accuracy conversion

###### T.tile.cast(dst, src, mode, count)

**mode values**: `"CAST_NONE"`, `"CAST_RINT"`, `"CAST_FLOOR"`, `"CAST_CEIL"`, `"CAST_ROUND"`, `"CAST_TRUNC"`, `"CAST_ODD"`

```python
T.tile.cast(b_ub, a_ub, "CAST_RINT", 4096)
```

#### 4.10 Data Operations

| API | Functions |
|-----|------|
| `T.tile.fill(buffer, value)` | Fill buffer with value |
| `T.tile.createvecindex(dst, first_value)` | Create an vector index sequence starting with first_value |
| `T.tile.transpose(dst, src)` | 16×16 2D matrix block conversion |
| `T.tile.gather(dst, src, src_offset, src_base_addr)` | Data collection by deviation |
| `T.tile.arith_progression(buffer, first_value, diff_value, count)` | Generate Parity Columns |

#### 4.10 Atomic Operations

###### T.tile.atomic_add(dst, src)

Adds local tensor tile atoms to the GM target area. The API is the original `T.tile` for Ascend, which does not cost the whole of the GPU-style `T.atomic_add`.

**V1 Scope of support**:

- `dst` shall be GM/global buffer, buffer load or region
- `src` must be local tensor, currently mainly for UB/shared buffer and L0C/ fragment buffer
- `src` and `dst` dtype must be consistent.
- Local ->GM atoms supporting 1D and 2D file region
- `return_prev`, `memory_order`, `use_tma`, constant src or arbitrary expression src

**Supported data type**:

int8, int16, float16, bfloat16, int32, float32

**Use of recommendations**:

- If the business syntax starts to add up from 0, call a visible zero GM output before or within Kernel.
- In hybrid mode, you can use automatic sync and memory planning without having to use handwritten `T.Scope("V")` or `T.barrier_all()`.

**UB-> GM Example**:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

src_ub = T.alloc_ub((tile_n,), "float32")
T.tile.fill(src_ub, 1.0)
T.tile.atomic_add(C[0], src_ub)
```
Pass_config in the example is a minimum use. `TL_ASCEND_AUTO_CV_COMBINE` can be opened at the same time when a hybrid mode or an automatic C/V separation is required; if there is an inter-nuclear C/V dependency, then `TL_ASCEND_AUTO_CV_SYNC` can be matched.

**L0C - > GM Example**:

The scenario that applies to the matrix calculation requires atoms to be added to GM, e.g. more block/core GMM.

```python
src_l0c = T.alloc_L0C((block_M, block_N), dtype)
T.gemm_v0(..., ..., src_l0c, init=True)
T.tile.atomic_add(C[..., ...], src_l0c)
```

**Bottom-level realization**:

The bottom will generate the DMA atomic add semantics of Ascend C: turn on `SetAtomicAdd<T>()`, execute `DataCopyPad` of local -> GM, then close the atomic state by compatible helper.
#### 4.11 Sorting Operations

###### T.tile.sort(dst, src, actual_num)

**Parameters**:

  - dst: target buffer for stored sorted results (val0, index0, val1, index1,...)
  - src: Source operations, data to be sorted (val0, val1, val2,...)
  - Number of elements actually involved in sorting in actual_num:src

**Function**: Sort function, sort any length data in a one-time descending order by numerical size

**Example:**

```
# Sort 131 numbers
# 131 up to 160, src.shape = (1,160), actual_num = 131
T.tile.sort(dst, src, actual_num)
```

**note**:
  - `dst`, like `src` data type, only supports float32 and float16 data type
  - `src` size needs to be 32 or 32 integer times

###### T.tile.merge_sort(dst, src0, src1, src2=None, src3=None)

Merges multiple sorted data blocks to support 2/3/4-way amalgamation. The input/output is in the value-index pair format.

```python
T.tile.merge_sort(merge_dst, src0, src1)            # 2-way
T.tile.merge_sort(merge_dst, src0, src1, src2)       # 3-way
T.tile.merge_sort(merge_dst, src0, src1, src2, src3) # 4-way
```

###### T.tile.topk(dst, src, K, actual_num)

**Parameters**:

  - dst: Target buffers that store TopK results (val0, index0, val1, index1,...)
  - src: source buffer with input data (val0, val1, val2,...)
  - K: Previous K Sorting Results
  - Actual_num: Number of elements actually involved in sorting

**Function**: TopK operation to achieve one-time sorting of source data from large to small, selection of the first K elements, output in (number, index)

**Example:**

```
# Sort 41 numbers and select the top 10
# Need to align 41 upwards to 32 * 2 = 64, K = 10, actual_num = 41
# topk_global.shape = (1, 20)sort_result.shape = (1, 64)
T.tile.topk(topk_global, sort_result, K, actual_num)
```

**note**:
  - `src` size needs to be 32 or 32 integer times

#### 4.12 Comparison of programming paradigms

```python
# Mode I: T. Parallel + symbol API (recommended, cross-platform compatible)
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    b_ub[i, j] = T.exp(a_ub[i, j])

# Mode II: T.tile Extension Original (Expert / Mixed mode, directly trigger hardware command)
T.tile.exp(b_ub, a_ub)
```

---

## Kernel definition, memory allocation and data removal

---

### 1. Kernel definition and startup

#### @T.prim_func

Defines a TileLang Kernel function. The parameter type is `T.Tensor` or `T.Buffer`.

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((M, N), dtype),
    B: T.Tensor((M, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    ...
```

**Supported dtype**: `float16, float32, bfloat16, int8, int16, int32, int64, uint8, uint16, uint32, uint64`

#### Dynamic Shape symbol

**T.symbolic(name, dtype)**: Create directly usable tir. Var
  ```python
  K = T.symbolic('K', 'int32')
  @T.prim_func
  def bar(A: T.Tensor((K,), 'float32')):
      for i in T.serial(K):
          ...
  ```


#### T.Kernel

Defines the context in which the kernel runs, creating the tile block binding with the logical core.

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num
    by = cid % n_num
    ...
```

- **cid**: Calculating Task ID, Range [0, block_num]
- **vid**: Index to Victor unit (0 or 1), A2/A3 architecture CV ratio can be 1:2 or 1:1
- **VEC_NUM**: usually set to 2, which means that each AI Core has 2 Victor computing units

#### @jit Decorator

Triggers the instant compilation, and compiles Kernel into NPU executable code.

```python
@jit(out_idx=[-1], pass_configs=pass_configs)
def tile_add(M, N, block_M, block_N, dtype='float'):
    @T.prim_func
    def main(...):
        ...
    return main
```

**Parameters**:
- `out_idx`: Specifies the output parameter index, e. g. `[-1]` means the last parameter is the output
- Index of `workspace_idx`: workspace parameters (e. g. `workspace_idx=[4,5,6]` in Flash Attention)
- `pass_configs`: Compiler Configuration Options

**Common pass_configs**:
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # AutoSync Insertion
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # AutoMemory Planning
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # AutoCVSeparation (nuclear space)pipeline(needs)
}
```

---

### 2. Memory Distribution Original Language

#### Devloper Mode

TileLang has abstract storage layers, divided into Global, Shared, and Fragment. In the Ascend platform, the Shared level corresponds to L1 Buffer and United Buffer, Fragment to L0A /L0B/L0C Buffer. Users do not need to specify specific hardware storage, TileLang compiler will automatically identify according to the context of the program.

###### T.alloc_shared(shape, dtype)

Allocation of storage space at the Shared Level.

```python
A_L1 = T.alloc_shared((block_M, block_K), dtype)
```

###### T.alloc_fragment(shape, dtype)

Allocation of storage space at the level of fragment.

```python
C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
```

###### T.alloc_var(dtype, init, scope='local.var')

Distribution of scalar variables to support initialization. This applies to markers, counters, and temporary scalar.

```python
flag = T.alloc_var("bool", init=False)
counter = T.alloc_var("int32", init=1)
b = T.alloc_var("int32", init=a)  # Initialize with another variable
```

#### Express Mode

Visible designation of storage locations applies to scenes that require precise memory allocation control.

| API | Storage Level | Annotations |
|-----|---------|------|
| `T.alloc_ub(shape, dtype)` | Unified Buffer (UB) | Victor Calculator |
| `T.alloc_L1(shape, dtype)` | L1 Buffer | Cache on film |
| `T.alloc_L0A(shape, dtype)` | L0A Buffer | Cube Left Matrix |
| `T.alloc_L0B(shape, dtype)` | L0B Buffer | Cube Right Matrix |
| `T.alloc_L0C(shape, dtype)` | L0C Buffer | Cube Output/Accumulation |

**Example used in practice**:

```python
A_L1 = T.alloc_L1([block_M, block_K], dtype)
B_L1 = T.alloc_L1([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)
```

---

### 3. Data moving original language

#### T.copy(src, dst)

Moves a tile data block between different memory levels. Supports the tir. Buffer, BufferLoad, BufferRegion types.

**Supported path**:

| src | dst | Annotations |
|-----|-----|------|
| GM | L1 | Global Memory → L1 Buffer |
| L1 | L0A | L1 Buffer → L0A Buffer (Cube Left Matrix)|
| L1 | L0B | L1 Buffer → L0B Buffer (Cube Right Matrix)|
| L0C | GM | L0C Buffer → Global Memory |
| GM | UB | Global Memory → Unified Buffer |
| UB | GM | Unified Buffer → Global Memory |
| UB | UB | Unified Buffer → Unified Buffer |
| UB | L1 | Unified Buffer → L1 Buffer |

**Use examples**:

```python
# GM → L1
T.copy(A[bx * block_M, k * block_K], A_L1)

# GM → UB(vid split)
T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

# UB → GM
T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

# L0C → GM
T.copy(C_L0, C[bx * block_M, by * block_N])

# BufferRegion Slice removal
T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
```

---

## Scheduled, synchronized

---

### 1. Loop Original Language

#### T.serial(N) / T.serial(start, end, step)

Normal for circulation.

```python
for i in T.serial(N):        # 0..N-1
for i in T.serial(0, N, 2):  # 0, 2, 4, ...
```

#### T.unroll(N)

Recycles the number of small cycles. TileLang transmits the extension hint to TIR.

```python
for k in T.unroll(K_TILE):
    acc += a[k] * b[k]
```

#### While Loop

The loop condition needs to be TIR export. TileLang detects dead loops that are compiled for error.

```python
i = 0
while i < N:
    ...
    if done:
        break
    i += 1
```

**Break and Continue**: all available in the T. Serial/T.unroll/T.Parallel/while cycle.

---

### 2. T.Pipelined

The pipeline is calculated/disposed in parallel and the memory access to latency is masked by pre-empting.

#### Syntax:

```python
for var in T.Pipelined(range, num_stages=N):
    ...
```

- `range`: Number of words
- `num_stages`: Prefeed stages (less than the positive integer of range-1)

#### Nuclear pipeline (Intra-core)

```python
for k in T.Pipelined(loop_k, num_stages=2):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)

    T.barrier_all()
    if k == 0:
        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
    else:
        T.gemm_v0(A_L1, B_L1, C_L0)

    T.barrier_all()
```

Execute order for `num_stages=2`:

| Time | Copy A/B | Compute |
|------|----------|---------|
| t₀ | copy_A_0, copy_B_0 | |
| t₁ | copy_A_1, copy_B_1 | |
| t₂ | copy_A_2, copy_B_2 | gemm_0 |
| t₃ | copy_A_3, copy_B_3 | gemm_1 |
| t₄ | | gemm_2 |
| t₅ | | gemm_3 |

#### Nuclear pipeline (Inter-core)

The water flow between Cube and Victor's nuclear core is parallel:

```python
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
    T.copy(acc_s_l0c, workspace_1[cid, :, :])

    T.tile.fill(acc_s_ub, 0.0)
    T.copy(workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
           acc_s_ub_)
    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
    ...
```

Note:
- pipeline cannot be activated at the same time as pipeline in the core.
- Use of nuclear pipeline shall be activated: `"tl.ascend_auto_cv_combine": True`, `"tl.ascend_auto_cross_core_sync": True`

---

### 3. T.Persistent

Optimizes the movement of data blocks between AI Core, which allows adjacent data blocks to be processed by the same AI Core, and increases the Cache Break rate.

```python
for bx, by in T.Persistent(domain, wave_size, index):
    ...
```

**Parameters**:
- `domain`: iterative space
- `wave_size`:wave size (usually core_num)
- `index`: Current nuclear index (usually cid)

**Example:**

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    A_L1 = T.alloc_shared((block_M, K_L1), dtype)
    B_L1 = T.alloc_shared((K_L1, block_N), dtype)
    C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

    for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                                core_num, cid):
        loop_k = T.ceildiv(K, K_L1)
        for k in T.serial(loop_k):
            T.copy(A[bx * block_M, k * K_L1], A_L1)
            T.copy(B[k * K_L1, by * block_N], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
            T.copy(C_L0, C[bx * block_M, by * block_N])
```

---

### 4. Sync Original Language

#### Pipeline synchronization

| API | Annotations |
|-----|------|
| `T.set_flag(src, dst, eventId)` | Set a nuclear pipeline sync sign (producer completion notification) |
| `T.wait_flag(src, dst, eventId)` | Waiting for nuclei pipeline sync sign (consumer block waiting) |
| `T.barrier_all()` | Global barriers to all pipes |
| `T.pipe_barrier(pipe)` | Barriers to specific pipe lines (e.g. `"MTE3"`, `"V"`) |
| `T.sync_all()` | Global Synchronization |

**Tube name**: `"fix"`, `"mte1"`, `"mte2"`, `"mte3"`, `"m"`, `"v"`

```python
T.set_flag("mte2", "v", 0)
T.wait_flag("mte2", "v", 0)
```

#### Nuclear Sync

| API | Annotations |
|-----|------|
| `T.set_cross_flag(pipe, flag)` | Set a nuclear sync sign |
| `T.wait_cross_flag(flag)` | Waiting for inter-nuclear sync sign |

```python
# Cube Notifications upon completion of the core, Victor
T.set_cross_flag("MTE3", 0)
T.wait_cross_flag(0)
```

> The `set_cross_flag` source code (`ascend.py:114`) also supports the third parameter, `mode` (default 2), which controls the synchronization range: 0 = between all AIC/AIV, 1 = between the same group AIV, 2 = between the same group AIC and AIV.

---

### 5. T.Scope

The execution field for labeling the code blocks.

```python
with T.Scope("C"):   # Cube Domain
    ...
with T.Scope("V"):   # Vector Domain
    ...
```
