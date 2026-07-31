---
name: triton-ascend-a5-api
description: "Atlas A5 (Ascend950) is an exclusive Cube/Vector collaborative programming interface. It covers the full usage and synchronization of Buffer Language (bl.alloc/to_buffer/to_tensor/subview) and Ascend Language (al.scope/fixpipe/sync_block_set/sync_block_wait/sub_vec_id/copy). It applies to high-performance core preparation scenarios that require Cube calculations on A5 hardware before they are handed over to Vector for reprocessing (e.g., Bias/relu/softmax)."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  operator_type: "all"
  requires_affinity: true
---

# A5 Cube/Vector Co-programming Interface

> The interfaces `al.fixpipe` and `al.copy` in this document are only available on A5 hardware (Ascend950).

## 1. Packet Requirements

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al   # Ascend Language
import triton.extension.buffer.language as bl        # Buffer Language
```

## 2. Buffer Language (bl) interface

### 2.1 `bl.alloc(dtype, shape, address_space)` - Buffer on distribution film

```python
# Allocation of a flat32[64,128] buffer on UB
c_ub = bl.alloc(tl.float32, (64, 128), al.ascend_address_space.UB)

# Distribution of Buffer on L1 for NZ format (intermediate results for storage of P Matrix)
p_l1 = bl.alloc(tl.float16, (BLOCK_N // 16, BLOCK_M // 16, 16, 16), al.ascend_address_space.L1)
```

**Address space options**: `al.ascend_address_space.UB` / `al.ascend_address_space.L1` / `al.ascend_address_space.L0C` / `al.ascend_address_space.L0A` / `al.ascend_address_space.L0B`

**Hard rule**: every dimension of `shape` must be the constant of the compilation period. Do not take runtime
tensor (e.g. `n_routed_experts` not indicated for `tl.constexpr`) is placed directly in the Shape.
Include `BLOCK_*: tl.constexpr` as a buffer Shape when dynamic dimensions are required, then use mask
Deal with real borders.

### 2.2 `bl.to_tensor(buffer)` - Buffer Turn tensor (for Victor 's calculations)

```python
# Read UB Buffer content in vector scope
with al.scope(core_mode="vector"):
    c = bl.to_tensor(c_ub)       # Convert to tensor You can do it later.vectorOperations
    result = c + bias[None, :]   # It's normal. tl Operations
```

### 2.3 `bl.to_buffer(tensor, address_space)` -Tensor Turn Buffer (target for fixpe/copy)

```python
# dst parameter for fixpipe must be buffer
al.fixpipe(acc, bl.to_buffer(c_ub, al.ascend_address_space.UB), ...)

# Copy src/dst must also be buffer
al.copy(bl.to_buffer(src_tensor, al.ascend_address_space.UB), dst_l1_buffer)
```

### 2.4 `bl.subview(buffer, offsets, sizes, strides)` - Buffer Slice

```python
# Removes a subview for L1 buffer (for sub_vec_id segment)
p_l1_sub = bl.subview(
    p_l1,
    [0, sub_vec_id * ((BLOCK_M // 2) // 16), 0, 0],
    [BLOCK_N // 16, (BLOCK_M // 2) // 16, 16, 16],
    [1, 1, 1, 1]
)
```

## 3. Ascend Language (al) interface

### 3.1 `al.scope(core_mode=...)` - Specifies the Cube/Vector execution domain

```python
with al.scope(core_mode="cube"):
    # The code in this area runs on Cube Core
    acc = tl.dot(a, b)   # GEMM
    al.fixpipe(acc, c_ub, ...)

with al.scope(core_mode="vector"):
    # Codes in this area run on Victor Core
    c = bl.to_tensor(c_ub)
    result = tl.exp(c)
    tl.store(out_ptr, result)
```

**Note: Only `tl.dot`/ `tl.load` (L0A/L0B)/ `al.fixpipe` etc. in Cubescope do element-wise / reduce / story.

### 3.2 `al.fixpipe(src, dst, dma_mode, dual_dst_mode)` — L0C → UB handling (A5 exclusive)

```python
al.fixpipe(
    acc,                                    # src: L0C Top tensor(tl.dot Results)
    bl.to_buffer(c_ub, al.ascend_address_space.UB),  # dst: UB buffer
    al.FixpipeDMAMode.NZ2ND,              # DMA Mode
    al.FixpipeDualDstMode.ROW_SPLIT,      # Two-Target Mode
)
```

**DMA mode**:
| Mode | Annotations |
|------|------|
| `NZ2ND` | NZ format transfer Normal Dense (most commonly used) |
| `NZ2DN` | NZ Format Translating DN |
| `NZ2NZ` | Keep NZ format |

**Dual_dst_mode**:
| Mode | Annotations |
|------|------|
| `NO_DUAL` | Do not split. Write the whole piece. |
| `ROW_SPLIT` | Split it in half by line for 2 sub-vector core |
| `COLUMN_SPLIT` | Split it in half by row. |

**float32**:
- The last dimension must be eight.
- When `ROW_SPLIT`/ `COLUMN_SPLIT`, the last dimension must be 32 alignment
- When `NZ2DN`, the first dimension must be 8 alignment

**Constraint of alignment (float16/bfloat16)**:
- The last dimension must be aligned with 16.

### 3.3 `al.sync_block_set / al.sync_block_wait` — Transnuclear Synchronization Original

```python
al.sync_block_set(sender, receiver, event_id, sender_pipe, receiver_pipe)
al.sync_block_wait(sender, receiver, event_id, sender_pipe, receiver_pipe)
```

**Parameter description**:
| Parameters | Annotations |
|------|------|
| `sender` / `receiver` | `"cube"` or `"vector"` defining the event channel orientation |
| `event_id` | 0-15, different resources in the same direction must be used |
| `sender_pipe` | SET time: the pipe is empty before the event is launched (ensure data are ready) |
| `receiver_pipe` | WAIT: The pipe is blocked until the event arrives (ensure dependency is satisfied) |

**Basic usage**:
```python
# Notification to Victor after Cube finish fixpipe (PIPE_FIX empty set, PIPE_V unblocked)
al.sync_block_set("cube", "vector", event_id, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

# Victor waits for the fixpe of Cube to complete
al.sync_block_wait("cube", "vector", event_id, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

# Victor 's calculation completed with Cube Notification (PIPE_V emptied set, PIPE_FIX unblocked)
al.sync_block_set("vector", "cube", event_id, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

# Cube waits for Victor to release UB
al.sync_block_wait("vector", "cube", event_id, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

### 3.3.1 Details of PIPE Equities

Ascend NPU has several separate hardware tubes running different operations in parallel. PIPE parameters control the sync point:

| PIPE Enumeration | Corresponding hardware module | Duties | Typical use of scene |
|---|---|---|---|
| `PIPE_V` | Victor Core Calculating Tube | /Reduce /vector | softmax,epilogue,flash_update |
| `PIPE_FIX` | Fixpipe DMA Pipeline | NZ↔ND handling for L0C →UB | After `al.fixpipe` set, fixpipe before wait |
| `PIPE_MTE1` | Memory Transfer Engine 1 | **external → film**removal (HBM →L1/L0) | `tl.load`, cume set after L1 release |
| `PIPE_MTE2` | Memory Transfer Engine 2 | Generic DMA | Less direct use |
| `PIPE_MTE3` | Memory Transfer Engine 3 | **→ Snippet / Outer** | `al.copy`(UB→L1),`tl.store`(UB→HBM) |
| `PIPE_M` | Cube Core (Matrix) pipe | matrix multiplication `tl.dot` | Fewer direct use (indirect synchronization through PIPE_FIX) |
| `PIPE_S` | Scalar Core Pipeline | scalar calculation | Less used |
| `PIPE_ALL` | All pipe lines. | Empty All | debug barrier |

### 3.3.2 Sender_pipe / receiver_pipe

Different Sender→receiver orientation corresponds to different PIPE combinations:

| Direction | Typical Sender_pipe | Typical receever_pipe | Semantic |
|---|---|---|---|
| cube→vector (fixpipe data ready) | `PIPE_FIX` | `PIPE_V` | vector readable UB after fixpipe |
| vector→cube (UB released) | `PIPE_V` | `PIPE_FIX` | cube can fixpipe overwrite UB after vector calculates |
| cube→vector (L1 released) | `PIPE_MTE1` | `PIPE_MTE3` | cube after reading L1 vector can copy L1 overwrite |
| vector→cube (L1 data ready) | `PIPE_MTE3` | `PIPE_MTE1` | vector copy after L1 |

### 3.3.3 Key rules

1. **set/wait of the same (sender, receiver, event_id) channel must be balanced**without a dead lock or data competition
2. **Different shared resources must use different events_id**to avoid conflict of events

### 3.4 `al.sub_vec_id()` - Retrieving sub vector nuclear ID

```python
with al.scope(core_mode="vector"):
    sub_vec_id = al.sub_vec_id()  # Back 0 or 1

    # Calculates the current responsible deviation of the vector nuclear by sub_vec_id
    OUT_block_ptr = tl.make_block_ptr(
        base=OUT + stride_m * sub_vec_id * (BLOCK_M // 2),
        ...
    )
```

When `fixpipe` uses `ROW_SPLIT`, BLONK_M lines are split in two:
- Sub_vec_id=0 for the first half of nuclear treatment [0, BLOCK_M/2]
- Sub_vec_id=1 lower part of nuclear treatment [BLONK_M//2, BLONK_M]

### 3.5 `al.copy(src_buffer, dst_buffer)` — UB→UB/L1 Removal (A5 Exclusive)

```python
# UB - > L1 (for delivering softmax results to L1 for Cube for PV matmul)
al.copy(bl.to_buffer(p_nz, al.ascend_address_space.UB), p_l1_sub)
```

**Constraint**: src must be in UB, dst must be in UB or L1, share/dtype must be the same.

> `al.copy_from_ub_to_l1` has been abandoned, please use `al.copy` as one.

## 4. note

1. **fixpipe can only be called in cube scope**, src must be L0C tensor (`tl.dot` result)
2. **`bl.to_tensor` to read UB data written by**fixpipe in vector scope; read L1 data in cube scope
3. **event_id range 0-15**, different shared resources must be used differently
4. **ROW_SPLIT mode, the share of UB buffer should read `(BLOCK_M // 2, ...)`**because each sub-vector sees only half
5. **al.copy / al.fixpipe only A5 available**
6. **set/wait must be strictly paired and balanced**, prefree + postwait to finish
7. **`al.compile_hint(tensor, name)` marks inter-temporal survival variables**(e.g. alpha/alpha_pong)
8. **`al.copy_from_ub_to_l1` has been abandoned**, please use `al.copy` as one
9. **`bl.alloc` / `bl.subview` / `bl.to_tensor(target_shape=...)` must have sape parameters for constexpr**
