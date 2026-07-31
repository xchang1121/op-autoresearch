---
name: triton-ascend-a5-attention
description: "Applicable toA5(Ascend950Attention.(attention)Mechanismsoperator. WhenoperatorThe core calculation is:TransformerThe style of attention calculation and target hardware isAscend950This guidance should be selected.Cube/VectorOperation,al.fixpipe/bl.allocData stream, serial sync mechanism (%2)sync_block_set/wait),Flash AttentionFour-stage decomposition,PMatrixND→NZFormat Conversion, etc.A5Special skills. This document gives the following:Cube/VectorSerial staggered execution version, not applicable to ordinary without attention structurematrix multiplicationOr a contractual operation."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  note: "A5 (Ascend950) Cube/Vector-Accessy Serial"
  operator_type: "attention"
  requires_affinity: true
---

# A5 Flash Attention Serial Optimization Guide

## 1. A5 Hardware Architecture and Flash Attention Map

Ascend950 AI Core contains two key computing units, and the four phases of Flash Attention are mapped on these two cores:

| Units | Duties | Role in Flash Attention |
|------|------|--------------------------|
| **Cube nuclear** | matrix multiplication | QK matmul,PV matmul |
| **Vector** | Element-by-Element | Softmax, exp, unified, flash_update |

## 2. Storage level and data stream

```
GM (Q/K/V/Out)
  ↓ tl.load
Cube: Q@K^T → L0C
  ↓ al.fixpipe (NZ2ND, ROW_SPLIT)
UB: qk_ub (BLOCK_M//2, BLOCK_N)  ← ROW_SPLIT Take it down to two. sub-vector
  ↓ Vector: softmax → p_nz
  ↓ al.copy (UB → L1)
L1: p_l1 (NZ Fractal Format)
  ↓ Cube: P@V → L0C
  ↓ al.fixpipe (NZ2ND, ROW_SPLIT)
UB: pv_ub (BLOCK_M//2, HEAD_DIM)
  ↓ Vector: flash_update → acc
  ↓ tl.store
GM: Out
```

Key points:
- `al.fixpipe` moves Cube 's L0C results to UB and automatically removes them to two sub-vector core using `ROW_SPLIT`
- `bl.alloc` allocates a buffer on UB/ L1 for Cube and Victor to share
- `bl.to_tensor` reads UB data written by fixpipe in Victor scope
- `al.copy` Move P Matrix from UB to L1 for Cube to make PV matmul

## 3. Serial Synchronization

Cube and Victor alternately execute each N-loop in a serial mode, using 3 synchronized events:

```
Cube:  QK matmul → fixpipe → [set 0] → [wait 1] → PV matmul → fixpipe → [set 2]
Vector:              [wait 0] → softmax → copy P→L1 → [set 1] → [wait 2] → flash_update
```

| Event | Direction | sender_pipe → receiver_pipe | Meaning |
|-------|------|-----|------|
| 0 | cube→vector | `PIPE_FIX` → `PIPE_V` | QK fixpipe complete, qk_ub ready |
| 1 | vector→cube | `PIPE_MTE3` → `PIPE_MTE1` | P already copy to L1, p_l1 ready |
| 2 | cube→vector | `PIPE_FIX` → `PIPE_V` | PV fixpipe complete, pv_ub ready |

## 4. Kernel Structure Design

### 4.1 Buffer Distribution (kernel entrance)

```python
qk_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)
pv_ub = bl.alloc(tl.float32, (BLOCK_M // 2, HEAD_DIM), al.ascend_address_space.UB)
p_l1  = bl.alloc(cast_dtype, (BLOCK_N // 16, BLOCK_M // 16, 16, 16), al.ascend_address_space.L1)
```

- UB buffer's line dimensions use `BLOCK_M // 2` because each sub-vector in ROW_SPLIT mode only handles half
- L1 Buffer Using NZ Fractal Format `(BLOCK_N//16, BLOCK_M//16, 16, 16)`

### 4.2 Cube Scope

```python
with al.scope(core_mode="cube"):
    for start_n in range(0, N_Loop, 1):
        _qk_matmul(q, K_block_ptr, qk_ub, HEAD_DIM, BLOCK_N)
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
        _pv_matmul(p_l1, pv_ub, V_block_ptr, HEAD_DIM, BLOCK_M, BLOCK_N)
        al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
```

### 4.3 Vector Scope

```python
with al.scope(core_mode="vector"):
    for start_n in range(0, N_Loop, 1):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        # ... online softmax on qk_ub ...
        # ... copy P (NZ format) to L1 ...
        al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
        al.sync_block_wait("cube", "vector", 2, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        acc = _flash_update(pv_ub, alpha, acc, HEAD_DIM, BLOCK_M)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr_sub, acc.to(Out.type.element_ty))
```

## 5. P Matrix ND → NZ conversion

Victor softmax got P (ND format) to write L1 in NZ fractal format for Cube to do PV matmul:

```
ND (BLOCK_M//2, BLOCK_N)
  → reshape → (BLOCK_M//2, BLOCK_N//16, 16)
  → permute [1,0,2] → (BLOCK_N//16, BLOCK_M//2, 16)
  → reshape → (BLOCK_N//16, BLOCK_M//32, 16, 16)   ← NZ Fractal Format
```

Two sub-vectors verify that each of the different areas of L1 is written through `bl.subview`, which together form a complete P matrix:

```python
p_l1_sub = bl.subview(
    p_l1,
    [0, sub_vec_id * ((BLOCK_M // 2) // 16), 0, 0],
    [BLOCK_N // 16, (BLOCK_M // 2) // 16, 16, 16],
    [1, 1, 1, 1],
)
p_nz = p_nz_tmp.reshape(BLOCK_N // 16, BLOCK_M // 32, 16, 16)
al.copy(bl.to_buffer(p_nz, al.ascend_address_space.UB), p_l1_sub)
```

## 6. Compile Options

```python
_attn_fwd[grid](
    ...,
    debug=True,
    disable_auto_inject_block_sync=True,
    vf_merge_level=1,
)
```

`disable_auto_inject_block_sync=True` is**Required:**: Flash Attention is manually controlled by `sync_block_set/wait` precision and automatically injects will cause death locks or data competition.

## 7. note

1. **fixpipe can only be called in cube scope**, src must be L0C tensor (`tl.dot` result)
2. **`bl.to_tensor` uses**to read UB data written by fixpipe in vector scope; can be used to read L1 data in cube scope
3. **event_id range 0-15**, different shared resources must be used differently
4. **ROW_SPLIT mode, the share of UB buffer should read `(BLOCK_M // 2, ...)`**because each sub-vector sees only half
5. **al.copy / al.fixpipe only A5 available**
6. **set/wait must be strictly paired and balanced**
