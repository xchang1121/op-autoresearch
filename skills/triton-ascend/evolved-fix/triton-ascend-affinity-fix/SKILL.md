---
name: triton-ascend-affinity-fix
description: Triton-ascend Cube/Vector prosthesis common issue fixation: pure matmul mishandled, single buffer intertemporal overlay (WAW), bl.allocshape must constexpr, fixpipe dst must be buffer
category: fix
version: "1.0.0"
metadata:
  case_type: fix
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  requires_affinity: true
---

# Cube/Vector restoration of prostheses

---

## 1. Single buffer is repeatedly covered in outer loop (WAW, most common accuracy match)

### The phenomenon
Cube scope repeats sexpe in the outer loop with the same `c_ub`, Victor scope.
Read the same `c_ub` over and over again in the outer loop. Result: All iterative output values equal**last written**
. Validation report `err_cnt=XXXX`, the largest error.

### Error Code

```python
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, ...):
    pid = tl.program_id(0)
    c_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)  # only 1 Grandpa.

    with al.scope(core_mode="cube"):
        for block_idx in range(pid, NUM_BLOCKS, MAX_CORES):
            acc = tl.dot(a, b, ...)
            al.fixpipe(acc, c_ub, ...)               # I don't think so. i Secondary

    with al.scope(core_mode="vector"):
        for block_idx in range(pid, NUM_BLOCKS, MAX_CORES):
            mm_result = bl.to_tensor(c_ub)            # Always read the last one.
            tl.store(out_block_ptr, mm_result.to(tl.float32))
```

### Cause
- The two `with al.scope` regions do not run concurrently. The generated IR
  executes the full Cube loop (N writes to the same `c_ub`) before the full
  Vector loop (which repeatedly reads the same `c_ub`).
- When the vector loop starts reading, only the last file of the cube loop is left in the `c_ub`.
- This is**WAW.**Sync Original language `al.sync_block_set/wait` solves RAW
  ("Reading occurs after writing" ), it's**not able to restore the historical values that were overwritten**— historical values in physics
  It's already covered.

### Fix: cume, set/wait immediately after each file, reuse c_ub but ensure vector is read out

```python
with al.scope(core_mode="cube"):
    for block_idx in ...:
        al.fixpipe(acc, c_ub, ...)
        al.sync_block_set ("cube",   "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        al.sync_block_wait("vector", "cube",   1, al.PIPE.PIPE_V,   al.PIPE.PIPE_FIX)

with al.scope(core_mode="vector"):
    for block_idx in ...:
        al.sync_block_wait("cube",   "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        tl.store(...)
        al.sync_block_set ("vector", "cube",   1, al.PIPE.PIPE_V,   al.PIPE.PIPE_FIX)
```
---

## 2. `bl.alloc`'s Shape must be a compilation constant (constexpr)

### The phenomenon

```
TypeError: get_buffer_ty(): incompatible function arguments. ...
Invoked with: <ir.builder>, [64, <triton.language.core.tensor>],
              <ir.type>, <ir.attribute>
```

Note that `<triton.language.core.tensor>` appears in the Shape list - a dimension is resolved to runtime
tensor, not constant.

### Error Code

```python
@triton.jit
def kernel(..., n_routed_experts, ...):           # n_routed_experts Yes. runtime Integer
    grad_router_logits_ub = bl.alloc(
        tl.float32, (BLOCK_M, n_routed_experts), al.ascend_address_space.UB)
```

### Gene.
`bl.alloc(etype, shape, address_space)` has sunk to MLIR with `builder.get_buffer_ty(shape, ...)`, which requires that all elements of the Shape element are `int`. `@triton.jit` treats the position parameters without `tl.constexpr` a runtime value and is passed in to `tl.tensor`.

### Fix A: all symbols used by Shape are marked with `tl.constexpr`

```python
@triton.jit
def kernel(..., n_routed_experts: tl.constexpr, ...):   # Compiler period constant
    grad_router_logits_ub = bl.alloc(
        tl.float32, (BLOCK_M, n_routed_experts), al.ascend_address_space.UB)
```

Cost: Each `n_routed_experts` value triggers a JIT recompilation once.

### Fix B: Introduce a new BLONK constant with runtime dimensions

```python
@triton.jit
def kernel(..., n_routed_experts, BLOCK_E: tl.constexpr, ...):   # n_routed_experts Still runtime
    grad_router_logits_ub = bl.alloc(
        tl.float32, (BLOCK_M, BLOCK_E), al.ascend_address_space.UB)
    for e_start in range(0, n_routed_experts, BLOCK_E):
        e_offsets = e_start + tl.arange(0, BLOCK_E)
        e_mask = e_offsets < n_routed_experts
        ...
```

The same.constexprTHE RELEVANT TO THE RELEVANT`bl.subview` of offsets/sizes/strides,`bl.to_tensor(target_shape=...)`,`bl.to_buffer(...)`The hidden formshape.

---

## 3. Pure matmul wrongly used prosthesis for → accuracy's Big error

### The phenomenon

Take it."No reprocessingGEMM"(e.g.`grad_fc2_weight = grad_output.T @ gelu_output`,
`grad_hidden_state = grad_fc1_output @ fc1_weight`) is written as a form of affection: cube
End fixpipe to UB, vector end `bl.to_tensor + tl.store` writing GM. Validation time
**Grad_fc2_bias is capable of pass, and error**appears when added grad_fc2_weather.

### Wrong cause

No reprocessing GEMM does not need vector intervention. Cost of cume→ UB →vector → GM:

1. Once more L0C→UB→GM in transit;
2. An additional set of cube/vector sync events;

### Repair: Rehabilitation to original birth, Triton

```python
@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    """No reprocessing GEMM: Do Not Write al.scope / al.fixpipe / bl.alloc."""
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
        block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
        # A_blk/B_blk/C_blk: tl.make_block_ptr Standard 3 packages (see Guide §6.1)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for _ in range(K_LOOP):
            a = tl.load(A_blk, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(B_blk, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(a, b, acc)
            A_blk = tl.advance(A_blk, (0, BLOCK_K))
            B_blk = tl.advance(B_blk, (BLOCK_K, 0))
        tl.store(C_blk, acc, boundary_check=(0, 1))
```
