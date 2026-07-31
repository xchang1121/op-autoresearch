---
name: triton-ascend-a5-matmul-vector
description: "The guide is selected when the core calculation for operator is matrix multiplication's after-element operation (e.g., bias plus, ReLU activation, disability add, quantification, etc.). The guide uses two-part movement: a full cycle + a full cycle vector scope + monobuff + a pair of graphic synchronized events. Covers Cube/Vector data stream, ROW_SPLIT split, sub_vec_id index, visible sync_block_set/wait pair, plain mattel recommend writing, key binding speed check, etc. does not apply to pure Vectr/Vect data stream, ROW_SPLIT split, sub_vec_id index, visible sync_block_set/wait pair, plain kernel recommend, key binding speed check."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  note: "A5 (Ascend950) Cube/Vector Family and Interface MatMul + Victor Post-Process Integration"
  operator_type: "matmul"
  requires_affinity: true
---

# MatMul + Victor Co-programming Optimization Guide

## 0. When does it have to be written?

A5CompassionAPI(`al.scope` / `al.fixpipe` / `al.sync_block_set/wait` / `bl.alloc`Real proceeds.**Not fromfixpipeIn itself**  And from it can make  GEMMResults"Don't drop.GMJust give it to me.vectorPost-processing."——Save one.(M, N)VolumeGMCentretensorAnd opencube/vectorThe flow of water overlaps.

This determines the benefits of probity and writing for a kernel form:**CV integration**- `tl.dot` (cube) and integrated vector reprocessing (GELU / ReLU / Sigmoid / Softmax / Bias-add / Scale / Mask / Reduce / Quantification, etc.) in a kernel.

### 0.1 Three hard rules

#### Rule A: Pure MatMul**Absolutely not**with a proximate interface - must be native Triton

> Application: operator's kernel**only**`tl.dot`,**there is no**integrated vector reprocessing. For example:
> - `matmul` operator (`Y = X @ W`);
> - `grad_fc2_weight = grad_output.T @ gelu_output`,
>   `grad_fc1_weight = grad_fc1_output.T @ hidden_state`,
>   `grad_hidden_state = grad_fc1_output @ fc1_weight`

**Must**be written in original Triton - `tl.make_block_ptr` + `tl.load` + `tl.dot` + `tl.store`,**absolutely prohibited**to write to `al.scope` / `al.fixpipe` / `bl.alloc`. Reason:

1. Acc calculated on L0C is the NZ format, and the original `tl.store(GM_block_ptr, acc)` will automatically lower the compiler on the cube path to**hidden fixpe (L0C → GM)**intangible GM - this is the best hardware command path for cube data exports.
2. If `al.fixpipe(acc, c_ub)` manually moves the data to UB and writes GM in `bl.to_tensor + tl.store`,**it will be more than once**L0C→ UB→GM in transit,**more than a pair of events**cube/vector sync, and it will lock the vector unit in vain.
3. At the same time, the UB transit path can easily introduce accuracy questions on ROW_SPLIT / sub_vec_id / non-alignedshape.

The recommended template for pure matmul Kernel is given in 6.1.

#### Rule B: Pure Victor (without `tl.dot`) does not mean anything to them by using native Triton vector, relative and API

Softmax / playnorm / reduce / pure elementwise and others operator all fall into this category.

---

## 1. Apply scene

Many of the core calculations for operator are "matrix multiplication + Element-by-Element Reprocessing" such as:

- **Linear + bias**:`Y = X @ W + bias`
- **MatMul + ReLU**:`Y = ReLU(X @ W)`
- **MatMul+Waste+**: `Y = X @ W + residual`
- **MatMul + GELU**:`Y = GELU(X @ W)`
- **MatMul + Quantification**: `Y = quantize(X @ W)`

This type of operator can be used on Atlas A5 to achieve efficient integration through Cube/Vector collaborative programming: Cube is responsible for matrix multiplication, Vector is responsible for reprocessing, delivering intermediate results through `al.fixpipe` on the film, avoiding backwriting of GM rereading costs.

## 2. Schedule structure (used-cv Kernel)

**Key design principles**: do not write cube and vector as "block-by-block interlocking". The correct formulation is**Two-part format**:

- cube scope:`for tile in [0..N): dot(K-loop) → fixpipe → c_ub; sync_set(cube→vector, EVT0); sync_wait(vector→cube, EVT1)`;
- vector scope: `for file in [0. N]: sync_wait (cube→vector, EVT0); read c_ub → reprocessing → store GM; sync_set (vector →cube, EVT1) '.

EVT0 = data-ready, EVT1 = buffer-free. Single buffer `c_ub` cross file reuse, without EVT1, cube second fixpage will overwrite ub before vector is finished, trigger WAW, and result**all tile output is equal to last tile**written.

## 3. Data stream

`GM(input/weight) → tl.load (cube) → L0C(fp32 acc) → al.fixpipe(NZ2ND,ROW_SPLIT) → UB c_ub(BM/2, BN) → bl.to_tensor (vector) →Reprocessing→ tl.store → GM`

## 4. Structure design for fused-cv Kernel

### 4.1 Buffer Distribution

```python
c_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)
```

- Line dimensions use `BLOCK_M // 2`: Every sub-vector core in `ROW_SPLIT` mode sees only half of the lines.
- dtype with `tl.float32`:Cube L0C loader is fp32, fixpipe directly removed.
- Shape must be `tl.constexpr`.

> Simplified variant: If no sub-vector split is required, `FixpipeDualDstMode.ROW_SPLIT`, Buffer Shape can be saved by using `(BLOCK_M, BLOCK_N)`, vector side without `sub_vec_id` offsets, simpler and more stable writing.

### 4.2 CubeScope — matrix multiplication + fixpipe

```python
with al.scope(core_mode="cube"):
    for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
        block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N

        # A_blk: (M,K), offsets=(block_m*BM, 0), block_shape=(BM, BK), order=(1,0)
        # B_blk: (K,N), offsets=(0, block_n*BN), block_shape=(BK, BN), order=(1,0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _k in range(K_LOOP):
            a = tl.load(A_blk); b = tl.load(B_blk)
            acc = tl.dot(a, b, acc)
            A_blk = tl.advance(A_blk, (0, BLOCK_K))
            B_blk = tl.advance(B_blk, (BLOCK_K, 0))

        al.fixpipe(acc, c_ub,
                   al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT)
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

Key points:

- `tl.dot(a, b, acc)` Three Operating Numbers for K V reduce; `acc` must be visible `tl.zeros` initialized (L0C does not guarantee zero).
- `al.fixpipe(NZ2ND, ROW_SPLIT)` expands the NZ fractal of L0C into ND line priority and automatically cuts to two sub-vectors by line.
- Set/wait: set with `PIPE_FIX → PIPE_V`, wait with `PIPE_V → PIPE_FIX`, match with id.

### 4.3 Victor Spope - Reprocessing + Storage

```python
with al.scope(core_mode="vector"):
    for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
        block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
        sub_vec_id = al.sub_vec_id()                    # 0 or 1

        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        tile = bl.to_tensor(c_ub)                       # (BM/2, BN) fp32
        tile = tl.maximum(tile, 0.0)                    # Post-treatment (%1)ReLU Example:

        # C_blk: (M,N), block_shape=(BM/2, BN), order=(1,0)
        # offsets=(block_m*BM + sub_vec_id*(BM/2), block_n*BN)
        tl.store(C_blk, tile.to(C_ptr.dtype.element_ty), boundary_check=(0, 1))

        al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

Key points:

- `al.sub_vec_id()` returns 0 or 1,**output line offset must be added to `sub_vec_id * (BLOCK_M // 2)`**, otherwise two sub-vecs write to overwrite on the same paragraph GM.
- Bias post-treatment (broadcast): `bias_tile = tl.load(bias_ptr + col_off + tl.arange(0, BLOCK_N)); tile = tile + bias_tile[None, :]`.
- Upon completion of ub `sync_set(EVT_BUF_FREE)`, allow cube to enter the next file.

### 4.4 Common design elements for used-cv Kernel

1. **Post-treatment at the end of vector scope**: cube fixpipe, vector `bl.to_tensor(c_ub)` takes the file, does elementwise/ redation, then `tl.store` goes out.
2. **Additional input involved in the reprocessing process**(e.g. `fc1_output` in GELU') read GM directly by vector with `tl.load`,**no cume**.

## 5. Synchronize event pairing speed check

| Event | Direction | sender_pipe → receiver_pipe | Meaning |
|-------|------|-----------------------------|------|
| 0 | cube → vector | `PIPE_FIX` → `PIPE_V` | "Tile's already in c_ub." |
| 1 | vector → cube | `PIPE_V` → `PIPE_FIX` | "c_ub, I'm finished. I can overwrite." |

**The twinning principle**: set/wait five-dollar group `(producer, consumer, event_id, src_pipe, dst_pipe)` must be fully consistent. Few set or more wait → dead locks; PIPE writes against →'s failure to hit.

## 6. Recommended formulation for plain matmul Kernel (**pure matmul)
### 6.1 Recommended template (no reprocessing GEMM)

```python
@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    """No reprocessing GEMM: Pure raw TritonNo use is permitted. al.scope / al.fixpipe / bl.alloc.
    Import stride,BLOCK constexpr,K_LOOP/NUM_BLOCKS/NUM_BLOCKS_N/CORE_NUM."""
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
        block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N

        # A_blk/B_blk/C_blk: tl.make_block_ptr Standard 3 package, order=(1)
        # A=(M,K)@(block_m*BLOCK_M, 0) (BLOCK_M, BLOCK_K)
        # B=(K,N)@(0, block_n*BLOCK_N) (BLOCK_K, BLOCK_N)
        # C=(M,N)@(block_m*BLOCK_M, block_n*BLOCK_N) (BLOCK_M, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for _k in range(K_LOOP):
            a = tl.load(A_blk, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(B_blk, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(a, b, acc)
            A_blk = tl.advance(A_blk, (0, BLOCK_K))
            B_blk = tl.advance(B_blk, (BLOCK_K, 0))

        tl.store(C_blk, acc, boundary_check=(0, 1))
```

Key points:

- `boundary_check=(0, 1), padding_option="zero"` has to be added, otherwise the data will be read undefined when M / N / K is not aligned.
- Do not pass `disable_auto_inject_block_sync=True` / `debug=True` -- these two are prosthetic, original Triton walking standard lowering.

## 7. Key constraints quick check.

1. **fixpe exit and exit**: must be called in cube scope, src is acc (L0C) for `tl.dot`, dst must be a buffer (UB/L1/L0x) for `bl.alloc`, pass GM block_ptr for `TypeError('dst is not of buffer type')`.
2. **`bl.alloc` Shape must constexpr**; neither is `bl.subview` / `bl.to_tensor(target_shape=...)`. Runtime integer (e. g. `n_routed_experts`) reports `get_buffer_ty()`'s error.
3. **set/wait full pair of five-dollar groups**: `(producer, consumer, event_id, src_pipe, dst_pipe)` no match for dead locks/discards; event id 0-15. Cube and Victor for cycles must have exactly the same block sequence (the starting point/step length).
4. **`disable_auto_inject_block_sync=True` is used only for pronunciation**; plain matmul/ pure vector Kernel is not passed. `al.fixpipe / al.copy / al.scope / bl.alloc` is only A5 available.
