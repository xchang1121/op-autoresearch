---
name: triton-ascend-example-matmul-vector
description: "A5(Ascend950) Full Triton-Ascend achieves the full integration of MatMul + Vector. It contains two complete operational cases: (1) MatMul + ReLU(fp16) Single Kernel CV integration; (2) Vision MLP + GELUward - one fusted-cv Kernel + three plain matmul Kernel + two pure-vector Kernel mixed with the file."
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  operator_type: "matmul"
  requires_affinity: true
---

# A5 MatMul + Vector Co-programming - Full code example

## 1. MatMul+ReLU(fp16) - Single Kernel CV Integration
```python

@triton.jit
def matmul_relu_kernel(A_ptr, B_ptr, C_ptr, ...):
    """Parameters:M/N/K + NUM_BLOCKS / NUM_BLOCKS_N / CORE_NUM """
    pid = tl.program_id(0)
    K_LOOP: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    c_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)

    with al.scope(core_mode="cube"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
            # A_blk: (M,K)@(block_m*BM, 0) (BM, BK), order=(1,0)
            # B_blk: (K,N)@(0, block_n*BN) (BK, BN), order=(1,0)

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

    with al.scope(core_mode="vector"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
            sub_vec_id = al.sub_vec_id()

            al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            tile = bl.to_tensor(c_ub)
            tile = tl.maximum(tile, 0.0)                       # ReLU
            # C_blk: (M,N), block_shape=(BM/2, BN), order=(1,0)
            # offsets=(block_m*BM + sub_vec_id*(BM/2), block_n*BN)
            tl.store(C_blk, tile.to(tl.float16), boundary_check=(0, 1))
            al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

Element: This example uses `(BM/2, BN)` UB Buffer + `ROW_SPLIT` + `sub_vec_id`, full of double-vectors; fp16 output requires visible `.to(tl.float16)`; host call must be passed to `disable_auto_inject_block_sync=True`.

---

## 2. Vision MLP + GELU backward - mixed form (used-cv + plain matmul + pure vector)

```python
grad_fc2_bias    = grad_output.sum(dim=0)
grad_fc2_weight  = grad_output.t().mm(gelu_output)
grad_gelu_output = grad_output.mm(fc2_weight)
# GELU backward (tanh approximation) ——
grad_fc1_output  = grad_gelu_output * GELU'(fc1_output)
grad_fc1_bias    = grad_fc1_output.sum(dim=0)
grad_fc1_weight  = grad_fc1_output.t().mm(hidden_state)
grad_hidden_state = grad_fc1_output.mm(fc1_weight)
```

How 7 operations are assigned to Kernel -- this is the integration operator selection strategy for CV integration of operator:

| Operation | Kernel type | Correspond to realization |
|---|---|---|
| `grad_gelu_output = grad_output @ fc2_weight` close to `grad_fc1_output = grad_gelu_output * GELU'(fc1_output)` | **used-cv**(one Kernel) | K1:cube counts GEMM, fixpipe to UB;vector read UB + read `fc1_output` calculates GELU'+ Multiplication + Write `grad_fc1_output` out of GM.**Saves tensor `grad_gelu_output` in the middle (S, I) completely on GM + read.** |
| `grad_fc2_weight = grad_output.T @ gelu_output` | **plain matmul** | K2-1: Native Triton GEMM, host with `grad_output.t().contiguous()` |
| `grad_fc1_weight = grad_fc1_output.T @ hidden_state` | **plain matmul** | K2-2: Native Triton GEMM, host with `grad_fc1_output.t().contiguous()` object |
| `grad_hidden_state = grad_fc1_output @ fc1_weight` | **plain matmul** | K2-3: Native Triton GEMM |
| `grad_fc2_bias = sum(grad_output, dim=0)` | **pure vector** | K3: pure vector reduce |
| `grad_fc1_bias = sum(grad_fc1_output, dim=0)` | **pure vector** | K3: pure vector reduce |

> Integration selection principle:**There is one worthy of being kissed and**- `GEMM-2' closely following the GM of the `GelU' multiplication (S, I) of the border. Other GEMMs are non-reprocessed, using native Triton.

```python

@triton.jit
def _gelu_grad(x):
    """GELU'(x) tanh-approx:tanh_out = tl.math.tanh(SQRT_2_OVER_PI*(x+GELU_C*x^3));
    return 0.5*(1+tanh_out) + 0.5*x*(1-tanh_out^2)*SQRT_2_OVER_PI*(1+3*GELU_C*x^2)"""
    ...

# K1: fused (grad_output @ fc2_weight) + (* GELU'(fc1_output)) -> grad_fc1_output
# Cube + Victor adhesive, middle result grad_gelu_output does not move into GM.
@triton.jit
def _k1_fused_gemm_gelu_kernel(GO_ptr, W_ptr, X1_ptr, GFO_ptr, S, H, I_DIM, ...):
    """Parameters:ptr (grad_output / fc2_weight / fc1_output / grad_fc1_output) + stride
    + BLOCK constexpr + K_LOOP / NUM_BLOCKS / NUM_BLOCKS_N / CORE_NUM."""
    pid = tl.program_id(0)
    fused_ub = bl.alloc(tl.float32, (BLOCK_M, BLOCK_N), al.ascend_address_space.UB)

    with al.scope(core_mode="cube"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
            # A_ptr: (S,H)@(block_m*BM, 0) (BM, BK), B_ptr: (H,I)@(0, block_n*BN) (BK, BN), order=(1,0)

            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for _k in range(K_LOOP):
                a = tl.load(A_ptr, boundary_check=(0, 1), padding_option="zero")
                b = tl.load(B_ptr, boundary_check=(0, 1), padding_option="zero")
                acc = tl.dot(a, b, acc)
                A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
                B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))

            al.fixpipe(acc, fused_ub, al.FixpipeDMAMode.NZ2ND)
            al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

    with al.scope(core_mode="vector"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
            al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            grad_gelu_tile = bl.to_tensor(fused_ub)              # (BM, BN) fp32

            # X1_blk / GFO_blk: (S, I_DIM)@(block_m*BM, block_n*BN) (BM, BN), order=(1,0)
            x = tl.load(X1_blk, boundary_check=(0, 1), padding_option="zero")
            grad_fc1_tile = grad_gelu_tile * _gelu_grad(x)
            tl.store(GFO_blk, grad_fc1_tile, boundary_check=(0, 1))

            al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

# K2: Three no reprocessing GEMM - native Triton. Ban al.scope / al.fixpipe /bl.alloc.
@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    """Native Triton GEMM"""
    ...

# K3: Pure vector.
@triton.jit
def _bias_reduce_kernel(X_ptr, Y_ptr, M, N, stride_xm, stride_xn, ...):
    pid = tl.program_id(0)
    with al.scope(core_mode="vector"):
        for block_n in range(pid, NUM_BLOCKS_N, CORE_NUM):
            n_offsets = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = n_offsets < N
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for m_start in range(0, M, BLOCK_M):
                m_offsets = m_start + tl.arange(0, BLOCK_M)
                m_mask = m_offsets < M
                offs = m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
                acc += tl.load(X_ptr + offs,
                               mask=m_mask[:, None] & n_mask[None, :], other=0.0)
            tl.store(Y_ptr + n_offsets, tl.sum(acc, axis=0), mask=n_mask)

class ModelNew(nn.Module):
    """Serial Cube/Vector affinity version of vision MLP + GELU backward."""

    def forward(self, grad_output, hidden_state, fc1_weight, fc1_bias,
                      fc2_weight, fc2_bias, fc1_output, gelu_output):
        # K1: tail rows must be zeros.
        grad_fc1_output = torch.zeros((S, I), device=device, dtype=torch.float32)
        _k1_fused_gemm_gelu_kernel[(num_cores,)](
            grad_output, fc2_weight, fc1_output, grad_fc1_output, S, H, I, ...,
            debug=True, disable_auto_inject_block_sync=True,
        )

        grad_output_T   = grad_output.t().contiguous()              # (H, S)
        grad_fc1_out_T  = grad_fc1_output.t().contiguous()          # (I, S)
        # K2-1: grad_fc2_weight   (H, I) = grad_output_T  @ gelu_output
        # K2-2: grad_fc1_weight   (I, H) = grad_fc1_out_T @ hidden_state
        # K2-3: grad_hidden_state (S, H) = grad_fc1_output @ fc1_weight
        # All three calls are made in _matmul_kernel; no call is sent to debug/disable_auto_inject_block_sync.

        # K3:
        # _bias_reduce_kernel(grad_output,      grad_fc2_bias, S, H, ...)  -> (H,)
        # _bias_reduce_kernel(grad_fc1_output,  grad_fc1_bias, S, I, ...)  -> (I,)
        # Both times, debug=True, disable_auto_inject_block_sync=True

        return grad_hidden_state, grad_fc1_weight, grad_fc1_bias, grad_fc2_weight, grad_fc2_bias
```

### 2.2 Sources of performance gains

K1 connects the `GEMM-2 + GELU' multiplication `on the film (UB) ',**eliminates (S, I) the GM writing + reading of (S, I) tensor `grad_gelu_output`**— this is the source of all the proceeds of this operator adhesion and writing. K2/K3 moves along the best path, not related to kinship; forceing K2 to become a relative and form, not only without gain, but also to introduce the accuracy question.
