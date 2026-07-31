---
name: triton-cuda-attention
description: "A Triton-CUDA realization guide for Attention operator. Includes certified full examples of Flash Attention, variation of variants (Causal/GQA/MQA/RoPE), online Softmax algorithms and common errors"
category: implementation
version: "2.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "attention"
  algorithms: "flash-attention, causal-attention, grouped-query-attention, multi-query-attention"
---

# Triton-CUDA Attention

## Online Softmax core algorithm

Flash Attention reduces memory from O(L²) to O(L) by block + Online Softmax. Each KV block is processed:

```python
# Maintenance of three states: m_i (maximum of line), l_i (exp and ), acc (output loader)
qk = tl.dot(q, k) * sm_scale_log2e       # Q @ K^T, pre-supplied log2(e) For use. exp2
m_ij = tl.maximum(m_i, tl.max(qk, 1))    # Update max
p = tl.math.exp2(qk - m_ij[:, None])     # Value stable exp(CUDA Use it. exp2 Faster)
alpha = tl.math.exp2(m_i - m_ij)         # Modify Factor
l_i = l_i * alpha + tl.sum(p, 1)         # Amend and update the denominator
acc = acc * alpha[:, None]               # Aggregated results before amendment
acc = tl.dot(p.to(v.dtype), v, acc)      # Add current block contribution
m_i = m_ij
# End of cycle: output = acc / l_i[:, None]
```

## Full example: Standard Flash Attention

Enter Q/K/V: `(B, H, L, D)`, validated by A100.

```python
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    N_CTX,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    # grid = (cdiv(L, BLOCK_M), B * H)
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // NUM_HEADS
    off_h = pid_bh % NUM_HEADS

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    # K shapeDeclare as(D, N_CTX),tl.dot(q, k)Directly.Q@K^TThere is no need to switch
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset, shape=(N_CTX, D), strides=(stride_qm, stride_qd),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, D), order=(1, 0))
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset, shape=(N_CTX, D), strides=(stride_om, stride_od),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, D), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset, shape=(D, N_CTX), strides=(stride_kd, stride_kn),
        offsets=(0, 0), block_shape=(D, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset, shape=(N_CTX, D), strides=(stride_vn, stride_vd),
        offsets=(0, 0), block_shape=(BLOCK_N, D), order=(1, 0))

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    q = tl.load(Q_block_ptr)
    sm_scale_log2e = sm_scale * 1.44269504

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        # Enter playout: (B, H, L, D)
        #   B = batch_size, H = num_heads, L = seq_len, D = head_dim
        # If external playout is different (e. g. (B, L, H, D)) it needs to be transpose and contigouous
        B, H, L, D = query.shape
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        out = torch.empty_like(query)
        sm_scale = 1.0 / math.sqrt(D)
        BLOCK_M, BLOCK_N = 64, 64
        D_padded = triton.next_power_of_2(D)
        grid = (triton.cdiv(L, BLOCK_M), B * H)
        _flash_attn_fwd_kernel[grid](
            query, key, value, out, sm_scale,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L, NUM_HEADS=H, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D_padded,
        )
        return out
```

## Variant (based on standard FA differences)

### Causal Attention

Two amendments:

```python
# 1. Cycle upper bounds: only go through the current Q block position (with approximately half of the calculations saved)
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)
hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)
for start_n in range(0, hi, BLOCK_N):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # ...load k, calculate qk...

    # 2. Add cause/effects to qk
    causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = qk + tl.where(causal_mask, 0.0, float("-inf"))
    # Follow-up online softmax remains the same...
```

### GQA(Grouped-Query Attention)

Q has H_q head, K/V has H_kv head (H_q is an integer multiple of H_kv). Only read index:

```python
# kernel parameter increase: Q_NUM_HEADS, KV_NUM_HEADS
off_h_q = pid_bh % Q_NUM_HEADS
# Key: Group map with whole / / not with % (% is staggered, not consistent with the PyTorch semantic)
off_h_kv = off_h_q // (Q_NUM_HEADS // KV_NUM_HEADS)

q_offset = off_b * stride_qb + off_h_q * stride_qh   # Q Use it. Q head
k_offset = off_b * stride_kb + off_h_kv * stride_kh   # K Use it. KV head
v_offset = off_b * stride_vb + off_h_kv * stride_vh   # V Use it. KV head
o_offset = off_b * stride_ob + off_h_q * stride_oh    # Out Use it. Q head
# host side Grid = (cdiv(L, BLONK_M), B*H_q)
```

### MQA(Multi-Query Attention)

Special case for GQA: H_kv=1. K/V does not have head dimensions:

```python
# Kernel: K/V stride remove stide_kh/stride_vh
k_offset = off_b * stride_kb        # Just... batch Offset
v_offset = off_b * stride_vb
# host side: key = key. squeeze(1), pass 3 instead of 4
```

## Transformer speed check.

| Variable | K/V shape | Kernel Changes | Host Changes |
|------|----------|-------------|-----------|
| Causal | Queen | + causal_mask + circulation upper boundary hi | None |
| GQA | (B,H_kv,L,D) | Head Map `//` | Grid Press H_q |
| MQA | (B,1,L,D) | K/V does not read distance | squeeze(1) |

Variables are free to combine, for example, Causal+GQA, with a head map and causal mask.

## Common Errors

| Error | Rehabilitation |
|------|------|
| GQA head map with `%` instead of `//` | `off_h_kv = off_h_q // (H_q // H_kv)` |
| Triton with `tensor[:, :half]` slice | Offset with `tl.arange` |
| Runtime variable marked `tl.constexpr` | Only compile-period constants |
| Forget `acc = acc * alpha[:, None]` | m_i Update must amend the previous acc |
| Forget the end `acc / l_i` | Collapse Normalization After Cycle |
| D unpad to 2 | `D_padded = triton.next_power_of_2(D)` |
| Enter not `.contiguous()` | stride calculation relies on continuous memory |

## Performance Point

- CUDA on `tl.math.exp2` + pre-plication `sm_scale * 1.44269504`, faster than `tl.exp`
- K shapeDeclare as`(D, N_CTX)`,`tl.dot(q, k)`It's straight.Q@K^TThere is no need to switch
- Using `tl.make_block_ptr` + `tl.advance` is safer than manual offset
- Thrusters must float32, `tl.dot(p.to(v.dtype), v, acc)` for accuracy conversion
- Autotune: BLOCK_M/N takes 64 or 128, num_warps=4~8, num_stages=3~4
