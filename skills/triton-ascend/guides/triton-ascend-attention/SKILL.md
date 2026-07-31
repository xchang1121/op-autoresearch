---
name: triton-ascend-attention
description: "It applies to attention.(attention)Type of mechanismoperator. WhenoperatorThe core calculation is:TransformerThis guide should be selected for style-based focus operations.operatorIncluding:self_attention, cross_attention, multi_head_attention, flash_attention, scaled_dot_product_attention, causal_attention, masked_attentionand so on.QKVMatrix multiplied by fraction, onlinesoftmaxits causes and consequencesmaskProcessing, processingFlash AttentionKey techniques such as segmented strategies. They don't apply to ordinary people without attention structures.matrix multiplicationOr a contractual operation."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "attention"
---

# Attention operator Optimization

## Standard Attention Calculating Process

Standard Scaled Dot-Product Attention:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### Three phases

1. **QK^TCalculate**: `scores = Q @ K^T / sqrt(d_k)`, calculate the attention score
2. **Softmax Normalization**: `attn_weights = softmax(scores)`, ensuring weight and 1
3. **Weighted sum**: `output = attn_weights @ V`, final output

### Achieving standards

```python
# Accomplishment in a simple manner (highly spent on memory)
scores = (Q @ K.T) / sqrt(d_k)  # (seq_len, seq_len)
attn_weights = softmax(scores)   # Need to store a complete attention matrix
output = attn_weights @ V
```

**Question**
- Attention matrix to store `(seq_len, seq_len)`
- Memory occupation: O(seq_len²)
- For long sequences (seq_len = 4096), memory occupancy is huge

## Flash Attention Optimizing Policy

Flash Attention avoids the storage of a complete attention matrix through segment calculations and online Softmax.

### Core thinking

1. **Branch calculation**: large arrays processed in blocks to reduce memory occupancy
2. **Softmax**online: using incremental softmax algorithms, block calculations, maintenance of global maximum values and aggregating factors
3. **Avoided storage**: No complete attention matrix stored

### Online Softmax algorithm

The key is to maintain global statistics and update them on a block-by-block basis:

```python
# Initialize global statistics
m_i = -float("inf")  # Maximum global value
l_i = 0.0           # Global exp and
acc = 0.0           # Output Thruster

# Part processing
for start_n in range(0, seq_len, BLOCK_SIZE):
    # Loading the fractions of the current block
    scores = tl.load(scores_ptr + start_n, mask=load_mask, other=-float("inf"))

    # 2. Update global max
    m_ij = tl.maximum(m_i, tl.max(scores, 0))

    # 3. Calculates the exp value of the current block (value stabilization)
    scores = scores - m_ij
    p = tl.math.exp2(scores * 1.44269504)  # log2(e)

    # Update global exp and
    l_ij = tl.sum(p, 0)
    alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)
    l_i = l_i * alpha + l_ij

    # 5. Update output compressor
    acc = acc * alpha + p

    # 6. Update global max
    m_i = m_ij

# And eventually, it's a homogeneity.
acc = acc / l_i
```