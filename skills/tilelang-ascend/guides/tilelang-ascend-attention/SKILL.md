---
name: tilelang-ascend-attention
description: "TileLang Ascend Action operator coding guide. Covers Cube+Vector integration programming paradigms, workspace cross-nuclear communication, online softmax cumulative mode. Reference is made to this guidance when generating the Attention class operator."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "attention"
---

# TileLang Ascend Action operator Encoding Guide

---

## Cube+Vector Integration Programming Model

AttentionIt's typical.Cube+VectorIntegration scene:QK^TUse it.CubeNuclearGEMM),softmax/reduceUse it.VectorNuclear,PVUse it.CubeNuclearGEMM).

**Data stream**: Cube and Vector communicate by**workspace**, creating a cycle of Cube writing → Victor reading → Victor writing → Cube reading.

**Key points**:
- `workspace_idx=[4, 5, 6, 7, 8]` states workspace tensor in `@tilelang.jit`
- Enable `AUTO_CV_COMBINE + AUTO_CV_SYNC` autoprocessing C/V integration and synchronization in Devloper mode
- Cube field must have `T.barrier_all()` sync C/V nuclear

---

## Workspace Cross Nuclear Communication

Workspace uses the `@tilelang.jit` declaration in `workspace_idx` to transmit intermediate results between the Cube nuclear and the Vector nuclear.

**Typical workspace use**(for example, sparse_flash_attention):
- `workspace_1/2`:KVData,VectorOrganisation→ CubeOrganisation
- `workspace_3`:QK^TOutcome (%)acc_s),CubeOrganisation→ VectorOrganisation
- Attention weight after `workspace_4`:softmax, Victor writes to → Cube
- `workspace_5`:PV result (acc_o), Cube nuclei write to → Victor

**Required pass_configs**:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

---

## Online Softmax Cumulative Mode

Attention's softmax needs a segment calculation (KV sequence too long to be dropped into UB at a time), using the online softmax algorithm to maintain three running state: `m_i` (maximum of line), `sumexp` (line index and row index), and `acc_o` (output cumulative).

**Algorithm Process**:

```
Initialize: m_i = -inf, sumexp = 0, acc_o = 0

For each of them. KV Blocks:
  1. from workspace Read Cube Calculated acc_s
  2. m_i_new = max(m_i_prev, max(acc_s))
  3. correction = exp(m_i_prev - m_i_new)
  4. sumexp = sumexp * correction + sum(exp(acc_s - m_i_new))
  5. acc_o = acc_o * correction + acc_s_normalized @ V_partial
```

**corpion factor decomposition**: code to split `exp(m_prev - m_new)` into three steps:
1. `m_i_prev[i] = m_i_prev[i] - m_i[i]` → `m_i_prev[i] = T.exp(m_i_prev[i])` calculation
2. `sumexp[i] *= m_i_prev[i]` fixes history sumext
3. `acc_o[h_i, j] = acc_o[h_i, j] * m_i_prev[h_i]` fixes history output

---

## Encoding Elements

1. **L0C cannot directly do reduce**: `T.copy` must go to UB before softmax returns
2. **init Parameter**: First block `init=True` Zero L0C, subsequent block `init=False` add
3. **correaction factor**: `exp(m_prev - m_new)` in online softmax is key to numerical stability and cannot be omitted
4. **SEQ_LEN non-match**: Handled with `T.ceildiv`, not presumed division
5. **Developer mode priority**: Automatically synchronize with `pass_configs` + unless extreme performance is required to avoid handwritten `T.set_flag`/`T.wait_flag`
6. **workspace index**: First dimension of using `cid` (kernel id) index workspace to ensure that different nuclei are written to different regions
7. **v_blockSeveration**:VectorOrganisation`vid`Split ProcessL0C  All the lines, every one  VectorNuclear handling`v_block = H_per_block // 2`Okay.
