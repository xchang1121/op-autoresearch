# Online Softmax / Tiled Softmax Optimizing Design

## 1. Optimization of objectives

in AttentionIn the long sequence scene,SoftmaxThe input is complete.$QK^T$Matrix (scales$S \times S$).NaiveCompute complete matrix before line-by-lineSoftmaxYes, I do.$O(S^2)$Memory, long series belowSRAMCan't let go.

This optimization willSoftmax from"Quantified after full calculation"was replaced by"PVTileCalculate, maintainrunning max/sum"The incremental algorithm. The core idea:S2DirectionalTileCalculate$Q_i K_j^T$  Over  TileMaintain the maximum value currently seen$m$and cumulative$l$Output$O$Incremental update.**Memory Complexity From$O(S^2)$Down to$O(S)$.**

| Indicators | naive | optimized | Proceeds |
|------|-------|-----------|------|
| Intermediate Memory Occupancy | $O(S^2)$, complete$QK^T$Matrix | $O(S)$, only current file + running state | Substantial reduction of memory under long sequence |
| max/sum calculation | Double pass (max and sum) | **Simple pass**, spot update in tile | Natural Safe Softmax |
| Output Update | Last one-time prob × V | Weighted weights per tile | No visible distribution program matrix |

> operator applies: `softmax` (with variations such as `softmax`, `log_softmax`) and `flash_attention`, `sparse_flash_attention` embedded in Softmax.

## 2. Overview of the structure

### 2.1 Storage tiers and data flows

```
GM (Global Memory)
  │
  │ MTE2 (Q_i, K_j, V_j from GM → L1/UB)
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SRAM / UB                                                                  │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ Q_i Tile │ × │  K_j Tile^T │ = │  S_ij Tile   │   │  O_i (output)   │  │
│  │  [B,S1,D]│   │   [D,S2tile]│   │ [B,S1,S2tile]│ → │ [B,S1,D]        │  │
│  └──────────┘   └─────────────┘   └──────────────┘   └─────────────────┘  │
│         │                                              ▲                    │
│         │        ┌─────────────────────────────┐       │                    │
│         └───────→│ Online Softmax (Vector PIPE)│───────┘                    │
│                  │  running m: max so far       │                            │
│                  │  running l: sum of exp       │                            │
│                  │  O_acc: weighted output      │                            │
│                  └─────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  │ MTE3 (O_i, m_final, l_final → GM / workspace)
  ▼
GM (Output)
```

### 2.2 Tiled data stream

- **Outer cycle**: Tile ($j = 0, 1, \dots, N_{tile}-1), K, V, divided by a single block size $B$.
- **EveryTileInternal**: Calculate$S_{ij} = Q_i K_j^T / \sqrt{d}$, and then doOnline SoftmaxUpdaterunningStatus.
- **Cross Tile status**: $m$ (running max) and $l$$ (running sum) are transmitted between Tile as vector. Ascend is usually required to save the middle of S2 file in the Ascend scenario.
- **OutputOIncremental Update**: Current per roundtile of SoftmaxWeights$P_{ij}$Yeah.$V_j$Weighted, with correction factorrescaleAfter that, add to it.$O_i$.

### 2.3 Core mathematics: Online Softmax Single Pass formulae

Calculate the current Q block $Q_i and K block $K_j:

**Step 1 — Score**
$$S_{ij} = Q_i \times K_j^T / \sqrt{d}$$

**Step 2 — Local max and updating running max**
$$m_{new} = \max(m_{old}, \ \text{rowmax}(S_{ij}))$$

**Step 3 — Probability index (intermediate variable)**
$$P_{ij} = \exp(S_{ij} - m_{new})$$

**Step 4 — Update running sum**
$$l_{new} = l_{old} \times \exp(m_{old} - m_{new}) + \text{rowsum}(P_{ij})$$

**Step 5 — Output O Incremental Update**
$$O_i = O_i \times \frac{l_{old} \times \exp(m_{old} - m_{new})}{l_{new}} + \frac{P_{ij} \times V_j}{l_{new}}$$

Equivalent summary form:
```
m_new = max(m_old, tile_max)
sum_new = sum_old * exp(m_old - m_new) + tile_sum_exp(tile - m_new)
```
Eventually.Softmax:$\text{softmax}(x) = \exp(x - m_{final}) / sum_{final}$

### 2.4 Event Synchronization Model

| Event type | Meaning | Purpose |
|---------|------|------|
| `MTE2_V` | MTE2 Move complete → allows Victor to read | Q/K/V file data ready |
| `V_MTE3` | Victor complete → to allow MTE3 to write back | O incremental calculation completed, writeable GM |
| `V_V` | Victor complete → to allow Victor to continue | Online internal dependencies |

## 3. Key Parameter Configuration

```cpp
// Host side TilingData
struct OnlineSoftmaxTiling {
    uint32_t B;            // tile Block Size (Uniform Splitting) K,V), usually 64
    uint32_t D;            // head dimension
    uint32_t seqLen;       // Sequence Length S
    uint32_t s2TileNum;    // S2 Direction tile Number = ceil(S / B)
};

// Kernel side running state (one per line)
TBuf<QuePosition::VECCALC> runningMaxBuff;   // m: running max
TBuf<QuePosition::VECCALC> runningSumBuff;   // l: running sum
TBuf<QuePosition::VECCALC> outAccBuff;       // O_acc: Weighted output loader
```

### 3.1 Tile Size Selection Principle

| Parameters | Typical value | Annotations |
|------|--------|------|
| $B$ | 64 / 128 | Harmonizes the size of the tile block, dividing K, V along S2 dimensions. Set UB required |
| D | 64 / 128 | He's dead. It's a model. |

**Ascend Cube Particle Level Alignment Constraint**:

Tile must be sized to fit the matmul particle size of the Cube unit, usually**16×16 or 32×32**. If $B$ is not aligned, CUBE core needs additional padding.

```
B = align_up(preferred_B, cube_granularity)   // typically 16 or 32
```

### 3.2 Memory budget

Online SoftmaxUse memory complexity from$O(S^2)$Down to$O(S)$,but score tile($B \times B$),Q tile,K/V tileIt's also there.UBto ensure that the total occupancy is less thanUBcapacity. Typical configuration is about65KBIt's safe.

## 4. Core Calculator Cycle

### 4.1 naive version (before optimization)

```cpp
// Phase1: Complete calculationQK^T(S×SMatrix) requiredO(S^2)Space
for (uint32_t i = 0; i < S; i++) {
    for (uint32_t j = 0; j < S; j++) {
        scoreGm[i * S + j] = ComputeScore(QGm, KGm, i, j);
    }
}

// Stage 2: Line-by-line Softmax, twice (first max, then ext/sum)
for (uint32_t i = 0; i < S; i++) {
    float maxVal = -INFINITY;
    for (uint32_t j = 0; j < S; j++) {
        maxVal = max(maxVal, scoreGm[i * S + j]);
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < S; j++) {
        sum += exp(scoreGm[i * S + j] - maxVal);
    }
    for (uint32_t j = 0; j < S; j++) {
        probGm[i * S + j] = exp(scoreGm[i * S + j] - maxVal) / sum;
    }
}

// Stage 3: prob × V
for (uint32_t i = 0; i < S; i++) {
    for (uint32_t d = 0; d < D; d++) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < S; j++) {
            acc += probGm[i * S + j] * VGm[j * D + d];
        }
        OGm[i * D + d] = acc;
    }
}
```

### 4.2 Optimized version (after optimization): Tiled Online Softmax

```cpp
// Init: Distribute running state buffer (one per line)
pipe->InitBuffer(runningMaxBuff, B * sizeof(COMPUTE_T));   // m
pipe->InitBuffer(runningSumBuff, B * sizeof(COMPUTE_T));   // l
pipe->InitBuffer(outAccBuff, B * D * sizeof(COMPUTE_T));   // O_acc

LocalTensor<COMPUTE_T> mUb = runningMaxBuff.Get<COMPUTE_T>();
LocalTensor<COMPUTE_T> lUb = runningSumBuff.Get<COMPUTE_T>();
LocalTensor<COMPUTE_T> oUb = outAccBuff.Get<COMPUTE_T>();

// Initialising Running State
Duplicate(mUb, FLOAT_NEG_INF, B);   // m = -inf
Duplicate(lUb, 0.0f, B);            // l = 0
Duplicate(oUb, 0.0f, B * D);        // O = 0

// MTE2 Load Q_i file (fixed to UB)
LocalTensor<T> qUb = LoadQTile(QGm, i * B, B, D);

// S2 Orientation tile loop (Online Softmax core)
for (uint32_t j = 0; j < s2TileNum; j++) {
    // 1. MTE2 Loading K_j, V_j tiles
    LocalTensor<T> kUb = LoadKTile(KGm, j * B, B, D);
    LocalTensor<T> vUb = LoadVTile(VGm, j * B, B, D);

    // 2.CalculateS_ij = Q_i * K_j^T  (CUBE / Vector)
    LocalTensor<COMPUTE_T> sUb = ComputeScore(qUb, kUb, B, D);

    // 3. Online Softmax Update (see formula 2.3)
    OnlineSoftmaxUpdate(sUb, vUb, mUb, lUb, oUb, B, D);

    // 4. Optional: MTE3 writes intermediate results back to GM workspace (cross file status transfer)
}

// After all tiles: O_final = O_acc / l_final
ElemwiseDiv(oUb, lUb, B, D);

// MTE3 Write Back to Final Output
WriteBackOGm(OGm, oUb, i * B, B, D);
```

### 4.3 Online Softmax Update pseudocode (Vector Core)

```cpp
void OnlineSoftmaxUpdate(LocalTensor<COMPUTE_T> sUb,   // [B, B] score tile S_ij
                         LocalTensor<T> vUb,           // [B, D] V tile V_j
                         LocalTensor<COMPUTE_T> mUb,   // [B] running max
                         LocalTensor<COMPUTE_T> lUb,   // [B] running sum
                         LocalTensor<COMPUTE_T> oUb,   // [B, D] output acc O_i
                         uint32_t B, uint32_t D) {
    for (uint32_t r = 0; r < B; r++) {
        // Step 1: m_new = max(m_old, rowmax(S_ij))
        float m_old = mUb[r];
        float m_local = ReduceMax(sUb[r * B], B);
        float m_new = max(m_old, m_local);

        // Step 2: P_ij = exp(S_ij - m_new)
        // Step 3: l_new = l_old * exp(m_old - m_new) + rowsum(P_ij)
        float l_old = lUb[r];
        float scale = exp(m_old - m_new);
        float sum_exp = 0.0f;

        LocalTensor<COMPUTE_T> pUb = tempBuff.Get<COMPUTE_T>();
        for (uint32_t c = 0; c < B; c++) {
            float p_val = exp(sUb[r * B + c] - m_new);
            pUb[c] = p_val;
            sum_exp += p_val;
        }
        lUb[r] = l_old * scale + sum_exp;

        // Step 4: O_i = O_i * (l_old * scale / l_new) + P_ij * V_j / l_new
        float rescale_o = (l_old * scale) / lUb[r];
        float rescale_p = 1.0f / lUb[r];

        for (uint32_t d = 0; d < D; d++) {
            float pv = 0.0f;
            for (uint32_t c = 0; c < B; c++) {
                pv += pUb[c] * vUb[c * D + d];
            }
            oUb[r * D + d] = oUb[r * D + d] * rescale_o + pv * rescale_p;
        }

        mUb[r] = m_new;
    }
}
```

## 5. Key change points from live to online_softmax

| Modify Item | (before optimization) | Online_softmax (after optimization) |
|--------|---------------|------------------------|
| Calculator Paradigm | First $QK, then Softmax, then × V. | Calculate by tile, incrementally update running status |
| Memory Complexity | $O, full matrix required. | $O(S)$, only currenttile + running m/l/O |
| max/sum calculation | Double pass (max and sum) | **Simple pass**, spot update in tile |
| Numerical stability | Minus max requires extra walk-through | Natural Safe Softmax, per step minus current m_new |
| Output Update | Last one-time prob × V | Weighted weights of $P_Z1XQ\cdotV_j$ to O |
| Cross file status | None (one-time processing after full volume) | **running m, running l, O_acc**cross file transfer |

## 6. note/ Constraint

1. **Numerical stability: must be maintained. $m_ {new} = \\max (m_{old}, m_{local}) $, all index calculations are based on $m_ {new} $.

2. **O_accAmendment factor**When...$m_{new} > m_{old}$ when,$O_{old}$ and $l_{old}$Multiplication required$e^{m_{old} - m_{new}}$,scalemust be calculated before it is updated.

3. **Relation to GM workspace**. Flash Attention usually requires workspace to keep running m/l across the tile. This optimization is a reduction in memory at**Algorithmic level**and workspace is a cross-call status transfer at**Engineering level**.

4. **Last file boundary processing**. When $S$ is not an integer of $B$, the last file has a number of valid columns smaller than $B$, avoiding padding value interference.

5. **accuracy and performance balance**. $m and $l$ are recommended for FP32 maintenance, even if the input/output is FP16.

6. **Ascend Cube-VectorLoad Balance**.in Ascend 910BGo, go, go.$QK^T$Run onCUBEThe core,softmax and $P \times V$Run onVectorCore, both ends need to be taken into account when optimizing.

7. **Workspace MTE3 writing back is a significant cost**. Running states across S2 file usually require GM workspace, MTE3 writing back to workspace is one of the performance bottlenecks.

8. **Block-level Skip**for Causal Mask. Decoder scenarios can skip rough particles at the file level:
   - if $s2\_start > s1\_end$:**SKIP**The whole...block
   - If $s2 ended up with a starter:**FULL**block
   - Otherwise:**PARTIAL**block

   Theoretical overflight rate: $ (n-1)/(2n) \to 50\$. Actual: $S=32K approximately 45%, $S=2K approximately 37% and $S=512 about 22%.

9. **KV-Cache: Difference between Prefill vs Decode**.
   - **Prefill phase**: $Q$ is required to complete the sequence.
   - **Decode phase**: $Q$ length is 1 and does not need to be on $Q$ dimension tiling, only along the file of KV sequence dimensions, fully memory-born.

10. **Sliding Window Attention**. If a slide window is used (only)attendRecent$W$individualtoken),validConditional$i - W < j \le i$.Block-levelExtra Skip$s2\_end < s1\_start - W$ of block.

11. **pretokens / nextTokens parameters**. AscendAttention API defines attestation windows via `preTokens` and `nextTokens`, which are equivalent to the band mask. Configure needs to ensure logical consistency with causal / drafting window.

## 7. Selective decision-making and self-check list

### 7.1 Selective decision-making

```
if (operatorOrganisation softmax && Enter the length of the sequence S > 512):
    → Enable online_softmax( tiled realized)
    → B = 64 or 128( Alignment) Cube Gravity)
    → running Status FP32
else:
    → Standard Safe Softmax It's fine.S Small, full matrix. fit SRAM)
```

## 8. AscendC Kernel Optimization

That's what I'm talking about.PythonLayer-by-Classtileat the time of realizationPython → NPUMovement control costs and independencekernel launch,latencyData do not reflect the real merits of algorithms.**Production level optimization requiredAscendCLevels willQK^T,Online SoftmaxUpdate,PVMultipliers are all merged to a singlekernelInside.**

### 8.1 Key Crosses from Python Layer to AscendC

| Dimensions | Python Layer Achieved | AscendC Fused Kernel |
|------|--------------|---------------------|
| Movement control expenses | 16~128 Python Loop + Kernel lanch | **single kernel lanch**, zero Python |
| Score Calculator | `torch.matmul` Call Independently | Inline `Mul` + `ReduceSum` |
| Softmax | `torch.max`/`torch.exp`/`torch.sum` | `ReduceMax` → `Adds` → `Exp` → `ReduceSum` |
| P @ V | `torch.matmul` Call Independently | Inline scalar cum (V for row priority, list access needsstrided) |
| Parallelity | Watch dimensional serials | **Multi-block**: Each AICore handles one or more Qrows |
| Causal Skip | Python `if` judgement | **Block-level `continue`**Skip whole file |

### 8.2 AscendC Kernel Architecture

```
Every one AI Core Block Deal with one or more Q rows
├─ Load Q row [D] → UB
├─ Cast FP16 → FP32, Muls(scale)
├─ Init: m=-inf, l=0, O_acc=0
├─ For each S2 tile j:
│   ├─ Causal check: tileStart > s1Idx ? skip
│   ├─ MTE2: Load K tile [actualB, D] → UB
│   ├─ Vector: Cast K→FP32, Mul(q, k_slice), ReduceSum → score[actualB]
│   ├─ MTE2: Load V tile [actualB, D] → UB
│   ├─ Causal: mask upper part to -inf (if tile crosses diagonal)
│   ├─ Vector: ReduceMax(score) → m_local
│   ├─ Vector: Adds(score, -m_new), Exp → P[actualB]
│   ├─ Vector: ReduceSum(P) → sum_exp
│   ├─ Scalar: update m, l, rescale factors
│   ├─ Vector: Muls(O_acc, rescale_o)
│   └─ Scalar+Vector: P @ V column-by-column, accumulate to O_acc
├─ Cast O_acc → FP16
├─ MTE3: Write O row → GM
└─ MTE3: Write L = l_final → GM
```

### 8.3 Victor API Replace Map

```cpp
// Score: Q @ K^T (OriginalscalarLoop)
Cast(kFloat, kLocal, ..., actualB * D);
for (b = 0; b < actualB; b++) {
    Mul(tmp, qFloat, kFloat[b * D], D);       // Vector Element by Elements Multiplication
    ReduceSum(sumBuf, tmp, sumBuf, D);         // Vector Peace on the Statute
    score.SetValue(b, sumBuf.GetValue(0));
}

// Softmax: max → exp → Sum (formerly scalar Cycle)
ReduceMax(maxBuf, score, maxBuf, actualB);     // Vector The Statute wants maximum.
Adds(tmp, score, -mNew, actualB);              // Vector Element-by-Element
Exp(tmp, tmp, actualB);                        // Vector Index
ReduceSum(sumBuf, tmp, sumBuf, actualB);       // Vector Peace on the Statute

// Orescale (formerly scalar cycle)
Muls(oAcc, oAcc, rescaleO, D);                 // Vector Element by Elements Multiplicationscalar
```

### 8.4 Multi-nuclear parallel strategy

- **Total row**: `totalRows = batch × num_heads × S1`
- **Block Allocation**: `blockDim = min(totalRows, 8)` (8 AICore matching Ascend 910B)
- **Workload per Block**: `rowsPerCore = ceil(totalRows / blockDim)`
- **Flat index decomposition**: `row → (batchIdx, headIdx, s1Idx)`

### 8.5 UB Memory Budget (Optimized)

| Buffer | Purpose | D=64, tileB=128 | D=128, tileB=128 |
|--------|------|----------------|-----------------|
| qBuf | Q row (FP16) | 128 B | 256 B |
| kBuf/vBuf | K/V tile (FP16) | 2 × 16 KB | 2 × 32 KB |
| qFloatBuf | Q row (FP32) | 256 B | 512 B |
| kvFloatBuf | K/V float (time reuse) | 32 KB | 64 KB |
| scoreBuf/pBuf | Score / Prob (FP32) | 2 × 512 B | 2 × 512 B |
| oAccBuf | Output accumulator | 256 B | 512 B |
| tmpBuf/reduceBuf | Vector workspace | 2 × 512 B | 2 × 512 B |
| Total** | | **~100 KB** | **~135 KB** |

UB capacity 192 KB, both configurations secure. K/V float buffer time reuse saves 32~64 KB.

### 8.6 Expected performance

| Indicators | Python Layer Online | AscendC Fused |
|------|-----------------|---------------|
| S=512 | Slower than five 28x | **Projected fast 1.2~1.5x** |
| S=4K | Slower than positive 1.4x | **Projected approaching 1.5~3x** |
| S=32K | Slower than five 25x | **Projected fast 2~5x**(causal+skip) |
| Memory | O(S) | O(S) (same as the Python Layer) |

> **Key Conclusion**: AscendC Kernel eliminated Python movement costs, making the bandwidth advantage of Online Softmax (to save O(S²) intermediate matrix handling) truly translates into latency revenue.


### 7.2 Self-check List

- [ ] Running m to `-inf`, runningl to `0`
- [ ] Update with `m_new = max(m_old, m_local)`
- [ ] $scale = e^{m_{old} - m_{new}}$Updatingl and OCalculated before
- [ ] O_ac and runningl use FP32 even if input output is FP16
- [ ] Last file boundary processing: Number of valid columns = `min(B, seqLen - j * B)`
- [ ] Causal Mask scene: whole Masked file correctly skip update
- [ ] Validation by accuracy: error < 1e-5 (FP32) or < 1e-3 (FP16), compared to naive Safe Softmax
- [ ] AscendC kernelUseVector APIAlternativescalarLoop`GetValue`/`SetValue`Only if necessary)
- [ ] Multi-block Distribution Qrows, No Repeats / Missing
- [ ] UB RAM < 192 KB (Ascend 910B)
