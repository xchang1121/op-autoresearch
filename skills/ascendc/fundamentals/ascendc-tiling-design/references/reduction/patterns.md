# Reduction-type operator scene route

> This document is used for**scene determination**and**policy selection**. Once the scene is determined, you enter the corresponding detailed document by link.

---

## Axes

Simplify the N-Vape +axes to less dimensions. Each dimension is marked as**A**(retention axis) or**R**(contracting axis) and then:
1. **Elimination of redundancy**: size = 1 and memory continuity dimension elimination
2. **Consolidation of adjacent congener type axes**: adjacent dimensions are A (or both R) → amalgamation (multiplier)

**Example:**
- Shape = [2,100,4], axes = [1,2] → Mark A, R, R → adjoining R Merge → [2,400] = (A, R) → Single axis AR
- Shape = [2,3,4,5,6,7,8], axes = [1,2,4] → Mark A, R, R, R, A, A → adjoining R Merged → [2, 12, 5, 6, 56] = (A, R, R, A) → Polyaxis

---

## scene determination process

```
Organisation: shape, axes (Axis of Return)

Step 0: Combined axis (see previous section)
  Tags A/R → Eliminate redundancies → Merge adjacent type axes

Step 1: Is it a single axle or a multiaxis?
  ├─ Single axle(s)AR or ARA)→ Step 2
  └─ MultiaxisARAR Reciprocal sequences)→ Shape Three-step transformation (for more details) multi-axis-transform.md)
       Expand into embedded loops, each R Viv's contract of return. Step 2 Decision

Step 2: A0 Decision modalities
  ├─ A0 = 1 → AR Mode (per line) R An element of continuity. Use Level 2 Reduce API)
  │   ├─ Yes.UBprocessing at least1Line Data → AR-FullLoad    → [ar-fullload.md]
  │   └─ Otherwise... → AR-ColSplit    → [ar-colsplit.md]
  │
  └─ A0 > 1 → ARA Mode[R,A0] Blocks continuous, use Pattern::Reduce::RA)
      ├─ Yes.UBprocessing allR*tileA0LenData (%1)32Byte Alignment) → ARA-FullLoad  → [ara-fullload.md]
      └─ Otherwise... → ARA-RowSplit  → [ara-rowsplit.md]

Common exceptions. AR or ARAPress up:
  - axis=0(extreme dimension)→ A1=1, ARA Mode
  - All-axis return.        → Merge all dimensions into R, A0=1, AR Mode
  - Norm Category (substance)+Radio conversion)→ Part of the contract. AR/ARAThe transformation is applied layer logic.
```

**Positive dimension**(selected on the basis of operator characteristics and data size after determination of branch):

| Dimensions | Options | Conditions of application | Detailed documents |
|------|------|---------|---------|
| Algebra Selection | Welford Online | Split + Require flow to calculate two related statistics | [algorithms.md](algorithms.md#2-welford-online-alpha one time) |
| Multi-nuclear strategy | Group Reduce | R Too big for mononuclear processing + A too small for parallel | [algorithms.md](algorithms.md #3-group-reduce cross-cutting Convention) |
| accuracy policy | Half plus | Great vector sum accuracy is sensitive. | [algorithms.md](algorithms.md#5-dichotomy-addition) |
| Index Tracking | With-Index | Reunification + Return polar position | [with-index.md](with-index.md) |

---

## General rules

The following rules apply to all scenarios. The Tiling Design Principles and the tmpBufSize formula are in [tiling-fields.md] (tiling-fields.md).

**RLength vs rLengthAlign parameter used in comparison table**:

| Parameter Position | Use rLength (valid data) | Use rLengthAlign (after alignment) |
|---------|:---:|:---:|
| DataCopyPad blockLen | ✅ | ❌ |
| Reduce API count(Level 2) | ✅ | ❌ |
| UB internal rowofset calculation | ❌ | ✅ |
| Buffer Size Distribution | ❌ | ✅ |

---

## S1: AR mode (best possible return)

Application: A single-axis return with a condensed tail axis

R elements in each row continue, using Level 2 Reduce API line-by-line return.

**AR branch decision tree**:

```
AR Mode (A1, R),A0 = 1
    │
    ├─ Yes.UBprocessing at least1The whole line of data?
    │   │
    │   YES → AR-FullLoad(All)
    │   │     Line R Elemental presence UB,CopyIn We'll finish our contract at once.
    │   │     The intermediate result is directly reused and does not need to be duplicated
    │   │
    │   NO  → AR-ColSplit(Distributed)
    │         Loading of column-direction sub-contracts, every move in Ar < AR An element
    │         Need to cross chunk MergeMax/Add)
    │
    └─ Both models are used. Level 2 Reduce API(data continuous, not required) Pattern)
```

| Branch | Conditions | Annotations | Detailed documents |
|------|------|------|---------|
| **FullLoad** | You can process at least one whole row of data in UB | Row-wide UB, intermediate result directly reused | [ar-fullload.md](ar-fullload.md) |
| **Partition (ColSpit)** | Other | Column directional segment, each time you move to Ar < AR elements, merge over chunk | [ar-colsplit.md](ar-colsplit.md) |

> **Index variant**: AR-FullLoad uses `ReduceMax(calIndex=true)`, as detailed in [with-index.md] (with-index.md), if return to polar position is required.

> **According to scalar broadcasting operation**: Level 2 Reduce contract [R] received 1 scalar.
> If follow-up needs to broadcast scalar to [R] vector involved in element-by-fact calculations, use `Adds(dst, src, scalar, count)`
> or `Muls(dst, src, scalar, count)`, an API call complete to replace Duplicate fill + Add/Mul operation.
> More about `/ascendc-api-best-practices`'s `api-arithmetic.md`.

---

## S2: ARA Mode

Application: One-axis return and one-axis non-axis scenario

> **Core Cognizance**: ARA mode `(A1, R, A0)`, after split along outer layer A1, handles one `[R, A0_inner]` block each time.
> The block continues in the GM, and the whole block moves into the UB, which is the two-dimensional matrix of `(R, alignedCols)`.

**Data stream**:
```
GM (A1, R, A0) → DataCopyPad(blockCount=R) → UB (R × alignedCols)
    ↓
[ReduceMax/ReduceSum Pattern::Reduce::RA, srcShape={R, alignedCols}] → result (alignedCols)
    ↓
UB (alignedCols) → GM (A1, A0)
```

**Key API Call**:
```cpp
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);
uint32_t srcShape[] = {R, alignedCols};
ReduceMax<float, Pattern::Reduce::RA>(resultLocal, xLocal, tmpLocal, srcShape, true);
```

**ARA branch decision tree**:

```
ARA Mode (A1, R, A0),A0 > 1
    │
    ├─ Yes.UBprocessing allR*tileA0LenData (%1)32Bytes Alignment)
    │   │
    │   YES → ARA-FullLoad(All)
    │   │     [R, tileA0Len] Entire Stay UB,CopyIn All at once. R The return of the line
    │   │     The intermediate result is directly reused and does not need to be duplicated
    │   │
    │   NO  → ARA-RowSplit(Distributed)
    │         Every move in r < R All right, let's split it up a few times before we finish it all. R The return of the line
    │         Need to cross chunk MergeMax/Add)
    │
    └─ Both models are used. Pattern::Reduce::RA  Don't need   Transpose)
```
- TileA0Len is a0 for each nuclear treatment following the polynuclear splitting of A0 (<=A0)

| Branch | Conditions | Annotations | Detailed documents |
|------|------|------|---------|
| **FullLoad** | You can process all R*tileA0Len data in UB (32 bytes alignment) | R-line once in UB, intermediate result directly reuse | [ara-fullload.md](ara-fullload.md) |
| **RowSpit** | Other | Cross chunk merger after each move to r < R row | [ara-rowsplit.md](ara-rowsplit.md) |

> **Index variant**: `Pattern::Reduce::RA` is replaced with `Compare+Select` line-by-line inverted positions that need to be returned.
> For details, see [with-index.md] (with-index.md).

> **ARA radio operation upon contract**:Pattern::Reduce::RA contract [R, signedcols] and then get the result of [1, signedcols] vector.
> If follow-up needs to broadcast the vector back [R, signedCols] participating in element-by-fact calculations, using the sub/Div/Mul equivalent binary API
> Binary RepeatParams version, setting up `src1RepStride=0`, an API call to finish all R lines,
> No manual cycle or extra broadcast buffer is required. See `api-arithmetic.md` for details of `/ascendc-api-best-practices`.

### S3: Multi-axis Convention

Application: Multi-axis return, e.g. ARARA scene

**Shape transformation**: Any N-Vipe +axes is compressed by a three-step transition to an A/R alternation sequence.
For details, see [multi-axis-transform.md] (multi-axis-transform.md).

### S4: Welford Online (R requires UB slices)

**Appendix**: two relevant statistics will need to be calculated in a fluid fashion under the ARA model (the second relies on the first incremental update). Typical operator: reduce_var / reduce_std. See [algorithms.md](algorithms.md#2-welford-online-on-line one time).

### S5: Group Reduce(R cross-nucleus)

**Conditions**: R is too big to complete all R returns; and A is too small to make full use of the polynuclear

```
Phase1(nuclear independence): Per nuclear handling A[Paragraph] × R[Paragraph]Output partial → workspace
SyncAll()
Phase2(consolidated): All over the place. partial, merge into final result
```

Workspace: `coreNum × CeilAlign(outAAlign × 2 × sizeof(int32_t), 256)`

---

## S6: Global return

Application: reduce_sum (axes = all axes), reduce_max (axes = all axes)

All elements are divided on a nuclear equal basis and are merged (Atomic or Visible Merge) in two stages after the individual nuclear reunification.

```
Stage1: All nuclear facilities ReduceSum(mySlice) → partial → workspace[blockIdx * 64B]
Stage2:
  ModalitiesA: SetAtomicAdd → DataCopy → SetAtomicNone → SyncAll
  ModalitiesB: SyncAll → core0 Walking through workspace Merge
```

---

## Cross-Scene Reference

| Theme | Documentation |
|------|------|
| Multiple Output Buffer Equation | [multi-output-buffer.md](multi-output-buffer.md) |
| Universal Tiling Field Definition | [tiling-fields.md](tiling-fields.md) |
| Index tracking variants | [with-index.md](with-index.md) |
