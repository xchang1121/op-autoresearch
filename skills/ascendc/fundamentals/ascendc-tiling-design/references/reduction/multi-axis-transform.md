# Multi-axis Convention Processing

> This document describes how to execute multiple A/R alternation sequences (e. g. ARAR) after the axis.
> For the rules of the axis, see [patterns.md](patterns.md# axis).

---

## Examples of actual operator

| operator | Shape | Axes | After the axis, Pattern. |
|------|-------|------|---------------|
| reduce_sum | [2,3,4,5] | [1,3] | **ARAR** [2,3,4,5] |
| reduce_sum | [2048,2,48,2,2,2] | [1,3,5] | **ARARAR**[2048, 2, 2 and 2] → Pad →**ARARARARAR**(8D) |
| bn_training_reduce | [N,C,H,W] | [0,2,3] | **ARAR** [1,N,C,H×W] |

## Typical scenario: BatchNomm, "Reserve C, Reunify N/H/W."

```
bn_training_reduce: Yeah. NCHW tensorAlong N,H,W Accession, reservation C ChannelDrill
  Input: [N, C, H, W]    axes=[0, 2, 3]
         R  A  R  R

  Axes: axes 2,3 Neighbored and same. R → Merge → R[N], A[C], R[H×W]
  Prefix A[1] → ARAR [1, N, C, H×W]

  Output: sum[C] and squareSum[C](double output, one value per channel)
  Characteristics: Double-output return — Same scan, same count. Σx and Σx², avoid going back and forth.
```

---

## Unified processing of Gauvi Pattern (≥5D)

If ≥ 5 is changed, it will be unified by PadDimonne() filling size = 1 dimension to**8 D (ARARARARARARARARARA) or 9 D (ARARARRARA)**:

```
5 V ARARA → pad to 9 V ARARARARA
6 V ARARAR → pad to 8 V ARARARAR
7 V ARARARA → pad to 9 V ARARARARA
8 V → Direct ARARARAR
9 V → Direct ARARARARA
```

---

## Kernel side execution method

The multiaxis Pattern expands into embedded loops, and each layer of R-axis moves independently AR or ARA determines that:

```
IterateInnerA<0, N>()   ← All over. A Axis (regression template, compilation period roll-out)
  for a0 in A_axis_0:
    for a1 in A_axis_1:
      ...
        LinearComputeR()  ← Deal with correspondence R Axis Return
          for r in R_axis:
            CopyIn → PreReduce → ReduceCompute → DoCaching → PostReduce → CopyOut
```

**AR/ ARA determination for each layer of R axis**: see if there is an A dimension on the right side of the R axis
- R Axis is the innermost (no A) →**AR mode**as detailed in the [ar-fullload.md] (ar-fullload.md) / [ar-colsplit.md] (ar-colsplit.md)
- And on the right side of the R axis is also the A dimension, →**ARA mode**, in Pattern::Reduce:RA, as detailed in [ara-fullload.md] (ara-fullload.md) / [ara-rowsplit.md] (ara-rowsplit.md)

**Example: ARR [2, 3, 4, 5] extended to:
```
for a0 in range(2):       ← A Axis 0
  for r0 in range(3):     ← R Axis 0On the right. A[4] → ARA mode)
    for a1 in range(4):   ← A Axis 1
      for r1 in range(5): ← R Axis 1(None on right) A → AR mode)
        Reduce(tile)
```

---

## Data handling in non-continuous multiaxis

The memory is often not continuous at the time of multiple axes (e.g. axes=[0,2] which allows the R axle to disperse the memory):

```
DAV_3510:    CopyInWithNddma() — Do-View. DMA Autoprocessing stride Jump.
DAV_2201: DataCopyPad of blockCount/blockLen/srcStride Parameter Configuration stride copy
       Or the outer circle. slice Removal (discontinuation intervals for cycle processing each time a continuous segment is moved)
```
