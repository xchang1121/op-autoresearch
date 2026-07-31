# Broadcast-type operator scene route

> This document is used for**scene determination**and**policy selection**. Once the scene is determined, you enter the corresponding detailed document by link.

---

## Dimension Collapse

The axis is a public prelude to all Broadcast branches with the aim of reducing dimensions to simplify the Kernel cycle.

### Rule

1. **Supplive-Divine**: Add 1 to the left when input size is not enough for output
2. **Marked Radio Axis**: calculates flag bitmap for each axis, and enter j into the axis dim=1 → flag set 1
3. **Merge adjacent flag axes**: adjacent two axes if all input flags are the same → merge (multiple dimensions)
4. **Calculated stride**: right-to-left multiplier; broadcast axis (dim=1 and output dim>1) length 0

### Example:

**Add(x=[4,3,8], y=[1,3,8])**:
```
After the supplement:x=[4,3,8], y=[1,3,8], out=[4,3,8]

flag Calculating2 One input, one input.bit0=y, bit1=x):
  Axis 0: x=4≠1, y=1 → flag=01 (y We need to broadcast.)
  Axis 1: x=3≠1, y=3≠1 → flag=00
  Axis 2: x=8≠1, y=8≠1 → flag=00

Axis: Axis 1 And the axes. 2 flag Both. 00 → Merge
  x: [4, 24]    strides: [24, 1]
  y: [1, 24]    strides: [0,  1]   ← Axis 0 stride=0It needs to be broadcast.
  out: [4, 24]  strides: [24, 1]
```

**Mul(x=[2,1,4], y=[2,3,1])**:
```
flag Calculate:
  Axis 0: flag=00
  Axis 1: x=1 → flag=10 (x We need to broadcast.)
  Axis 2: y=1 → flag=01 (y We need to broadcast.)

Axis: Axis 0 of flag=00Axis 1 of flag=10 → Different, do not merge
  x: [2, 1, 4]    strides: [4, 0, 1]
  y: [2, 3, 1]    strides: [3, 1, 0]
  out: [2, 3, 4]  strides: [12, 4, 1]
```

---

## scene determination process

```
Organisation: N Individual Inputs shape + 1 Output shape(all inputs continuously)

Step 0: Complementary + Combined axis (see previous section)
  Got it. dims[N+1][≤8] + strides[N+1][≤8]

Step 1: Branch determination
  ├─ There's only one after the axis. 1 V → OneDim  Pure ElementwiseMaybe.scalarEnter) → [onedim.md]
  │
  └─ More than one trailing dimension → Select a broadcast implementation
      ├─ DAV_2201 → UB Broadcast(Prior static interface, removal command option when alignment is not met) → [ub-broadcast.md]
      │
      └─ DAV_3510 → What stage did the broadcast take?
          ├─ GM→UB Moving to phase → See the decision chain below.
          │
          └─ UB Internal broadcasting (intermediate calculations required broadcasting):
              → UB Broadcast Dynamic interface (Voice interface)rank 1~9) → [dynamic-ub-broadcast.md]
```

### Decision-making chain for broadcast input (DAV_3510)

| Priority | Conditions | Selection |
|--------|------|------|
| 1 | Force user to specify NDDMA or UB BRC | Compliance |
| 2 | NLast scene, tail axis > = dcache/2 | UB BRC → [dynamic-ub-broadcast.md] |
| 3 | dtype is INT8/FP16/BF16 and 32B alignment | UB BRC → [dynamic-ub-broadcast.md] |
| 4 | Other | NDDMA → [nddma-broadcast.md] |

**NLast**= tail axis does not need to be broadcast (stride≠0), but non-tail axis does need to be broadcast (stride=0).

---

## General rules

The following rules apply to all Broadcast branches.

**Description of variables**:

| Variables | Meaning |
|------|------|
| `dims[i]` | Post-axis output size of i-dimensional of sape |
| `shapeLen` | Total dimensions after axle |
| `ubSize` | UB Total Capacity (bytes) |
| `extraSize` | Extra reserved space (bytes), e. g. tmpBuffer, etc. |
| `bufferNum` | Calculates the number of Buffer survives in the figure (input + output + middle) |
| `maxDtypeBits` | Calculate the width of the maximum data type in the figure |
| `minDtypeBits` | Calculate the width of the smallest data type in the figure |

**UB split**: multiplication from the innermost axis to find the first indisposable axis as ubSplitAxis:
```
curProduct = 1
ubSplitAxis = 0
allFit = true
for i = shapeLen-1 downto 0:
    curProduct *= dims[i]
    if curProduct > maxElemNum:
        ubSplitAxis = i
        curProduct /= dims[i]
        allFit = false
        break

if allFit:                              # All dimensions fit in. UB,ubSplitAxis Keep Initial Value 0
    curProduct /= dims[0]               # Slice at the outermost level.

if shapeLen == 1:                       # Single-dimensional scene (usually already) OneDim Interception)
    ubFormer = maxElemNum
else:
    ubFormer = maxElemNum / curProduct

ubOuter = ceil(dims[ubSplitAxis] / ubFormer)
ubTail  = dims[ubSplitAxis] - (ubOuter-1) * ubFormer
```

Where maxElemNum calculates:
```
maxElemNum = (ubSize - extraSize) * 8 / (bufferNum * maxDtypeBits)
maxElemNum = floor_align(maxElemNum, 256 * 8 / minDtypeBits)
```

**Polynuclear cut**: ubSplitAxis and its outer layers are equal to the polynuclear:
```
fusedProduct = ubOuter × (ubSplitAxis All axes before)
blockFormer  = ceil(fusedProduct / coreNum)
blockNum     = ceil(fusedProduct / blockFormer)
blockTail    = fusedProduct - (blockNum - 1) * blockFormer
```

When nuclear power is insufficient (`blockNum < coreNum`), the cycle is reduced by maxElem Num (by CACHE_LINE) and recalculated by ubFormer /ubOuter/fusedProduct until more cores can be fed.

**Alignment**:
- OneDim: 128B (CACHE_LINE) Alignment
- Multi-dimensional: 256B (REPEAT) Alignment

**NDDMA processing over 5 axis (DAV_3510)**:
```
axesAfterSplit = shapeLen - ubSplitAxis

≤ 5 → WITHOUT_LOOP: Once NDDMA Call completed (%1)API as DataCopy<T, 5, config>)
> 5 → WITH_LOOP: Inside 5 Give me the axle. NDDMAouter axle Kernel for-loop Walking through
```
In the WITH_LOOP mode, NDDMA config is the same, only GM/UB is offset by the outer circle. See [nddma-broadcast.md] (nddma-broadcast.md) for further details.

---

## Cross-Scene Reference

| Theme | Documentation |
|------|------|
| OneDim branch (one-dimensional after-axis) | [onedim.md](onedim.md) |
| UB Broadcast (DAV_2201, static interface + moving command fallback) | [ub-broadcast.md](ub-broadcast.md) |
| UB Broadcast Dynamic Interface (DAV_3510, rank 1-9) | [dynamic-ub-broadcast.md](dynamic-ub-broadcast.md) |
| NDDMA Broadcast (DAV_3510, GM→UB hardware broadcast) | [nddma-broadcast.md](nddma-broadcast.md) |
