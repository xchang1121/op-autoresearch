# Access Bund Optimizing Policy

## Conditions for determination

- MTE2 High utilization rate (AIC bandwidth approaching theoretical peak)
- The Victor/Cube unit has an empty bubble waiting for data
- Calculating low density (ops/ Byte below hardware capacity)

## Elements of a simulation map analysis

- View the overlap between MTE2 moving time lines and calculating time lines
- Identification of large data volume migration windows to mark small migration sequences that can be merged or optimized

---

## Policy 1: bandwidth is not filled with → L2 reuse

| Operation | Annotations |
|------|------|
| Check the current bandwidth utilization factor | Compare `actual bandwidth / theoretical peak bandwidth ' |
| L2 Cache Presence Optimization | Resize the file to keep the data in L2, less double read DDR |
| Multinucle L2 Sharing | Neighbourhood nuclear processing of adjacent data, sharing of L2 cache lines |

## Policy 2: Command Efficiency / Select Basic Blocks

| Operation | Annotations |
|------|------|
| Analysis of MTE2 command launch efficiency | Check for redundant removal instructions. |
| Select Basic Blocks | Prefer to a continuous address + a large particle transfer command |
| Reduce stride move | stride move efficiency is lower than continuous move and reset data |

## Policy 3: Reduce small loads / Merge

| Operation | Annotations |
|------|------|
| Identification of small loads | recognition of less than threshold moves in projecting |
| Merge continuous blocks | Merge multiple adjacent small moves into a single large move |
| DataCopyPad Parameter Optimization | Adjusting alignment parameters to avoid debrisation and removal |
| Unnecessary removal | Check for in situ consumption of UB data to reduce in- and out-migration |

### Continuous block priority

operator for each element scalar often becomes a memory base. As long as the index forms a continuous segment within a local window, priority is given to consecutive block moving:

```cpp
// Discrepancies: Each element triggers a GM scalar visit.
for (int32_t i = 0; i < len; ++i) {
  float v = xGm.GetValue(srcBase + i);
  yGm.SetValue(dstBase + i, v);
}

// All right: Move over to UB, write back again.
DataCopy(tileLocal, xGm[srcBase], len);
DataCopy(yGm[dstBase], tileLocal, len);
```

If only tail is unmatched, the main path should still use normal `DataCopy`, tail with a single `DataCopyPad`:

```cpp
int32_t aligned = len / elemsPer32B * elemsPer32B;
if (aligned > 0) {
  DataCopy(local, gm[offset], aligned);
}
if (aligned < len) {
  DataCopyPad(local[aligned], gm[offset + aligned], len - aligned, padParams);
}
```

### UB Inline Reuse

When the same input or coefficient is consumed at multiple stages, priority is given to remaining in the UB, so do not write back to GM and read:

```cpp
DataCopy(xLocal, xGm[rowBase], D);
DataCopy(xCache, xLocal, D);        // pass 2 Reuse
ReduceSum(sumLocal, Square(xLocal), tmp, D);

float inv = Rsqrt(sumLocal.GetValue(0) / D + eps);
Muls(xLocal, xCache, inv, D);
Mul(outLocal, xLocal, gammaLocal, D);
DataCopy(yGm[rowBase], outLocal, D);
```

This type of reuse is most suitable for normalization, rotary, softmax, multi-pass correction; if D exceeds UB, change the ingredient file size or only cache a small coefficient.

## Tiling Amendments

- Resize UB file to increase L2 hit rate
- Optimizing data transfer particle size and alignment parameters
- Adjusting polynuclear cut-off to reduce mononuclear data
- For multiple output or small result scenarios, cumulative multi-line/multi-part results in UB, and once again Copyout, avoiding a lower-case return for each line.
