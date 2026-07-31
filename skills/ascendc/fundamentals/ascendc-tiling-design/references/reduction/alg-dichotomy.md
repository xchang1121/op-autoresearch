# Dichotomy Addition / Half-Interval

**Applicable scene**: Sum is dedicated to solving the accuracy problem of the order plus the large and medium edible

******: When the sequence adds `sum = a1 + a2 + a3 + ...`, when the sum is already large and the subsequent element is small, the decimals will be "eat" because of the floating point accuracy.

**Rationale**: Summation with a dident tree structure to add numbers at comparable scales.

## Core algorithm

```cpp
float DichotomyReduceSum(LocalTensor<float>& src, int count) {
    // Step 1:Found the biggest2^k ≤ count
    int powerTwo = FindNextPower2LessEqual(count);

    // Step 2: End folding
    int tail = count - powerTwo;
    if (tail > 0) {
        Add(src, src, src[powerTwo], tail);
        // Variant (Half-Interval): protection of tail with mask
        // Add(src, src, src[powerTwo], GenMask(tail));
    }

    // Step 3: Double folding
    while (powerTwo > 64) {
        powerTwo /= 2;
        Add(src, src, src[powerTwo], powerTwo);
    }

    // Step 4: HoleReduceSum Hardware Command (≤64 Element)
    WholeReduceSum(result, src, powerTwo);
    return result;
}
```

## Compare with direct ReduceSum

| In terms of | Order Sum | Half plus |
|------|-------------------|---------|
| accuracy | Eat a lot. | accuracy is better than that. |
| Apply Operation | Sum only | Sum only (Max/Min is not affected) |
| UB Expenditure | No Additional | In situ operation, no extra buffer |
| Typical scene. | R ≤ VL, even scale | R > VL, big difference in volume |

## Multiline version (MergeN mode)

Recapitulation of multiple lines with recapitulation (`reduce_common.h` from rms_norm / player_norm):

```cpp
void ReduceSumMultiN(LocalTensor<float>& src, int numRows,
                     int colsPerRow, int stride) {
    uint64_t rptCfg = BuildRepeatConfig(numRows, stride);
    WholeReduceSum(dst, src, rptCfg);
}
```

