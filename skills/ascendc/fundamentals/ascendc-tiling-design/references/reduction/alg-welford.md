# Welford Online Algorithm (one-time online)

**Applicable scenario**: two related statistical volumes (e.g., meaning + variance) need to be calculated in flow mode under the split mode and a single scan is completed.

**Strength**: single scan (vs TwoPass twice), good numerical stability, support for parallel group consolidation.

---

## Core Update Formula

```
Initial: mean = 0, M2 = 0, count = 0

To Every New Element x:
    count += 1
    delta1 = x - mean           ← Old deviation
    mean = mean + delta1/count  ← Incremental update average
    delta3 = x - mean           ← New deviation
    M2 = M2 + delta1 * delta3   ← Incremental Update Square

Eventually.: var = M2 / (count - correction)
```

## Two combinations and formulae

When more than one nuclear or group calculates each part (mean, M2, count), it is merged into a global result:

```
Merge (mean_a, M2_a, count_a) and (mean_b, M2_b, count_b):

count_total = count_a + count_b
delta = mean_b - mean_a
mean_total = mean_a + delta * count_b / count_total
M2_total = M2_a + M2_b + delta² * count_a * count_b / count_total
```

## Group Welford (groups merged)

When the number of split chunks is large, every eight chunks are combined in an intermediate way to prevent the accumulation of floating points error.

## vector Welford Update (AscendC achieves example)

```cpp
// Welford Updates data on a UB chunk
void WelfordUpdate(LocalTensor<float>& x, int curLen,
                   LocalTensor<float>& mean, LocalTensor<float>& M2,
                   int& count) {
    for (int i = 0; i < curLen; i++) {
        count++;
        float scale = 1.0f / static_cast<float>(count);

        // Delta1 = x - means (vector: whole A-D)
        Sub(delta1Buf, x[i * A_aligned], mean, A_aligned);

        // mean = mean + delta1 * scale
        Muls(tmpBuf, delta1Buf, scale, A_aligned);
        Add(mean, mean, tmpBuf, A_aligned);

        // delta3 = x - mean_new
        Sub(delta3Buf, x[i * A_aligned], mean, A_aligned);

        // M2 = M2 + delta1 * delta3
        Mul(tmpBuf, delta1Buf, delta3Buf, A_aligned);
        Add(M2, M2, tmpBuf, A_aligned);
    }
}
```

## Selection with TwoPass

| Conditions | Recommendations |
|------|------|
| FullLoad + Double-order return | TwoPass |
| Split + Relevant Conventions | Welford (single flow, round IO) |
