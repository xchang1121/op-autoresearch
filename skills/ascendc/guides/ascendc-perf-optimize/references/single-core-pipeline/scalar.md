# Scalar Base / Small Case Optimizing Policy

## Conditions for determination

- Total calculated amount is very small (e. g. element count < threshold)
- Scalar commands are high, Victor/Cube units are idle
- Low command launch rate, low IPC

## Elements of a simulation map analysis

- Positioning Scalar command for the larger time window
- Recognize unnecessary sync waiting between Scalar and Victor

## Optimizing Policy

| Policy | Operation | Effects |
|------|------|------|
| **Scalar Optimization** | Reduce redundancy scalar calculation, merge condition branches | Decrease the number of scalar commands |
| **Circle development** | Start small cycles to reduce branch costs | Raise IPC |
| **Reducing circulation axes** | Simplified Loop Axes by Tiling | Lower scalar |
| **Command Selection** | Use efficient scalar command instead of inefficient sequences | Shorten critical path |
| **Reductionscalar-vectorConvert** | Avoid unnecessary Scalar ↔ Victor data migration | Reduce removal costs |
| **Use of performance-friendly API** | Replace Queue with set_frag and wait_frag, use LocalTensor instead of Tbuffer and remove Tpipe | Scalar with reduced seals |

## Operational Mode

### 1. Move index algorithms out of the inner layer

When `div/mod`, multi-stage stride multiplication, Shape branch appears in the inner layer, they are usually first transformed into a catch level variable or a host tilling field:

```cpp
// Difference: Each element repeats the linear subscript.
int64_t n = linear / (C * H * W);
int64_t c = (linear / (H * W)) % C;
int64_t h = (linear / W) % H;
int64_t w = linear % W;

// Good: Push on a straight line.
int64_t rowBase = ((n * C + c) * H + h) * W;
for (int32_t w = wStart; w < wEnd; ++w) {
  Compute(rowBase + w);
}
```

If the range is sufficient, the inner layer counter will give priority to `int32_t`/`uint32_t`, reducing the 64-bit integer command pressure.

### 2. Quantified small results

In scenarios such as aragmax, cros entropy, Foreach norm, only one or more elements per line. Do not separate copy from each row:

```cpp
constexpr int32_t BATCH = 32;
auto outLocal = outBuf.Get<int64_t>();

for (int32_t base = 0; base < rows; base += BATCH) {
  int32_t n = Min(BATCH, rows - base);
  for (int32_t i = 0; i < n; ++i) {
    outLocal.SetValue(i, ComputeSmallRow(base + i));
  }
  DataCopy(outGm[base], outLocal, n);
}
```

### 3. Little D, the Statute is open to scalar

When `D <= 32` or `D <= 64`, the Synchronization and Temporary Buffer of vector's Statute may be more expensive than scalar's cycle:

```cpp
if (D <= smallDThreshold) {
  float maxVal = -INFINITY;
  int32_t maxIdx = 0;
  for (int32_t i = 0; i < D; ++i) {
    float v = static_cast<float>(xLocal.GetValue(i));
    if (v > maxVal) {
      maxVal = v;
      maxIdx = i;
    }
  }
  outIdxLocal.SetValue(row, maxIdx);
} else {
  ReduceMax(maxLocal, xLocal, tmpBuf, D);
}
```

### 4. Move Branch Up

Fixed mode judgement should not be placed on the tile inner layer:

```cpp
// Init or Process begins with a decision.
bool sameShape = mode_ == MODE_SAME_SHAPE;

if (sameShape) {
  ProcessSameShape();
} else {
  ProcessGeneric();
}
```

Do not repeatedly judge dtype, rank, Broadcast mode in `for tile` or `for element`.

## Tiling Amendments

- Appropriately increase the size of single-processed particles and reduce the number of cycles
- Consider integrating with other kernels to reduce lanch costs
- For small lines/small result scenes, priority is given to the search for `rowsPerTile`, `outputsPerCopy`, `smallDThreshold`, rather than only to `TILE_LENGTH`.
