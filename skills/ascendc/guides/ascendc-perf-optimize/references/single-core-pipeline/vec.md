# VEC Bound Optimizing Policy

The VEC base is the most common bottleneck for the Elementwise, Activity, Reduction type operator. Victor ' s computing unit is time-consuming and the MTE2/MTE3 moving unit is relatively idle.

---

## Conditions for determination

- Veter unit is highly utilized, `aiv_vec_ratio` is time-consuming
- Victor command takes the lead time.
- The MTE2 and Cube units are relatively idle

**Severation of severity of bottlenecks**:

| VEC % | Level | Optimizing direction |
|----------|------|---------|
| 50–65% | Light | DoubleBuffer + UB has a lot to gain from integration. |
| 65–80% | Medium | Reduce Cast or Integration Commands |
| >80% | Depth | VEC itself is nearing the theoretical limit and optimizing space is limited |

---

## Elements of a simulation map analysis

- Identification of Victor command concentration area
- Check data dependency between Victor's commands and find opportunities for parallel implementation

**VEC base track feature**:

```
Time: 0 ---------------------------------------- 100ms

SCALAR     |##............................| 5%
SCALARLDST |##............................| 4%
MTE2       |.##....##....##....##.........|15%
VECTOR     |..############################|65%  <- Lead
MTE3       |..........................####.|10%
```

VECTOR lines continue to be active, MTE2 has a clear free wait, indicating that the speed of removal is faster than the rate of calculation.

---

## VEC Internal Command Analysis

Extract pid=5 (VECTOR) events from Chrome Trace JSON, by `name` classification:

| Command Type | Typical command | Meaning |
|---------|---------|------|
| Numerical type | `vec_add`, `vec_mul`, `vec_sub` | Basic calculation, low latency |
| Beyond the Function Class | `vec_exp`, `vec_log`, `vec_rec`, `vec_rsqrt` | High latency command, no hardware acceleration |
| Type Conversion Class | `vcvt_f2f`, `vcvt_f2s`, `vcvt_s2f` | Cast Cost |
| Affiliation | `vec_reduce_sum`, `vec_reduce_max` | Reduction expenses |

**Rules of judgement**:
- Cast > 20% → type conversion intensive scenes optimize Cast high returns
- Exceed function ratio > 30% → Depth VEC base, optimising space limited

> **RVEC**Modules: RVECEX (execution), RVECLD (loading), RVECST (storage), RVECSU (set-up)

---

## Policy 1: UB Integration

Multiple successive Vector operations are done directly in UB, and the intermediate results are not written back to GM, eliminating unnecessary MTE2/MTE3 round-trip.

```
Not integrated: GM → UB → Compute1 → GM → UB → Compute2 → GM    // 6 Numbers GM Visits
Integrating: GM → UB → Compute1 → Compute2 → UB → GM          // 2 Numbers GM Visits
```

**Check method**: Watch whether MTE3 (repeated)+MTE2 (read) has been inserted between two VECTOR active sections in the track. If yes, indicate that the intermediate result has passed GM and is not integrated.

| Operation | Annotations |
|------|------|
| Recognize integrated adjacent Vector operations | Elimination of intermediate moves and suspense |
| Chain Victor Operations Merge | Mul+Add → MulAdd, multiple active function chain processing |
| Reduce intermediate result writing back to UB | Convergence complete transmission in the repository |

---

## Policy 2: Reduce type conversion

Cast (type conversion) is the most common hidden expense in the VEC base. Typical mode: `fp16 → Cast fp32 → calculates → Cast fp16 '. When calculating itself only 1 –2 Directives, Cast may account for 50% of the total VEC time of 30 –.

| Operation | Annotations |
|------|------|
| Batch | Merges a multiple Cast into a large particle size operation |
| Avoid unnecessary, Cast. | Check if accuracy must be converted |
| Select the appropriate calculation accuracy | Full link fp16 or full link fp32, avoid round-trip conversion |

Common modifications:

```cpp
// Difference: float input also goes through the Cast/identity copy.
if constexpr (std::is_same_v<T, float>) {
  Adds(xf, xLocal, 0.0f, count);
} else {
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
}
ComputeFp32(xf, count);

// Good: float is a direct calculation source, not float before fp32 scratch.
if constexpr (std::is_same_v<T, float>) {
  ComputeFp32(xLocal, count);
} else {
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
  ComputeFp32(xf, count);
}
```

If fp16/bf16 native error meets reference requirements, you can keep separate native path for half-accuracy:

```cpp
if constexpr (std::is_same_v<T, half>) {
  Sigmoid(yLocal, xLocal, count);  // Avoid Cast to fp32 Again. Cast Come back. half
} else {
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
  Sigmoid(yf, xf, count);
  Cast(yLocal, yf, RoundMode::CAST_RINT, count);
}
```

---

## Policy 3: Integration Directives

Use integration command to reduce VEC commands:

| Command | Equivalent Operations | Annotations |
|------|---------|------|
| VMULA | VMUL + VADD | Multiplier Integration |
| VMULS | VMUL + VSUB | Multiplier/minus integration |
| VMADD | Aggregated Mode | Single Instructions Plus |

Typical replacement:

```cpp
// Discrepancies: two vector pass.
Muls(tmp, x, scale, count);
Adds(y, tmp, bias, count);

// Good: When supporting, use integration times plus, or put bias/scale into a previous phase.
Mad(y, x, scaleLocal, biasLocal, count);
```

For the activation chain, pre-empt the middle buffer to the minimum:

```cpp
// softplus-like: log(1 + exp(-abs(x))) + max(x, 0)
Abs(tmp0, x, count);
Muls(tmp0, tmp0, -1.0f, count);
Exp(tmp0, tmp0, count);
Adds(tmp0, tmp0, 1.0f, count);
Log(tmp0, tmp0, count);
Max(tmp1, x, zero, count);
Add(y, tmp0, tmp1, count);
```

Do not open a new float buffer for the phase that can be covered in situ; draw live-range, then decide the number of calcBuf segments.

---

## Policy 4: Return to low latency

For reduction operations, preference is given to reduceSum / ReduceMax / ReduceMin, avoiding manual for circular element-to-element returns.

Exceptions: When reduce dim is small (e.g. 2, 4, 8, 16, 32) and a barrier is done once in every line, the cost of synchronizing hardware may exceed the scalar cycle. You can press the D-discretion:

```cpp
if (D <= 32) {
  float acc = 0.0f;
  for (int32_t i = 0; i < D; ++i) {
    float v = static_cast<float>(xLocal.GetValue(i));
    acc += v;
  }
  outLocal.SetValue(0, acc);
} else {
  ReduceSum(sumLocal, xLocal, tmpBuf, D);
  PipeBarrier<PIPE_V>();
  outLocal.SetValue(0, sumLocal.GetValue(0));
}
```

---

## Policy 5: RegBase Access

| Operation | Annotations |
|------|------|
| Adjust Data Layout to RegBase Friendly | Repeated reading, alignment access |
| Use RegBase loading command | Reduce address calculation costs |
| Decrease number of vload/vstore | RegBase, big particle transfer advantage. |

---

## DoubleBuffer Check

Watching MTE2 and VECTOR rows over time in Chrome Trace:

| Mode | Features | Meaning |
|------|------|------|
| **DB Entry into force** | MTE2 and VECTOR appear alternately | Move in and calculate over and over and over and over and over again. |
| **DB Not in force** | MTE2 is all ahead, VECTOR is all behind | Serial execution, to open or repair DoubleBuffer |

---

## Tiling Amendments

- Adjust UB layout to support more efficient RegBase access mode
- Adjusts the tile particle size match for the Vector integration window
- Increases the size of the tile to reduce the number of cycles and the cost of streaming water outages (UB capacity: 192KB / 910B2, 248KB / 950)
- When enabling DoubleBuffer, the actual available UB needs to be divided by 2
