# Common accuracy trap details

## Trap 1: FP16 accuracy is inadequate

### Symptom
- Simple calculations have obvious errors.
- FP16 error is clearly greater than FP32
- Multiple cumulative error accumulations

### Reason
PP16 Only about 3-4 bits (FP32 about 6-7 bits) are vulnerable to the loss of accuracy in calculations requiring high accuracy.

### Solutions

**Mixed accuracy design principles**:
- Input/output maintenance FP16 (saving bandwidth and storage)
- Critical intermediate calculation FP32 (upgrade accuracy)
- Aggregation Operations Priority FP32

```cpp
// Critical middle value FP32
float sum_fp32 = 0.0f;
for (int i = 0; i < n; ++i) {
    sum_fp32 += static_cast<float>(values[i]);  // Turn first to FP32 Gradient
}
output = static_cast<half>(sum_fp32);  // Finally, turn around. FP16
```

**Typical application**: Reduce, Sum, Mean, Softmax, etc.

### Plan Phase prevention
When developing the operator development plan, ask voluntarily:
> "What is the operator to accuracy? Do you want to use FP32 to upgrade accuracy in the intermediate calculation? "

---

## Trap 2: exp/log spill

### Symptom
- Inf (infinite)
- NN (non-value) appears in output
- Large input value results abnormal

### Reason
- exp(x) overflow at x > 88 (FP16)
- log(x) was not defined at x≤0

### Solutions: numerically stable Softmax

```cpp
// Minus maximum, then ext, avoid spilling.
half max_val = input[0];
for (int i = 1; i < size; ++i) {
    max_val = max(max_val, input[i]);
}

half exp_sum = 0.0h;
for (int i = 0; i < size; ++i) {
    half shifted = input[i] - max_val;  // Key! Make the maximum input is0
    half exp_val = Exp(shifted);
    exp_sum += exp_val;
    output[i] = exp_val;
}

for (int i = 0; i < size; ++i) {
    output[i] = output[i] / exp_sum;
}
```

**The principle of numerical stability**:
- After subtracting maximum value, maximum input becomes 0, ext. (0) = 1
- Other inputs become negative, ext. (negative) < 1
- Avoided exp spills

### Other numerical stabilization techniques

```cpp
// Stable log-sum-exp (for log-softmax)
half max_val = ReduceMax(input);
half sum_exp = 0.0h;
for (int i = 0; i < size; ++i) {
    sum_exp += Exp(input[i] - max_val);
}
output = max_val + Log(sum_exp);  // Value stable log(sum(exp(x)))

// Stable Sigmoid
half sigmoid = 1.0h / (1.0h + Exp(-x));

// Avoid excessive alternatives to ext(x)
// If you know the range of x, you can precut it.
half safe_exp = Exp(min(x, 10.0h));  // Maximum limit index is 10
```

---

## Trap 3: Cutoff offset (Catastropic Regulation)

### Symptom
- When the two near numbers fell, the error suddenly got bigger.
- a ≈ b, a -b results inaccurate

### Reason
When the numbers are close to each other, the valid numbers are lost in large quantities, leading to an increase in relative error.

### Example:
```
FP16: 1.234 - 1.233 = 0.001(Perhaps it's just... 1 Bits effective)
```

### Solutions

**Method 1: Reordered formulae**
```cpp
// Original formula (value unstable)
half result = sqrt(x + 1) - sqrt(x);

// Stable version (reasonable)
half result = 1.0h / (sqrt(x + 1) + sqrt(x));
```

**method 2: Upgrade of the center accuracy**
```cpp
// Use FP32 for subtraction operations
float diff_fp32 = static_cast<float>(a) - static_cast<float>(b);
half result = static_cast<half>(diff_fp32);
```

**Methodology 3: Use of mathematical equivalents**
```cpp
// For example: Calculating 1 - cos(x)
// Unstable: 1 - cos(x)
// Stability:2 * sin(x/2)^2
```

---

## Trap 4: Reduce Operation accuracy Loss

### Symptom
- Reduce, the result is that error is bigger than the element-by-fact operation.
- Sum/Mean et al. accuracy inadequate

### Reason
The Reduce operation involved multiple additions, and the lack of FP16 accuracy led to the accumulation of error.

### Solutions

```cpp
// Use FP32 loader
float sum_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    sum_fp32 += static_cast<float>(input[i]);
}
output = static_cast<half>(sum_fp32);

// ReduceMax/ReduceMin is unaffected and can maintain FP16
half max_val = ReduceMax(input);  // FP16 Enough.

// ReduceSum/ReduceMean recommends FP32
float mean_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    mean_fp32 += static_cast<float>(input[i]);
}
mean_fp32 /= static_cast<float>(size);
```

> **PP32 Thrust is still insufficient**(restricted fp64 reference / big K contract K~1e3-1e4 / drop):
> See [high-precision-reduction.md] (high-precision-reduction.md) - Compensated-Neumaier,
> TwoProduct, block compensation matmul, HF32 trap, and when the hardware ceiling should stop.

---

## Trap 5: Remove Zero Risk

### Symptom
- Output appears, NAN.
- An abnormally large value appears in the output

### Solutions

```cpp
// Method 1: Add small constant (Epsilon)
half eps = 1e-7h;
half safe_div = numerator / (denominator + eps);

// Method 2: Conditional judgement
half eps = 1e-7h;
half safe_div = (abs(denominator) < eps) ? 0.0h : numerator / denominator;

// Method 3: Use maximum value protection
half safe_div = numerator / max(denominator, eps);
```

---

## Trap 6: Hardware constraints are not satisfactory

### Symptom
- Unusual results for specific input sizes
- Normal size is normal, border is wrong.

### Typical case: SoftmaxV5

**Found**: ReduceMax/ReduceSum calculates incorrectly when column < 8

**Reason**: Hardware constraint

**Solution**:
```cpp
// Add Input Authentication
if (cols < 8) {
    printf("Error: cols must be >= 8 (got %d)\n", cols);
    return;
}

// or specify constraints in the document
// Known limits: Columns must be ≥ 8
```

### Common hardware constraints

| Type of binding | Request | Inspection methods |
|---------|------|---------|
| Reduce Operations | Minimum number of elements ≥ 8 | Check reduce dimensions |
| Data Alignment | 32 Byte Alignment | Check the length of the tail axis |
| Single processing cap | Limited by UB | Big data needs a segment |

**32 Byte Reference**:
| data type | Bytes per Element | 32 byte alignment elements |
|---------|-------------|----------------|
| FP16 | 2 bytes | 16 elements |
| FP32 | 4 bytes | 8 elements |
| INT8 | 1 byte | 32 elements |

---

## Trap 7: Type conversion accuracy loss

### Symptom
- FP32 → FP16 has fallen accuracy
- Multiple conversion cumulative error

### Solutions

```cpp
// Avoid unnecessary type conversion
// Not recommended: frequent conversion
half temp = static_cast<half>(float_value);
float result = static_cast<float>(temp);

// Recommendation: maintain a type
float result = float_value;  // Use as much as possible. FP32 Calculate

// Convert only if necessary
half output = static_cast<half>(final_result_fp32);
```

---

## Trap 8: Cast API RoundMode with error ⭐

### Symptom
- Postcast data is completely wrong (not accuracy, data confusion)
- The multiline data output is identical.
- Error related to RoundMode selection, not accuracy loss

### Reason
Error selecting `RoundMode` parameter for Cast API. Key perception:

**Semantics of `CAST_NONE`**: The `CAST_RINT` model is expressed in the conversion of accuracy's loss,**the accuracy's loss is not included in the list**.

### Use correctly

| Convert direction | RoundMode | Reason |
|---------|-----------|------|
| half → float | `CAST_NONE` | Low → High accuracy, no accuracy loss, no rounding |
| float → half | `CAST_ROUND` | High →'s low accuracy, with accuracy's loss, needs to be rounded. |

### Example of error

```cpp
// ❌ Error: half → float used CAST_ROUND
AscendC::Cast<float, half>(xLocal, xLocalHalf, AscendC::RoundMode::CAST_ROUND, cols);
// Result: Data is completely wrong, multiple lines are the same
```

### Correct Example

```cpp
// ✅ Correct: half → float used CAST_NONE
AscendC::Cast<float, half>(xLocal, xLocalHalf, AscendC::RoundMode::CAST_NONE, cols);

// ✅ Correct: float → half used CAST_ROUND
AscendC::Cast<half, float>(yLocalHalf, xLocal, AscendC::RoundMode::CAST_ROUND, cols);
```

### RoundMode full description

```cpp
enum class RoundMode {
    CAST_NONE = 0,   // NoneaccuracyNot rounded at loss, yes.accuracyEquivalence of loss CAST_RINT
    CAST_RINT,       // 50% double rounded (bankers rounded)
    CAST_FLOOR,      // Infinitely rounded
    CAST_CEIL,       // It's a roundup.
    CAST_ROUND,      // Rounded
    CAST_TRUNC,      // Rounded to zero
    CAST_ODD,        // Recent Neighbors Rounded
    CAST_HYBRID,     // Random rounding (specified scene)
};
```

### Field case: SoftmaxV5 FP16 Mixed accuracy

```cpp
__aicore__ inline void ComputeFp16()
{
    // Step 1: half → float (low → height accuracy)
    AscendC::Cast<float, half>(xLocal, xLocalHalf, AscendC::RoundMode::CAST_NONE, cols);

    // Step 2: perform softmax calculations on FP32
    // ... ReduceMax, Adds, Exp, ReduceSum, Muls ...

    // Step 3: float → half(High)→Lowaccuracy)
    AscendC::Cast<half, float>(yLocalHalf, xLocal, AscendC::RoundMode::CAST_ROUND, cols);
}
```

### Preventive measures

1. **Before using Cast API, the official document must be consulted to confirm RundMode**
2. **Low accuracy → High accuracy: Use `CAST_NONE`**
3. **High accuracy → Low accuracy: Use `CAST_ROUND` or other rounding mode**

---

## Trap 9: All output is 0 ⭐ ⭐ ⭐

### Symptom
- Output data is all 0 or random error
- Expected value but actual value 0

### Reason 1: pipeline Synchronization (EnQue/DeQue Missing) ⭐ ⭐ ⭐

**Core issue**: DataCopy is a step DMA, return immediately.

| Mode | Code | Evaluation |
|------|------|------|
| ❌ error | `DataCopy(x, gm, n); Compute(x);` | Synchronising folder |
| ✅ Correct | `DataCopy → EnQue → DeQue → Compute` | Recommendations |
| ⚠ ️ debug | `DataCopy → PipeBarrier → Compute` | Authentication for |

```cpp
// ❌ error
LocalTensor<T> x = allocator.Alloc<T, 64>();
DataCopy(x, xGm, count);
Cast<half, int8_t>(xHalf, x, ...);  // ⛔️ Data may not be available

// ✅ Correct: EnQue/DeQue
void CopyIn() {
    LocalTensor<T> x = inQueue.AllocTensor<T>();
    DataCopy(x, xGm, count);
    inQueue.EnQue(x);
}
void Compute() {
    LocalTensor<T> x = inQueue.DeQue<T>();  // Waiting for data to be ready
    Cast<half, int8_t>(xHalf, x, ...);
    inQueue.FreeTensor(x);
}

// ⚠ ️ transfer testimonial: temporary plus Pipe Barrier
DataCopy(x, xGm, count);
PipeBarrier<PIPE_ALL>();  // If the result is correct, confirm it's a problem of synchronization.
```

### Reason 2: DataCopy non-32 byte alignment

**Question**: `DataCopy(dst, src, count)` requires `count * sizeof(T)` to be 32 bytes aligned.

| data type | 32B Alignment Elements |
|---------|---------------|
| FP16 | 16, 32, 48... |
| FP32/INT32 | 8, 16, 24... |
| INT8 | 32, 64, 96... |

```cpp
// ❌ error: 8 byte incoherent
DataCopy(indicesGm, indicesLocal, 2);  // 2 * 4 = 8B

// ✅ Correct: Use DataCopyPad
DataCopyExtParams p{1, rowsThisCore * sizeof(int32_t), 0, 0};
DataCopyPad(indicesGm, indicesLocal, p);
```

### Reason 3: GlobalTensor. SetValue

**Question**: GlobalTensor. SetValue may not be in force.

```cpp
// ❌ Avoid
outGm.SetValue(0, 10);

// ✅ Recommendations
LocalTensor<T> tmp = buf.Get<T>();
tmp.SetValue(0, value);
DataCopyPad(dstGm, tmp, {1, sizeof(T), 0, 0});
```

### Diagnostic process

```
Output All As 0?
│
├─ [1] InspectionpipelineSync ⭐⭐⭐
│   └─ DataCopy Later. EnQue/DeQue?
│       └─ Yes → Add Synchronization (on a temporary basis) PipeBarrier Validation)
│
├─ [2] Check Data Alignment
│   └─ count * sizeof(T) Yes. 32 A multiple?
│       └─ Yes → Change DataCopyPad
│
└─ [3] Inspection GlobalTensor
    └─ It's working. SetValue?
        └─ Yes. → Change LocalTensor + DataCopyPad
```

### Field cases

**Abs operator output all 0**:
- Reason: DataCopy directly after Cast, missing sync
- Rehabilitation: shift to the EnQue/DeQue mechanism

**Arg Max 40 Nuclear Failure**:
- Reason: 8B per nuclear output, not 32B alignment
- Repair: Change to DataCopyPad

---

## Trap 10: MX block quantification format Cast formula floor offset caused NAN

### Symptom

- operator output tensor has a large number of NNs (typically 50-80%)
- NAN is concentrated in continuous areas of certain rows (larger rows of amax at the time of quantification)
- Change to mxfp8 / mxfp4 after MX block quantification format
- NN decreases without disappearing after replacing scale with constant (e. g. e8m0 = 0x7F, decoded scale = 1.0)

### Reason

E8M0 code uses floor offset instead of ceil offset:

```cpp
// ❌ floor offset
e8m0_byte = biased_exp_amax - emax_quant_dtype;
```

Floor offset `amax / decoded_scale` may fall between `[quant_dtype_max, 2 × quant_dtype_max)` (for e4m3 = `[448, 896)`). `Cast' quantified dtype, fp32, RINT' will output NAN (for fp8_e4m3 as `0x7F`) over dtype_max.

### Solutions

Change to ciel offset:

```cpp
// ✅ ceil offset
e8m0_byte = (biased_exp_amax - emax_quant_dtype) + 1;
```

ceil offset ensures that `amax / decoded_scale ≤ quant_dtype_max`, Cast doesn't spill.

### Query Process

1. Check if NN is distributed in a specific row → is an → skeptical formula for quantitative path values
2. Replace the original scale runback with stub scale = neutral constant (0x7F = 1.0)
   - If NN disappears → root cause in scale path (playout / index etc.)
   - If NN is still in → root cause in Cast numeric formula (this trap)
3. Insert `AscendC::printf` sampling before and after Quantification Cast
   - NN byte → confirmed Cast overflow after Cast prefp32 values are normal + Cast
4. Check E8M0 generation code, select ciel offset

Parameter contrast: `emax_quant_dtype` take values can be found in [api-precision.md MX block quantitative format accuracy path] (../../ascendc-api-best-practices/references/api-precision.md).

---

## The trap inventory.

When faced with accuracy problems, check sequentially:

- [ ] **Is the output all 0?**(Check DataCopy Alignment / GlobalTensor. SetValue) ⭐ ⭐ ⭐
- [ ] Has Cast API been used? (Checks whether RoundMode is correct) ⭐
- [ ] Has a FP16 builder been used? (replaced with FP32)
- [ ] Is there an exp/log operation? (Check spill)
- [ ] Is there a reduction of close to a number? (reset formulae or increase accuracy)
- [ ] Is there a Reduce operation? (using FP32 loader)
- [ ] Is there a division? (Except zero protection)
- [ ] Satisfactory hardware constraints? (Accompanying, minimum elements)
- [ ] Is the type conversion reasonable? (avoid unnecessary conversion)
- [ ] **MX Block Quantification Format Can the Cast formula be offset by ceil?**(floor offset will make max/ scale super dtype max cause NAN)
