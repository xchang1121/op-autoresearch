# Field debugging cases

## Case 1: SoftmaxV5 accuracy debug

### Description of the problem

The Softmax operator output accuracy at a specified input scale did not meet expectations, and the validation of the foot report accuracy failed.

### Debug process

#### Step 1: Minimum Recoverable Test

```bash
# Start with minimum shape, 32 byte alignment
./softmaxv5 16 16 8 fp32

# If passed, test FP16.
./softmaxv5 16 16 8 fp16
```

Results**: small-scale tests passed and large-scale tests failed

#### Step 2: error analysis

```python
# Analyse error distribution
pred = np.load('output.npy')
truth = np.load('expected.npy')
error = np.abs(pred - truth)

print(f"Max error: {error.max():.2e}")
print(f"Mean error: {error.mean():.2e}")

# Find the worst samples.
worst_idx = error.argmax()
print(f"Worst @{worst_idx}: pred={pred.flat[worst_idx]}, truth={truth.flat[worst_idx]}")
```

**Found**: error is concentrated in specific columns

#### Step 3: Secondary disassembly certification

```cpp
// Softmax formula: softmax(x)=exp(x-max)/ sum(exp(x-max))

// Step 1: Verify ReduceMax
half max_val = ReduceMax(input);
printf("Step1 - max: %.6f\n", static_cast<float>(max_val));

// Step 2: Sub-Commission after validation of broadcast
for (int i = 0; i < size; ++i) {
    half shifted = input[i] - max_val;
    if (i < 3) {
        printf("Step2 - shifted[%d]: %.6f (input=%.6f, max=%.6f)\n",
               i, static_cast<float>(shifted),
               static_cast<float>(input[i]), static_cast<float>(max_val));
    }
}

// Step 3: Validate Exp
for (int i = 0; i < size; ++i) {
    half exp_val = Exp(input[i] - max_val);
    if (i < 3) {
        printf("Step3 - exp[%d]: %.6f\n", i, static_cast<float>(exp_val));
    }
}

// Step 4: Validate ReduceSum
half exp_sum = ReduceSum(exp_values);
printf("Step4 - exp_sum: %.6f\n", static_cast<float>(exp_sum));

// Step 5: Harmonization of certification
for (int i = 0; i < size; ++i) {
    output[i] = exp_values[i] / exp_sum;
    if (i < 3) {
        printf("Step5 - output[%d]: %.6f\n", i, static_cast<float>(output[i]));
    }
}

// Validation: The sum of the output should be close to 1.
half output_sum = ReduceSum(output);
printf("Verification - output_sum: %.6f (should be 1.0)\n",
       static_cast<float>(output_sum));
```

**Step 4: Border conditions test**

```bash
# Test hardware bound borders
./softmaxv5 8 8 8 fp32    # Minimum Columns=8
./softmaxv5 16 8 8 fp16   # FP16 Minimum Columns

# Testing large scale
./softmaxv5 1024 256 8 fp32

# Test Non-Square
./softmaxv5 256 512 4 fp32
```

**Key findings**:
- Columns must be ≥-8, otherwise ReduceMax/ReduceSum is not correctly calculated
- This is a constraint to the hardware reduce operation.

#### Step 5: Solutions

1. Add input authentication:
```cpp
if (cols < 8) {
    printf("Error: cols must be >= 8 (got %d)\n", cols);
    return;
}
```

2. Make this constraint clear in the document:
```markdown
## Known Limits
- Columns must ≥ 8(hardware) Reduce OPERATIONAL CONSTRAINTS)
```

### Lessons learned

| Problem | Gene. | Solutions |
|-----|------|---------|
| accuracy-specific size anomaly | Hardware constraints not satisfactory | Add Input Validation + Document Description |
| Reduce, it's a mistake. | Columns < 8 | Ensure column count ≥8 |

---

## Case 2: Sinh operator FP16 accuracy is deficient

### Description of the problem

Sinh operator is clearly inadequate under FP16 accuracy, which is more than 1% relative to error.

### Math Formula

```
sinh(x) = (exp(x) - exp(-x)) / 2
```

### Debug process

#### Step 1: FP32 vs FP16 Contrast

```python
# FP32 Test
result_fp32 = sinh_fp32(test_input)
error_fp32 = np.abs(result_fp32 - expected)
print(f"FP32 max error: {error_fp32.max():.2e}")  # ~1e-6

# FP16 Test
result_fp16 = sinh_fp16(test_input)
error_fp16 = np.abs(result_fp16 - expected)
print(f"FP16 max error: {error_fp16.max():.2e}")  # ~1e-2
```

**Found**: FP16 error significantly greater than FP32

#### Step 2: Debug

```cpp
// Dismantling calculation step
half x = 1.5h;

// Step 1: exp(x)
half exp_x = Exp(x);
printf("exp(%.2f) = %.6f\n", static_cast<float>(x), static_cast<float>(exp_x));

// Step 2: exp(-x)
half exp_neg_x = Exp(-x);
printf("exp(%.2f) = %.6f\n", static_cast<float>(-x), static_cast<float>(exp_neg_x));

// Step 3: Subtract
half numerator = exp_x - exp_neg_x;
printf("numerator = %.6f - %.6f = %.6f\n",
       static_cast<float>(exp_x),
       static_cast<float>(exp_neg_x),
       static_cast<float>(numerator));

// Step 4: Division
half result = numerator / 2.0h;
printf("result = %.6f / 2 = %.6f\n",
       static_cast<float>(numerator),
       static_cast<float>(result));
```

**Found**: Decreasing step `exp_x - exp_neg_x` lost significantly under FP16 accuracy

#### Step 3: Solutions

Using FP32 middle accuracy:

```cpp
half SinhStable(half x) {
    // Use FP32 for intermediate calculations
    float x_f32 = static_cast<float>(x);

    float exp_x = exp(x_f32);
    float exp_neg_x = exp(-x_f32);
    float numerator_f32 = exp_x - exp_neg_x;

    return static_cast<half>(numerator_f32 / 2.0f);
}
```

### Lessons learned

| Problem | Gene. | Solutions |
|-----|------|---------|
| FP16 accuracy is inadequate | Reduced offset caused accuracy losses | Use of FP32 for critical steps |
| FP16 error Big | Exp(x)-exp(-x) accuracy Loss | Intermediate calculation using FP32 |

---

## Case 3: ReduceSum plus accuracy loss

### Description of the problem

ReduceSum is not enough for accuracy when a large number of elements are added, especially FP16.

### Debug process

#### Step 1: Revert the problem

```python
# Generate test data
size = 1000
input_data = np.ones(size, dtype=np.float16) * 0.1

# Expected results
expected = 100.0  # 1000 * 0.1

# Actual results
result = reducesum_fp16(input_data)
print(f"Expected: {expected}, Got: {result}, Error: {abs(result - expected)}")
```

**Result**: error reaches 1.5 (relative to error 1.5%)

#### Step 2: Analysis of causes

FP16 accuracy is limited, and many times cumulatively results in the accumulation of error:
- FP16 About 3-4 significant places
- 1,000 times cumulative, each time error

#### Step 3: Solutions

Use FP32 loader:

```cpp
half ReduceSumAccurate(half* input, int size) {
    // Use FP32 loader
    float sum_fp32 = 0.0f;

    for (int i = 0; i < size; ++i) {
        sum_fp32 += static_cast<float>(input[i]);
    }

    return static_cast<half>(sum_fp32);
}
```

**Validation**: error down to 1e-4 below

### Lessons learned

| Problem | Gene. | Solutions |
|-----|------|---------|
| Reduce accuracy losses | FP16 excise error accumulation | Use FP32 loader |

---

## Case 4: exp spill caused Inf

### Description of the problem

Softmax operator output Inf when the input value is larger.

### Debug process

#### Step 1: Problem positioning

```python
# Test Big Inputs
large_input = np.array([[100.0, 101.0, 102.0]], dtype=np.float16)
result = softmax(large_input)
print(result)  # [nan, nan, nan]
```

#### Step 2: Printf debug

```cpp
// Direct exp spills
half x = 100.0h;
half exp_x = Exp(x);
printf("exp(%.1f) = %f\n", static_cast<float>(x), static_cast<float>(exp_x));
// Output: ext(100.0) = inf
```

#### Step 3: Solutions

Softmax:

```cpp
half SoftmaxStable(half* input, int size) {
    // Maximum first value
    half max_val = input[0];
    for (int i = 1; i < size; ++i) {
        max_val = max(max_val, input[i]);
    }

    // Calculate exp(x-max), avoid spills
    float exp_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        half shifted = input[i] - max_val;  // Maximum input becomes 0
        exp_sum += static_cast<float>(Exp(shifted));
    }

    // Normalization
    for (int i = 0; i < size; ++i) {
        half shifted = input[i] - max_val;
        output[i] = Exp(shifted) / static_cast<half>(exp_sum);
    }
}
```

### Lessons learned

| Problem | Gene. | Solutions |
|-----|------|---------|
| exp spill | Excessive input value | Less max, then ext. |
| Softmax Output Inf | exp(x) when spilling x>88 | Numerical stabilization algorithm |

---

## Case 5: Multi-matrix, head vivid NAN when slammed in L1

### Description of the problem

Some of the operators (typical multi-head category: multi-head MatMul/ multi-head Attention, etc.) bring multiple matrices to the same piece of L1 NZ Buffer and load them once to L0. Each matrix at the beginning of L1 has to be correctly calculated by NZfset physical layout, which is strongly related to data dtype.

Toggle dtype path (e. g. fp16 → fp8) or add scale tensor, offset formulae can easily go wrong.

### Symptom

Watch the output of**chilling NAN mode**by head-dimensional slices:

```
head 0: ✅ Validity, none NaN
head 1: ❌ All NaN(fp8 0x7F)
head 2: ❌ All NaN
head 3: ✅ Validity, none NaN
```

Typical migraines such as `[OK, BAD, BAD, OK]` / `[OK, BAD, OK, BAD]`.

### Debug process

#### Step 1: Degrade to single matrix (single head) test

Degrade operator to `multi-matrix-count = 1` (single head) with other parameters unchanged.

- Single matrix output normal → problem in multiple matrix fusion (this case)
- SinglematrixYeah.NaN →The problem is at the bottom.scalePath/ CastFormula/ LoadDatafields), see[common-traps.mdA trap.10](common-traps.md)

#### Step 2: Constant replacement isolation

Replace all suspicious scale tensor with a neutral constant (scale = 1.0), rerun:

- NN disappears → root cause in scale path
- NAN is still at → root in data carrier (this case)

#### Step 3: Sample by stage printf

Insert printf at each Company status output, with a different medium for sampling headrows:

- Early stage output is normal at head 0 / 3, head 1 / 2 has exploded → locking on the stage's fusion matrix used
- Early stage all normal, late stage before error → locks the later stage used

The corresponding L1 fusion matrix is the suspect after locating it to a specific status.

#### Step 4: Head amongfset formulae for contrasting data carriers

Checks whether multiple matrix collating with L1's head formula coefficient matches the number of NZ C0 elements in dtype:

```cpp
// ❌ Copy fp16 formulae to fp8 data carrier
DataCopy(aL1[g * mEff * CUBE_BLOCK], aGm[...], ...);   // CUBE_BLOCK=16 Yes. fp16 C0 Number of elements

// ✅ fp8 data carrier should use FP8_C0_ELEMS =32
DataCopy(aL1[g * mEff * FP8_C0_ELEMS], aGm[...], ...);
```

The restored mode may change from `[OK, BAD, BAD, OK]` to `[OK, BAD, OK, BAD]` (partly restored but still incorrect, indicating that the scale carrier formula is also incorrect).

#### Step 5: Check an independent offset formula for scale carriers

Scale tensor is usually loaded with a B16 view + Dn2Nz, which is M-fractal element count**different from the data carrier**:

```cpp
// ❌ scale carrier misused data carrier formula coefficient
DataCopy(aScaleL1B16[g * mEff * CUBE_BLOCK], aScaleGmB16[...], ...);
// or
DataCopy(aScaleL1B16[g * mEff * FP8_C0_ELEMS], aScaleGmB16[...], ...);

// ✅ scale carriers use their own M-fractal element count
DataCopy(aScaleL1B16[g * mEff * scaleK_b16], aScaleGmB16[...], ...);
```

Fix scale after all head output is normal.

### Lessons learned

| Problem | Gene. | Solutions |
|-----|------|---------|
| Multi-head NAN | Data carrier head input formula dtype misused | Press dtype for C0 elements (fp16 = 16, fp8 = 32, fp4 = 64) |
| One of the repairs is still partial, NAN. | Scale carrier formula coefficient misusing data carrier formulae | scale carrier formula independently extrapolates, without copying data carriers |
| Multi-head All NAN | Error on a lower path outside the spell | Decline to single matrix priority |

### Key principles

1. **Degradation test is the gold standard for multi-matrix-matrix problems**: single matrix PASS / multiple matrix FAIL → locks the adhesion immediately
2. **The data carrier vs scale carrier means the formula has to be independently extrapolated**: possibly multiple C0 sizes in the same operator, cannot copy the UCF
3. **head NN hint misspelled**: All BAD is a data access problem (scale axes / Cast formula / LoadData field) and the weirder BAD is a patchwork problem
4. **Constance replacement separation root factor**: replacement of suspected tensor with neutral constant, identifying "the tensor path vs other path"

Parameters contrast: NZ C0 size, Multi-matrix head input formula is relevant to Cube/matmul class operator and is not active in this repository.

---

## Debugging lessons learned

### Debug Efficiency Sorting

| Methodology | Apply scene | Efficiency |
|-----|---------|------|
| error analysis | Preliminary diagnosis | ⭐⭐⭐ |
| Printf debug | Quick positioning | ⭐⭐⭐ |
| Common trap screening. | Typical problem. | ⭐⭐ |
| Debug 2 | Complex issues | ⭐⭐ |
| Data comparison | System Authentication | ⭐ |

### Rapid diagnostic tips

| Symptom | Possible causes | Fast Check |
|-----|---------|---------|
| All the results are bad. | Formula/API problem | Check formula realization |
| FP16 Specially bad | accuracy is inadequate | Try FP32 Median |
| Inf/NAN | Spills/de-zeros | Check boundary values |
| Error on a specific scale | Hardware constraints | Check the alignment/minimal number of elements |
| Subtract error Large | Subtract offset | Reorder Formulae |

### Debug the laws of gold

1. **FP32 and FP16**: exclusion algorithm problems
2. **Alignment, later non- Alignment**: exclusion of alignment issues
3. **Simplicity, complexity**: starting with the smallest test case
4. **Fast, then deep**: fast-track method is not working and debugped in half
5. **Record everything**: every change is recorded error
