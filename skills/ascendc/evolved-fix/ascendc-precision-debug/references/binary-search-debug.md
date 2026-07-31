# Detailed guidance on the dichotomy

## Rationale

The dichotomy is based on a mathematical formula of operator, which divides the calculation process into multiple stages, gradually validating the intermediate results of each stage.

**Why as a means of securing:**
- Need to modify code to add printf output, more time-consuming
- Let's try a quick-track approach like error analysis, checking common traps.
- But when the problem is hard to locate, it's the surest way.

## When to use

Switch immediately to a half debug if either of the following conditions is met:
1. **Fast-track approach tried more than 7**still unpositioned
2. **All previous instruments have been tried**(error analysis, Printf location, common trap mapping)

> **Key principle**: do not blindly fail more than seven times, dichotomy can be faster to locate.

## Implementation steps

### Basic processes

```
Mathematical Formula Decomposition
    │
    ├─ I don't think so.1Layer: Outer Layer Operations
    │   └─ Add printf, validate intermediate results
    │
    ├─ I don't think so.2Layer: Inner layer
    │   └─ Add printf, validate intermediate results
    │
    ├─ ...Keep breaking up....
    │
    └─ First step in finding a difference → Problem positioning
```

### Example 1: sinh(x) debug

Mathematical formula: `sinh(x) = (exp(x) - exp(-x)) / 2`

```cpp
// Full split code.

// Step 1: Verify exp(x)
half exp_x = Exp(input);
printf("Step1 - exp(%.6f) = %.6f\n",
       static_cast<float>(input),
       static_cast<float>(exp_x));

// Step 2: Verify exp(-x)
half exp_neg_x = Exp(-input);
printf("Step2 - exp(%.6f) = %.6f\n",
       static_cast<float>(-input),
       static_cast<float>(exp_neg_x));

// Step 3: Validation of molecular abatement
half numerator = exp_x - exp_neg_x;
printf("Step3 - numerator = %.6f - %.6f = %.6f\n",
       static_cast<float>(exp_x),
       static_cast<float>(exp_neg_x),
       static_cast<float>(numerator));

// Step 4: Final division of certification
half result = numerator / 2.0h;
printf("Step4 - result = %.6f / 2 = %.6f\n",
       static_cast<float>(numerator),
       static_cast<float>(result));
```

### Example 2: Softmax debug

Mathematical formula: `softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))`

```cpp
// Step 0: Print input (first three elements)
printf("Input samples: [%.6f, %.6f, %.6f]\n",
       static_cast<float>(input[0]),
       static_cast<float>(input[1]),
       static_cast<float>(input[2]));

// Step 1: Verify ReduceMax
half max_val = ReduceMax(input);
printf("Step1 - max_val = %.6f\n", static_cast<float>(max_val));

// Step 2: Sub(first three elements) after validation of broadcast
for (int i = 0; i < 3 && i < size; ++i) {
    half shifted = input[i] - max_val;
    printf("Step2 - shifted[%d] = %.6f - %.6f = %.6f\n",
           i,
           static_cast<float>(input[i]),
           static_cast<float>(max_val),
           static_cast<float>(shifted));
}

// Step 3: Validate Exp (first three elements)
LocalTensor<half> exp_vals;
// ...allocation of memory...
for (int i = 0; i < 3 && i < size; ++i) {
    half shifted = input[i] - max_val;
    half exp_val = Exp(shifted);
    exp_vals[i] = exp_val;
    printf("Step3 - exp(%.6f) = %.6f\n",
           static_cast<float>(shifted),
           static_cast<float>(exp_val));
}

// Step 4: Validate ReduceSum
half exp_sum = ReduceSum(exp_vals);
printf("Step4 - exp_sum = %.6f\n", static_cast<float>(exp_sum));

// Step 5: Validation of conversion (first three elements)
for (int i = 0; i < 3 && i < size; ++i) {
    half output_val = exp_vals[i] / exp_sum;
    output[i] = output_val;
    printf("Step5 - output[%d] = %.6f / %.6f = %.6f\n",
           i,
           static_cast<float>(exp_vals[i]),
           static_cast<float>(exp_sum),
           static_cast<float>(output_val));
}

// Validation: The sum of the output should be close to 1.
half output_sum = ReduceSum(output);
printf("Verification - output_sum = %.6f (expected: 1.0)\n",
       static_cast<float>(output_sum));
```

### Example 3: ReduceSum debugging

```cpp
// Step 1: Print input (front N elements)
printf("Input samples: ");
for (int i = 0; i < min(5, size); ++i) {
    printf("%.6f ", static_cast<float>(input[i]));
}
printf("...\n");

// Step 2: Validate the cumulative process (sub-printing)
float sum_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    float val = static_cast<float>(input[i]);
    sum_fp32 += val;

    // Print every 100 elements
    if ((i + 1) % 100 == 0 || i == size - 1) {
        printf("Step2 - accumulated %d elements: sum = %.6f\n",
               i + 1, sum_fp32);
    }
}

// Step 3: Verify final output
half output = static_cast<half>(sum_fp32);
printf("Step3 - final output = %.6f\n", static_cast<float>(output));
```

## Debug techniques

### 1. Validate From Outward Inner

Starting at the outermost level of the mathematical formula, the layer-by-story validation is:
- The outermost is usually the easiest to verify.
- If you find out something's wrong, focus on the inside.

### 2. Comparative Reference Values

```cpp
// Calculate reference values using CPU/Python
// Python: numpy.exp(x)
half exp_ref = /* Reference values obtained from outside */;
half exp_npu = Exp(input);

printf("exp comparison: NPU=%.6f, Ref=%.6f, Diff=%.2e\n",
       static_cast<float>(exp_npu),
       static_cast<float>(exp_ref),
       static_cast<float>(abs(exp_npu - exp_ref)));
```

### 3. Selective Printing

To avoid an output explosion, only key messages are printed:

```cpp
// Print pre-N elements only
const int PRINT_N = 3;
for (int i = 0; i < PRINT_N && i < size; ++i) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}

// Conditional printing: only large error locations
for (int i = 0; i < size; ++i) {
    if (abs(output[i] - expected[i]) > threshold) {
        printf("Mismatch @%d: got %.6f, exp %.6f\n",
               i, output[i], expected[i]);
    }
}

// Sample printing: Print one every N
for (int i = 0; i < size; i += 100) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}
```

### 4. Cluster border checks

```cpp
// Print the array boundaries and check if they cross the border.
printf("Array bounds: [0] = %.6f, [size-1] = %.6f\n",
       static_cast<float>(arr[0]),
       static_cast<float>(arr[size - 1]));

// Length of print arrays
printf("Array size: %d\n", size);
```

### 5. Mathematical Validation

Validate using the mathematical nature of operator:

```cpp
// Softmax: The sum of the output should equal 1
half output_sum = ReduceSum(output);
printf("Softmax check: sum(output) = %.6f (expected: 1.0)\n",
       static_cast<float>(output_sum));

// RELU: Output should not have negative values
bool has_negative = false;
for (int i = 0; i < size; ++i) {
    if (output[i] < 0.0h) {
        has_negative = true;
        break;
    }
}
printf("ReLU check: has_negative = %s\n", has_negative ? "true" : "false");

// Symmetric: sin(-x) = -sin(x)
half sin_x = Sin(x);
half sin_neg_x = Sin(-x);
half symmetry_sum = sin_x + sin_neg_x;
printf("Symmetry check: sin(%.2f) + sin(%.2f) = %.6f (expected: 0)\n",
       static_cast<float>(x),
       static_cast<float>(-x),
       static_cast<float>(symmetry_sum));
```

## Debug Decision Tree

```
Start debugging in half.
    │
    ├─ Select the first intermediate step of the mathematical formula
    │   └─ Add printf, output middle result
    │
    ├─ Comparative reference value (%2)CPU/Python (calculated)
    │   │
    │   ├─ Unanimously → This is the right step. Go on.
    │   └─ Inconsistencies → The problem is in-depth analysis.
    │       │
    │       ├─ Inspection API The call is correct
    │       ├─ Inspectiondata typeRight or wrong?
    │       ├─ Check for spills/Spill
    │       └─ Check if hardware constraints are met
    │
    └─ Repeat until we find the source of the problem.
```

## common issue positioning

| Step output anomaly | Possible causes | Direction of inspection |
|------------|---------|---------|
| Exp() result is Inf | Excessive input | Whether to reduce max first |
| Reduce Max, it's a mistake. | Dimensions/ Alignment Issues | Check hardware constraints |
| The sum of the output is not 1 | Question of regularization | Check Div Operations |
| Subtract result error Large | Subtract offset | Reorder Formulae |
| Cumulative result error Large | accuracy is inadequate | Using FP32 excavator |
