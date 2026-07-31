# Data Comparison Method

## Overview

The data comparison method is the method of locating the accuracy problem by systematically constructing test case, comparing the output results under different conditions.

## Minimum Recoverable Test

### Test order principle

```
Prioritization:
1. 32Byte Alignment + FP32  → Exclude AlignmentaccuracyProblem
2. Non-matching test          → Check for alignment
3. FP16 accuracyTest      → Authentication FP16 accuracy
```

### Why is that order?

| Test Type | Purpose | Adoption of the note |
|---------|------|---------|
| Alignment +FP32 | Exclude alignment and accuracy issues | The algorithm is correct. |
| Inconsistent | Check for alignment | Need to process incoherent input |
| FP16 | Authentication FP16 accuracy | FP16 accuracy Sufficient |

### test case construction

#### 1. Minimum Alignment Test (Priority)

```python
import numpy as np

# 32 byte alignment + FP32 (exclude alignment and accuracy issues)
# FP32: 32 bytes = 8 elements
test_input_fp32 = np.random.rand(8, 8, 8).astype(np.float32)  # Endaxis8=8*4Bytes=32Byte Alignment

# Use simple values to facilitate authentication
test_input_fp32 = np.ones((8, 8, 8), dtype=np.float32)

# Validate output using known input
test_input_fp32 = np.zeros((8, 8, 8), dtype=np.float32)
test_input_fp32[0, 0, 0] = 1.0  # Single non-zero value
```

#### 2. Non-matching test

```python
# Non-matched input, need for special handling for testing
test_input_unaligned = np.random.rand(8, 8, 9).astype(np.float32)  # Endaxis9non32Byte Alignment

# or
test_input_unaligned = np.random.rand(8, 8, 17).astype(np.float32)  # Endaxis17
```

#### 3. FP16 accuracy Test

```python
# FP16 Test (Almost aligned)
# FP16: 32 bytes = 16 elements
test_input_fp16 = np.random.rand(8, 8, 16).astype(np.float16)  # Endaxis16=16*2Bytes=32Byte Alignment

# Comparison of FP16 and FP32 results
result_fp32 = run_operator(test_input_fp32.astype(np.float32))
result_fp16 = run_operator(test_input_fp16.astype(np.float16))

# Analysis of accuracy variances
error = np.abs(result_fp32 - result_fp16)
print(f"FP16 vs FP32: max error = {error.max():.2e}")
```

## Reference for alignment calculations

### 32 byte alignment rule

| data type | Bytes per Element | 32 byte alignment of elements | Example: shape |
|---------|-------------|----------------|---------|
| FP16 | 2 bytes | 16 elements | (..., 16), (..., 32), (..., 48) |
| FP32 | 4 bytes | 8 elements | (..., 8), (..., 16), (..., 24) |
| INT8 | 1 byte | 32 elements | (..., 32), (..., 64), (..., 96) |

### Check for alignment

```python
def is_32byte_aligned(shape, dtype):
    """InspectionshapeWhether the tail axis or not32Byte Alignment"""
    element_size = np.dtype(dtype).itemsize
    last_dim = shape[-1]
    return (last_dim * element_size) % 32 == 0

# Example:
print(is_32byte_aligned((8, 16), np.float16))  # True: 16*2=32Bytes
print(is_32byte_aligned((8, 8), np.float32))   # True: 8*4=32Bytes
print(is_32byte_aligned((8, 17), np.float16))  # False: 17*2=34Bytes
```

### Generate alignment test data

```python
def generate_aligned_test(shape, dtype):
    """Generate32Byte Alignment Test Data"""
    element_size = np.dtype(dtype).itemsize
    aligned_size = 32 // element_size

    # Resize tail axis to multiple of alignment size
    adjusted_shape = list(shape)
    adjusted_shape[-1] = ((shape[-1] + aligned_size - 1) // aligned_size) * aligned_size

    return np.random.rand(*adjusted_shape).astype(dtype)

# Example:
test_data = generate_aligned_test((8, 10), np.float32)  # Resize the tail axis to16(8Number of times)
print(f"Adjusted shape: {test_data.shape}")  # (8, 16)
```

## Boundary Value Test

### Standard boundary values set

```python
boundary_cases = {
    "zero value": 0.0,
    "Very small": 1e-10,
    "Small value": 1e-6,
    "Normal value": 1.0,
    "Great value": 1e6,
    "Extreme value": 1e10,
    "Negative value": -1.0,
    "FP16Saturation": 65504.0,     # FP16 Maximum value
    "FP16Negative saturation": -65504.0,  # FP16 Min
}

# Generate border test data
for name, value in boundary_cases.items():
    test_input = np.full((8, 8), value, dtype=np.float32)
    result = run_operator(test_input)
    print(f"{name}: input={value}, output={result[0, 0]}")
```

### Special Value Test

```python
# Special Float Value
special_values = {
    "It's perfect.": np.inf,
    "Negative.": -np.inf,
    "NaN": np.nan,
}

# Note: Ascend C may not support Inf/NAN, needing special treatment
```

## Comparison of intermediate results

### Step-by-step approach

```python
# Assume that operator is divided into multiple steps
def step1_exp(x):
    return np.exp(x)

def step2_minus(exp_x, exp_neg_x):
    return exp_x - exp_neg_x

def step3_divide(numerator):
    return numerator / 2.0

# Individual validation of each step
x = 1.5
exp_x = step1_exp(x)
exp_neg_x = step1_exp(-x)
numerator = step2_minus(exp_x, exp_neg_x)
result = step3_divide(numerator)

print(f"Step 1 - exp({x}) = {exp_x}")
print(f"Step 1 - exp({-x}) = {exp_neg_x}")
print(f"Step 2 - {exp_x} - {exp_neg_x} = {numerator}")
print(f"Step 3 - {numerator} / 2 = {result}")
```

### CPU vs NPU comparison

```python
# CPU Reference (using NumPy)
def softmax_cpu(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# NPU Results (from operator)
npu_output = run_operator_on_npu(input_data)

# Contrast
cpu_output = softmax_cpu(input_data)
error = np.abs(npu_output - cpu_output)

print(f"Max error: {error.max():.2e}")
print(f"Mean error: {error.mean():.2e}")

# Find maximum error position
max_error_idx = error.argmax()
print(f"Worst case @ {max_error_idx}:")
print(f"  CPU: {cpu_output.flatten()[max_error_idx]}")
print(f"  NPU: {npu_output.flatten()[max_error_idx]}")
```

## Type Conversion Test

### Decline step by step accuracy test

```python
# Gradual reduction of accuracy to FP16 from FP32
input_data = np.random.rand(8, 16).astype(np.float64)  # Highestaccuracy
expected = softmax_cpu(input_data)

for dtype in [np.float32, np.float16]:
    result = run_operator(input_data.astype(dtype))
    error = np.abs(result - expected)

    print(f"dtype={dtype.__name__}:")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
```

### Mixed accuracy test

```python
# Test the effects of different intermediate accuracys
# operator code needs to be modified to support different loaders accuracy

# Test 1: All FP16
result_all_fp16 = run_operator_all_fp16(input_data)

# Test 2: Composer FP32
result_fp32_accum = run_operator_fp32_accum(input_data)

# Contrast
print(f"All FP16: max error = {np.abs(result_all_fp16 - expected).max():.2e}")
print(f"FP32 Accum: max error = {np.abs(result_fp32_accum - expected).max():.2e}")
```

## Scale Test

### Tests at different scales

```python
# Test accuracy at different scales
test_shapes = [
    (8, 8),      # Small
    (16, 16),    # Small- and medium-sized
    (32, 32),    # Medium size
    (64, 64),    # Large and medium scale
    (128, 128),  # Large
    (256, 256),  # It's huge.
]

for shape in test_shapes:
    test_input = np.random.rand(*shape).astype(np.float32)
    result = run_operator(test_input)
    expected = reference_implementation(test_input)
    error = np.abs(result - expected)

    print(f"Shape {shape}: max error = {error.max():.2e}")
```

### Hardware bound boundary testing

```python
# Test the minimum element limit for reduce operations
reduce_sizes = [1, 2, 4, 8, 16, 32, 64]

for size in reduce_sizes:
    test_input = np.random.rand(8, size).astype(np.float32)
    result = run_operator(test_input)
    expected = reference_implementation(test_input)
    error = np.abs(result - expected)

    status = "PASS" if error.max() < 1e-5 else "FAIL"
    print(f"Reduce size {size}: {status}, max error = {error.max():.2e}")
```

## Test Script Template

```python
import numpy as np

def compare_results(output, expected, rtol=1e-5, atol=1e-6):
    """Compare results and print detailserrorInformation"""
    abs_error = np.abs(output - expected)
    rel_error = abs_error / (np.abs(expected) + atol)

    print(f"Max abs error: {abs_error.max():.2e}")
    print(f"Mean abs error: {abs_error.mean():.2e}")
    print(f"Max rel error: {rel_error.max():.2e}")
    print(f"Mean rel error: {rel_error.mean():.2e}")

    # Pass rate
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_rate = pass_mask.sum() / pass_mask.size * 100
    print(f"Pass rate: {pass_rate:.2f}%")

    # The worst sample.
    worst_idx = abs_error.argmax()
    print(f"Worst case @ {np.unravel_index(worst_idx, output.shape)}:")
    print(f"  Output: {output.flatten()[worst_idx]:.6f}")
    print(f"  Expected: {expected.flatten()[worst_idx]:.6f}")
    print(f"  Abs error: {abs_error.flatten()[worst_idx]:.2e}")

    return pass_rate > 99.0  # 99% By qualifying
```
