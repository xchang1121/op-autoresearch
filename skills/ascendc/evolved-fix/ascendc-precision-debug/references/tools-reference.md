# Tools and command references

## error analysis command

### Quick error Statistics

```bash
# Max. error and average error
python3 -c "import numpy as np; pred=np.load('output.npy'); truth=np.load('expected.npy'); \
  print(f'Max: {abs(pred-truth).max():.2e}, Mean: {abs(pred-truth).mean():.2e}')"

# Find the worst samples.
python3 -c "import numpy as np; pred=np.load('output.npy'); truth=np.load('expected.npy'); \
  err=abs(pred-truth); idx=err.argmax(); \
  print(f'Worst@{idx}: pred={pred.flat[idx]}, truth={truth.flat[idx]}')"

# Full error statistics (including relative error and fractions)
python3 -c "import numpy as np; pred=np.load('output.npy'); truth=np.load('expected.npy'); \
  err=abs(pred-truth); rel_err=err/(abs(truth)+1e-9); \
  print(f'Max abs: {err.max():.2e}, Max rel: {rel_err.max():.2e}, 95th: {np.percentile(rel_err, 95):.2e}')"
```

### Detailed error Analysis Script

```python
# error_analysis.py
import numpy as np
import sys

def analyze_error(pred_file, truth_file, rtol=1e-5, atol=1e-6):
    pred = np.load(pred_file)
    truth = np.load(truth_file)

    abs_error = np.abs(pred - truth)
    rel_error = abs_error / (np.abs(truth) + atol)

    print("=" * 60)
    print("errorAnalytical reports")
    print("=" * 60)
    print(f"Projection documents: {pred_file}")
    print(f"Real value files: {truth_file}")
    print()

    # Absolute error statistics
    print("Absolutely.errorStatistics:")
    print(f"  Maximum value: {abs_error.max():.6e}")
    print(f"  Average: {abs_error.mean():.6e}")
    print(f"  Medium: {np.median(abs_error):.6e}")
    print(f"  Standard deviation: {abs_error.std():.6e}")
    print()

    # error Statistics
    print("RelativeerrorStatistics:")
    print(f"  Maximum value: {rel_error.max():.6e}")
    print(f"  Average: {rel_error.mean():.6e}")
    print(f"  Medium: {np.median(rel_error):.6e}")
    print(f"  95Division: {np.percentile(rel_error, 95):.6e}")
    print(f"  99Division: {np.percentile(rel_error, 99):.6e}")
    print()

    # Pass rate
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_rate = pass_mask.sum() / pass_mask.size * 100
    print(f"Pass rate: {pass_rate:.2f}%")
    print()

    # error distribution
    print("errorDistribution:")
    for threshold in [1e-3, 1e-4, 1e-5, 1e-6]:
        count = (abs_error > threshold).sum()
        rate = count / abs_error.size * 100
        print(f"  error > {threshold:.0e}: {count} ({rate:.2f}%)")
    print()

    # The worst sample.
    worst_idx = abs_error.argmax()
    worst_pos = np.unravel_index(worst_idx, pred.shape)
    print(f"The worst sample. @ {worst_pos}:")
    print(f"  Projected: {pred[worst_pos]:.6f}")
    print(f"  True value: {truth[worst_pos]:.6f}")
    print(f"  Absolutely.error: {abs_error[worst_pos]:.6e}")
    print(f"  Relativeerror: {rel_error[worst_pos]:.6e}")

    return pass_rate > 99.0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 error_analysis.py <output.npy> <expected.npy>")
        sys.exit(1)

    success = analyze_error(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
```

## Printf Formatt Reference

### Basic Format

| Formatting | Type | Annotations | Example: |
|-------|------|------|------|
| `%f` | float | Decimal Forms | `3.141593` |
| `%.6f` | float | 6 decimal places | `3.141593` |
| `%.2e` | float | Scientific mode | `3.14e+00` |
| `%d` | int | Integer | `42` |
| `%u` | unsigned | Unsigned Integer | `42` |
| `%x` | hex | Hexadecimal | `0x2a` |
| `%c` | char | Character | `A` |
| `%s` | string | String | `hello` |

### Example: Ascend C Printf

```cpp
#include "kernel_printf.h"

// Base Print
printf("Value: %f\n", value);

// Specify decimal places
printf("Value: %.6f\n", value);     // 6Digits
printf("Value: %.2f\n", value);     // 2Digits

// Scientific mode
printf("Large: %.2e\n", large_value);

// Multiple values
printf("x=%.6f, y=%.6f\n", x, y);

// Integer
printf("Index: %d\n", index);
printf("Size: %d x %d\n", height, width);

// String
printf("Status: %s\n", "OK");

// Debug Information
printf("[DEBUG] Line %d: value=%.6f\n", __LINE__, value);

// FP16 needs conversion
half h = 3.14h;
printf("Half: %.6f\n", static_cast<float>(h));
```

## NPU Run Command

### Basic test running

```bash
# Enter Docker Container Run
./env_setup.sh "cd ops/my_operator/build && ./my_operator"

# Run with arguments
./env_setup.sh "cd ops/my_operator/build && ./my_operator 16 16 8 fp32"

# FP16 Test
./env_setup.sh "cd ops/my_operator/build && ./my_operator 16 16 8 fp16"
```

### Batch test script

```bash
#!/bin/bash
# batch_test.sh

# Test different input scales
shapes=("8:8:8" "16:16:8" "32:16:8" "64:32:8")
dtypes=("fp32" "fp16")

for shape in "${shapes[@]}"; do
    for dtype in "${dtypes[@]}"; do
        IFS=':' read -r M N K <<< "$shape"
        echo "Testing: M=$M, N=$N, K=$K, dtype=$dtype"

        ./env_setup.sh "cd ops/my_operator/build && ./my_operator $M $N $K $dtype"

        if [ $? -eq 0 ]; then
            echo "  PASS"
        else
            echo "  FAIL"
        fi
    done
done
```

## Data Generation Script

### Generate alignment test data

```python
# gen_aligned_data.py
import numpy as np
import sys

def generate_aligned(shape, dtype, output_path):
    """
    Generate32Byte Alignment Test Data

    shape: tuple, Datashape
    dtype: str, data type (fp16, fp32, int8)
    output_path: str, Output file path
    """
    np_type = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
    }[dtype]

    # Check and align
    element_size = np.dtype(np_type).itemsize
    aligned_size = 32 // element_size

    adjusted_shape = list(shape)
    adjusted_shape[-1] = ((shape[-1] + aligned_size - 1) // aligned_size) * aligned_size

    # Generate Random Data
    data = np.random.rand(*adjusted_shape).astype(np_type)

    # Save
    np.save(output_path, data)
    print(f"Generate Data: {output_path}")
    print(f"  Originalshape: {shape}")
    print(f"  Adjustmentsshape: {tuple(adjusted_shape)}")
    print(f"  data type: {dtype}")
    print(f"  Data range: [{data.min():.6f}, {data.max():.6f}]")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 gen_aligned_data.py <M> <N> <K> <dtype>")
        sys.exit(1)

    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    dtype = sys.argv[4]

    generate_aligned((M, N, K), dtype, f"input_{M}_{N}_{K}_{dtype}.npy")
```

### Generate boundary value test data

```python
# gen_boundary_data.py
import numpy as np

def generate_boundary_tests(dtype):
    """Generate boundary value test data"""
    np_type = np.float16 if dtype == 'fp16' else np.float32

    cases = {
        "zero": 0.0,
        "tiny": 1e-10,
        "small": 1e-6,
        "normal": 1.0,
        "large": 1e6,
        "huge": 1e10,
        "negative": -1.0,
    }

    if dtype == 'fp16':
        cases["saturation"] = 65504.0
        cases["negative_saturation"] = -65504.0

    for name, value in cases.items():
        data = np.full((8, 16), value, dtype=np_type)
        np.save(f"boundary_{name}_{dtype}.npy", data)
        print(f"Generate: boundary_{name}_{dtype}.npy (value={value})")

if __name__ == "__main__":
    generate_boundary_tests("fp32")
    generate_boundary_tests("fp16")
```

## Results validation scripts

```python
# verify_result.py
import numpy as np
import sys

def verify_result(output_file, expected_file, rtol=1e-5, atol=1e-6):
    """AuthenticationoperatorOutput result"""
    output = np.load(output_file)
    expected = np.load(expected_file)

    # Check shape
    if output.shape != expected.shape:
        print(f"shapeDo not match: output={output.shape}, expected={expected.shape}")
        return False

    # Calculate error
    abs_error = np.abs(output - expected)
    rel_error = abs_error / (np.abs(expected) + atol)

    max_abs_error = abs_error.max()
    max_rel_error = rel_error.max()

    # Print Results
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Maximum absoluteerror: {max_abs_error:.6e}")
    print(f"Maximum relativeerror: {max_rel_error:.6e}")

    # It's a judgment.
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_count = pass_mask.sum()
    total_count = pass_mask.size
    pass_rate = pass_count / total_count * 100

    print(f"Pass rate: {pass_count}/{total_count} ({pass_rate:.2f}%)")

    if pass_rate >= 99.0:
        print("Authentication: PASS")
        return True
    else:
        print("Authentication: FAIL")

        # Print failed sample
        fail_indices = np.where(~pass_mask)
        if len(fail_indices[0]) > 0:
            print("\nFailed sample (former)10Numbers):")
            fail_count = min(10, len(fail_indices[0]))
            for i in range(fail_count):
                idx = tuple(dim[i] for dim in fail_indices)
                print(f"  @{idx}: output={output[idx]:.6f}, "
                      f"expected={expected[idx]:.6f}, "
                      f"abs_err={abs_error[idx]:.2e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 verify_result.py <output.npy> <expected.npy> [rtol] [atol]")
        sys.exit(1)

    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6

    success = verify_result(sys.argv[1], sys.argv[2], rtol, atol)
    sys.exit(0 if success else 1)
```

## Recommended tolerance configuration

### Default tolerance difference

| scene | rtol | atol | Annotations |
|-----|------|------|------|
| FP16 | 1e-3 | 1e-4 | Floating point accuracy limited |
| FP32 | 1e-5 | 1e-6 | Standard accuracy |
| INT8 | - | 0 | It has to match exactly. |
| Softmax (FP16) | 1e-3 | 1e-4 | Probability Output |
| Softmax (FP32) | 1e-5 | 1e-6 | Probability Output |
| Reduce (FP16) | 5e-3 | 1e-4 | error is bigger. |
| Reduce (FP32) | 1e-5 | 1e-6 | Standard accuracy |

### Use Example

```bash
# FP16 Validation (Easy Portability)
python3 verify_result.py output.npy expected.npy 1e-3 1e-4

# FP32 Validation (Standard Capability)
python3 verify_result.py output.npy expected.npy 1e-5 1e-6

# Reduce operator Validation (Easier)
python3 verify_result.py output.npy expected.npy 5e-3 1e-4
```
