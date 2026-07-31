# Printf debugging

## Overview

Printf debugging is the most direct and quickest debugging method to locate the problem by printing variables in critical locations.

## Ascend C Printf Foundation

### Include header

```cpp
#include "kernel_printf.h"
```

### Basic Syntax:

```cpp
// Print a single value
printf("Value: %f\n", value);

// Print multiple values
printf("x = %.6f, y = %.6f\n", x, y);

// Print Integer
printf("Index: %d\n", index);

// Scientific mode
printf("Value: %e\n", large_value);
```

### Format Options

| Format | Purpose | Example: |
|-----|------|------|
| `%f` | Floating points (in decimal form) | `3.141593` |
| `%.6f` | Floats (6 decimal places) | `3.141593` |
| `%.2e` | Scientific mode (2 decimals) | `3.14e+00` |
| `%d` | Integer | `42` |
| `%s` | String | `hello` |
| `\n` | Line Break | - |

## Printf debugging techniques

### 1. Selective Printing

Avoiding an output explosion, printing only suspicious locations:

```cpp
// N elements before printing only
const int PRINT_N = 3;
for (int i = 0; i < PRINT_N && i < size; ++i) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}

// Conditional printing: only error large position
half threshold = 1e-3h;
for (int i = 0; i < size; ++i) {
    if (abs(output[i] - expected[i]) > threshold) {
        printf("Mismatch @%d: got %.6f, exp %.6f, diff=%.2e\n",
               i,
               static_cast<float>(output[i]),
               static_cast<float>(expected[i]),
               static_cast<float>(abs(output[i] - expected[i])));
    }
}

// Sample printing: Print one every N
for (int i = 0; i < size; i += 100) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}
```

### 2. FP16 Print note

FP16 needs to be converted to float to print correctly:

```cpp
half value = 3.14h;

// Error: Direct printing may not be accurate
printf("Value: %f\n", value);

// Correct: convert first to float
printf("Value: %.6f\n", static_cast<float>(value));
```

### 3. Key Location Marker

```cpp
printf("[DEBUG] Entering function\n");
// Code...
printf("[DEBUG] Before reduce\n");
// Code...
printf("[DEBUG] After reduce\n");
// Code...
printf("[DEBUG] Exiting function\n");

// Automark with Line Numbers
printf("[DEBUG] Line %d: value=%.6f\n", __LINE__, value);
```

### 4. Contrast Printing

```cpp
// Parallel printing of expected and actual values
for (int i = 0; i < size; i += 10) {
    printf("[%d] got=%.6f, exp=%.6f, diff=%.2e\n",
           i,
           static_cast<float>(output[i]),
           static_cast<float>(expected[i]),
           static_cast<float>(abs(output[i] - expected[i])));
}
```

### 5. Cluster border checks

```cpp
// Print the array boundaries and check if they cross the border.
printf("Array [0] = %.6f\n", static_cast<float>(arr[0]));
printf("Array [size-1] = %.6f\n", static_cast<float>(arr[size-1]));

// Length of print arrays
printf("Array size: %d\n", size);
```

### 6. Statistical information printing

```cpp
// Print minimum, maximum
half min_val = input[0];
half max_val = input[0];
float sum = 0.0f;

for (int i = 0; i < size; ++i) {
    min_val = min(min_val, input[i]);
    max_val = max(max_val, input[i]);
    sum += static_cast<float>(input[i]);
}

printf("Array stats: min=%.6f, max=%.6f, mean=%.6f\n",
       static_cast<float>(min_val),
       static_cast<float>(max_val),
       sum / static_cast<float>(size));

// Check for Inf/NAN
bool has_inf = false;
bool has_nan = false;
for (int i = 0; i < size; ++i) {
    float val = static_cast<float>(input[i]);
    if (isinf(val)) has_inf = true;
    if (isnan(val)) has_nan = true;
}
printf("Array checks: has_inf=%d, has_nan=%d\n", has_inf, has_nan);
```

## Advanced Usage

### 1. Conditional Debug Switches

```cpp
// Define Debug Switches
#define DEBUG_PRECISION 1

#if DEBUG_PRECISION
    #define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

// Use
DEBUG_PRINT("Debug info: value=%.6f\n", value);
```

### 2. Segment Printing

```cpp
// Print segment in long cycle
for (int i = 0; i < size; ++i) {
    // Calculating...

    // Print progress once per 1000 inverts
    if ((i + 1) % 1000 == 0) {
        printf("Progress: %d/%d (%.1f%%)\n",
               i + 1, size, (i + 1) * 100.0f / size);
    }
}
```

### 3. Function Entry/Export Tracking

```cpp
half Compute(half x) {
    printf("[ENTER] Compute(%.6f)\n", static_cast<float>(x));

    // Calculating...

    printf("[EXIT] Compute() -> %.6f\n", static_cast<float>(result));
    return result;
}
```

### 4. Acclaimed printing

```cpp
// Print and Verify Conditions
bool condition = /* ... */;
printf("[ASSERT] condition=%s (expected: true)\n",
       condition ? "true" : "false");

// Print & Validation Values
half expected = 1.0h;
half actual = /* ... */;
printf("[VERIFY] expected=%.6f, actual=%.6f, match=%s\n",
       static_cast<float>(expected),
       static_cast<float>(actual),
       (abs(actual - expected) < 1e-6h) ? "true" : "false");
```

## Common Printf Mode

### Model 1: numeric range check

```cpp
void CheckValueRange(const char* name, half* arr, int size) {
    half min_val = arr[0];
    half max_val = arr[0];
    float sum = 0.0f;
    int inf_count = 0;
    int nan_count = 0;

    for (int i = 0; i < size; ++i) {
        min_val = min(min_val, arr[i]);
        max_val = max(max_val, arr[i]);
        sum += static_cast<float>(arr[i]);

        float val = static_cast<float>(arr[i]);
        if (isinf(val)) inf_count++;
        if (isnan(val)) nan_count++;
    }

    printf("[%s] min=%.6f, max=%.6f, mean=%.6f, inf=%d, nan=%d\n",
           name,
           static_cast<float>(min_val),
           static_cast<float>(max_val),
           sum / static_cast<float>(size),
           inf_count,
           nan_count);
}
```

### Mode 2: Step tracking

```cpp
void TraceSteps(const char* step, half value) {
    printf("[STEP] %s: value=%.6f\n", step, static_cast<float>(value));
}

// Use
TraceSteps("initial", input);
TraceSteps("after_exp", exp_result);
TraceSteps("after_sum", sum_result);
TraceSteps("final", output);
```

### Mode 3: Error positioning

```cpp
void LocateErrors(half* output, half* expected, int size) {
    int error_count = 0;
    half max_error = 0.0h;
    int max_error_idx = -1;

    for (int i = 0; i < size; ++i) {
        half error = abs(output[i] - expected[i]);
        if (error > 1e-3h) {
            error_count++;
            if (error > max_error) {
                max_error = error;
                max_error_idx = i;
            }
        }
    }

    printf("[ERRORS] count=%d, max_error=%.2e @%d\n",
           error_count,
           static_cast<float>(max_error),
           max_error_idx);
}
```

## Printf Performance Note

1. **Production code removal**: Printf affects performance and should be removed after debugging
2. **Avoids over-output**: Too-too-Printf will output explosions, impact debugging
3. **Compiled using conditions**: debug output controlled by macro definition
4. **Selective printing**: only key information is printed to avoid full printing
