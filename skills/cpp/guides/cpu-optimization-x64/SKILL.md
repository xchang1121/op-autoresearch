---
name: cpu-optimization-x64
description: "x64 CPU structured performance optimization techniques, SIMD/AVX vectorization, numerical stability and debugging policy"
category: method
version: "1.0.0"
metadata:
  backend: cpu
  dsl: cpp
  architecture: x86_64
  optimization_techniques: "SIMD, AVX, AVX2, AVX-512, cache optimization, loop unrolling"
---

# x64 CPU Performance Optimization Guide

## 1. x64 Architecture features and optimization policy

### 1.1 Architectural logo

- **Structure**: x86_64 (also known as x64, AMD64)
- **Main manufacturer**: Intel, AMD
- **SIMD Extension**: AVX, AVX2, AVX-512

### 1.2 Core optimization principles

1. **Multiple data processing using SIMD parallelity**: using AVX/AVX2/AVX-512 command
2. **Optimized Cache Use**: Access by line priority to increase Cache Rate
3. **Reduced branch prediction failure**: cycle roll-out, reduced condition judgement
4. **RAM Alignment**: Ensure Data Alignment to 32/64 Byte Boundary

## 2. SIMD/AVX vector Optimization

### 2.1 Basic concepts

**AVX (Advanced Victor Extensions)**is an extension of the SIMD command set for x86-64:

- **AVX**: 256-bit repository, capable of processing 8 float32 or 4 float64 simultaneously
- **AVX2**: enhanced AVX to support integer operation
- **AVX-512**: 512-bit repository capable of processing 16 float32 or 8 float64 simultaneously

### 2.2 compiler Autovector

**Recommended Method**:Jean.compilerAutovector, enabled by the compilation option:

```python
# Add vector options to load_inline
op_module = load_inline(
    name="custom_op",
    cpp_sources=cpp_source,
    extra_cflags=[
        "-O3",              # Maximum Optimization Level
        "-march=native",    # For the current CPU Structure optimization
        "-ftree-vectorize", # Enable AutovectorDilution
    ],
    verbose=True
)
```

### 2.3 Example of circular optimization

**Simple approach**(not optimized):

```cpp
torch::Tensor elementwise_add(torch::Tensor a, torch::Tensor b) {
    if (!a.is_contiguous()) a = a.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();

    torch::Tensor output = torch::zeros_like(a);
    auto a_ptr = a.data_ptr<float>();
    auto b_ptr = b.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    int64_t numel = a.numel();

    // Simple Loop
    for (int64_t i = 0; i < numel; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    return output;
}
```

**Optimized mode**(recycling, facilitating vector):

```cpp
torch::Tensor elementwise_add_optimized(torch::Tensor a, torch::Tensor b) {
    if (!a.is_contiguous()) a = a.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();

    torch::Tensor output = torch::zeros_like(a);
    auto a_ptr = a.data_ptr<float>();
    auto b_ptr = b.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    int64_t numel = a.numel();

    // Cycle Expand 8 times (compatible with the width of the AVX repository)
    int64_t i = 0;
    int64_t step = 8;
    for (; i + step <= numel; i += step) {
        out_ptr[i]     = a_ptr[i]     + b_ptr[i];
        out_ptr[i + 1] = a_ptr[i + 1] + b_ptr[i + 1];
        out_ptr[i + 2] = a_ptr[i + 2] + b_ptr[i + 2];
        out_ptr[i + 3] = a_ptr[i + 3] + b_ptr[i + 3];
        out_ptr[i + 4] = a_ptr[i + 4] + b_ptr[i + 4];
        out_ptr[i + 5] = a_ptr[i + 5] + b_ptr[i + 5];
        out_ptr[i + 6] = a_ptr[i + 6] + b_ptr[i + 6];
        out_ptr[i + 7] = a_ptr[i + 7] + b_ptr[i + 7];
    }

    // Handle the remaining elements
    for (; i < numel; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    return output;
}
```

**Optimization effect**: compiler is easier to identify and generate AVX vector Qid commands with increased performance 4-8 times.

### 2.4 Reduction Optimization

**Simple approach**

```cpp
float sum_simple(const float* data, int64_t size) {
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}
```

**Optimization**(parts added):

```cpp
float sum_optimized(const float* data, int64_t size) {
    // Use of 8 loaders to reduce data dependence
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        sum0 += data[i];
        sum1 += data[i + 1];
        sum2 += data[i + 2];
        sum3 += data[i + 3];
        sum4 += data[i + 4];
        sum5 += data[i + 5];
        sum6 += data[i + 6];
        sum7 += data[i + 7];
    }

    // Merge Results
    float sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;

    // Handle the remaining elements
    for (; i < size; ++i) {
        sum += data[i];
    }

    return sum;
}
```

**Key Optimization**: Cyclops are used to avoid loop-carrying dependence, allowing command-level parallels and vector.

## 3. Cache Optimization

### 3.1 Cache level

- **L1 Cache**: 32-64 KB, latency ~4 cycle
- **L2 Cache**: 256-512 KB, latency ~12 Period
- **L3 Cache**: 8-32 MB (shared), latency ~40 cycle
- **Main memory**: latency ~200 cycle

### 3.2 Optimizing strategies

**Principle**: Priority access by line to enhance spatial location

```cpp
// Example of 2D matrix transition optimization
torch::Tensor transpose_optimized(torch::Tensor input) {
    if (!input.is_contiguous()) input = input.contiguous();

    auto sizes = input.sizes();
    int64_t M = sizes[0];
    int64_t N = sizes[1];

    torch::Tensor output = torch::zeros({N, M}, input.options());
    auto in_ptr = input.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();

    // Discretion to increase C. O. R.
    const int64_t BLOCK_SIZE = 64;  // Fit to Cache Line Size

    for (int64_t i = 0; i < M; i += BLOCK_SIZE) {
        for (int64_t j = 0; j < N; j += BLOCK_SIZE) {
            int64_t i_max = std::min(i + BLOCK_SIZE, M);
            int64_t j_max = std::min(j + BLOCK_SIZE, N);

            for (int64_t ii = i; ii < i_max; ++ii) {
                for (int64_t jj = j; jj < j_max; ++jj) {
                    out_ptr[jj * M + ii] = in_ptr[ii * N + jj];
                }
            }
        }
    }

    return output;
}
```

## 4. Numerical stability optimization

### 4.1 Prevention of Softmax Spill

```cpp
torch::Tensor softmax_stable(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();

    torch::Tensor output = torch::zeros_like(x);
    auto x_ptr = x.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    int64_t numel = x.numel();

    // Maximum value found (prevent spill)
    float max_val = x_ptr[0];
    for (int64_t i = 1; i < numel; ++i) {
        max_val = std::max(max_val, x_ptr[i]);
    }

    // Calculate after the maximum value
    float sum = 0.0f;
    for (int64_t i = 0; i < numel; ++i) {
        float exp_val = std::exp(x_ptr[i] - max_val);
        out_ptr[i] = exp_val;
        sum += exp_val;
    }

    // Normalization
    for (int64_t i = 0; i < numel; ++i) {
        out_ptr[i] /= sum;
    }

    return output;
}
```

### 4.2 Kahan Summation (upgrading accuracy)

```cpp
float kahan_sum(const float* data, int64_t size) {
    float sum = 0.0f;
    float c = 0.0f;  // Compensatory variable

    for (int64_t i = 0; i < size; ++i) {
        float y = data[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}
```

**Use scenario**: accuracy loss reduced when processing the accumulation of large floating point numbers.

## 5. Full Optimization Example: ReLU

```cpp
torch::Tensor relu_optimized(torch::Tensor x) {
    // 1. Ensuring continuity
    if (!x.is_contiguous()) x = x.contiguous();

    // 2. Type check and conversion
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;

    // 3. Creation of output
    torch::Tensor output = torch::zeros_like(input);

    // 4. Optimized calculation logic
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();

        // Looping 8 times
        int64_t i = 0;
        for (; i + 8 <= numel; i += 8) {
            out_ptr[i]     = std::max(0.0f, x_ptr[i]);
            out_ptr[i + 1] = std::max(0.0f, x_ptr[i + 1]);
            out_ptr[i + 2] = std::max(0.0f, x_ptr[i + 2]);
            out_ptr[i + 3] = std::max(0.0f, x_ptr[i + 3]);
            out_ptr[i + 4] = std::max(0.0f, x_ptr[i + 4]);
            out_ptr[i + 5] = std::max(0.0f, x_ptr[i + 5]);
            out_ptr[i + 6] = std::max(0.0f, x_ptr[i + 6]);
            out_ptr[i + 7] = std::max(0.0f, x_ptr[i + 7]);
        }

        // Handle the remaining elements
        for (; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();

        // The same cycle spreads.
        int64_t i = 0;
        for (; i + 4 <= numel; i += 4) {  // double Expand 4 Double
            out_ptr[i]     = std::max(0.0, x_ptr[i]);
            out_ptr[i + 1] = std::max(0.0, x_ptr[i + 1]);
            out_ptr[i + 2] = std::max(0.0, x_ptr[i + 2]);
            out_ptr[i + 3] = std::max(0.0, x_ptr[i + 3]);
        }

        for (; i < numel; ++i) {
            out_ptr[i] = std::max(0.0, x_ptr[i]);
        }
    }

    // 5. Type reduction
    if (need_convert) output = output.to(dtype);
    return output;
}
```

## 6. Performance debugging and analysis

### 6.1 Performance Checklist

- [ ] Whether `-O3` optimization is enabled?
- [ ] Add `-march=native`?
- [ ] Is the cycle spread (8 times for float32, 4 times for float64)?
- [ ] Access memory according to line priority?
- [ ] Are unnecessary types of conversions avoided?
- [ ] Does the Reduction operation use multiple loaders?

### 6.2 Proposal for compilation options

```python
extra_cflags = [
    "-O3",                  # Maximum Optimization Level
    "-march=native",        # For the current CPU
    "-ftree-vectorize",     # AutovectorDilution
    "-ffast-math",          # Rapid mathematics (sacrifice)accuracy)
    "-funroll-loops",       # Looping
]
```

**Note: `-ffast-math` may affect the value accuracy, used with caution.

## 7. Common optimization error zone

| Error | Annotations | Recommendations |
|------|------|------|
| Excess manual vector | Handwritten AVX intrinsics code complex and error-prone | Prioritize compiler for automatic vector |
| The cycle spreads too much. | Overexpanding to increase code volume and reduce I-Cache hit rate | Float32 expands eight times, Float64 expands four times |
| Ignore Data Alignment | No alignment access reduced performance | Automatically align with `torch::zeros_like`, etc. |
| Unreasonable accuracy lifting | Double does not need to be mandatory for internal calculations | Float32 is enough to avoid unnecessary conversion. |

## 8. Summary

### x64 Optimization of key principles

1. **compilerAutovectorDilution**:Use`-O3 -march=native -ftree-vectorize`
2. **Recycling spread**: Float32 expands 8 times, Float64 expands 4 times
3. **Mulner**: Reduction operation avoids dependency with multiple loaders
4. **Cache friendly**: line-based access, large matrix segment processing
5. **Stable value**: Softmax minus maximum value with significant cumulative use of Kahan algorithm

### References

- Intel Optimization Manual: https://www.intel.com/content/www/us/en/developer/articles/technical/improve-performance-with-vectorization.html
- AVX Inline Function Reference: https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/details-of-avx-intrinsics.html
