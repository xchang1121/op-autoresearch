---
name: cpu-optimization-arm
description: "ARM CPU Architecture Performance Optimization Techniques, NEON SIMD vectorization, Numerical Stability and Debugging Strategies"
category: method
version: "1.0.0"
metadata:
  backend: cpu
  dsl: cpp
  architecture: aarch64
  optimization_techniques: "NEON, SIMD, cache optimization, loop unrolling, ARM-specific"
---

# ARM CPU Performance Optimization Guidelines

## 1. ARM Architecture Characteristics and Optimization Policy

### 1.1 Architectural logo

- **Structure**: aarch64 (ARM 64-bit, ARMv8-A)
- **Main manufacturer**: ARM, Apple Silicon (M1/M2/M3), AWS Graviton, Hua
- **SIMD Extension**: NEON (Advanced SIMD)

### 1.2 Core optimization principles

1. **Multiple data processing using NEON parallelity**: using NEON commands
2. **Eliminate data dependence**: Avoiding register dependence between successive commands
3. **Optimized Cache Use**: Access by line priority to increase Cache Rate
4. **Reduced branch prediction failure**: cycle roll-out, reduced condition judgement

## 2. NEON SIMD vector Optimization

### 2.1 Basic concepts

**NEON (Advanced SIMD)**is an ARM SIMD command collection:

- **Reposer width**: 128 bits
- **Parallel processing capacity**:
  - 4 float32 (one accuracy float)
  - 2 float64 (two accuracy floats)
  - 16 int8, 8 int16, 4 int32, 2 int64

### 2.2 compiler Autovector

**Recommended Method**:Jean.compilerAutovector, enabled by the compilation option:

```python
# Add ARM vector option to load_inline
op_module = load_inline(
    name="custom_op",
    cpp_sources=cpp_source,
    extra_cflags=[
        "-O3",                  # Maximum Optimization Level
        "-mcpu=native",         # For the current ARM CPU Optimization
        "-ftree-vectorize",     # Enable AutovectorDilution
        "-ffast-math",          # Rapid Math Optimization (optional)
    ],
    verbose=True
)
```

**Note: ARM uses `-mcpu=native` instead of `-march=native`.

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

**Optimization method**(recycling, facilitating NEON vectorization):

```cpp
torch::Tensor elementwise_add_optimized(torch::Tensor a, torch::Tensor b) {
    if (!a.is_contiguous()) a = a.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();

    torch::Tensor output = torch::zeros_like(a);
    auto a_ptr = a.data_ptr<float>();
    auto b_ptr = b.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    int64_t numel = a.numel();

    // Looping 4 times (compatible NEON processing capacity for float32)
    int64_t i = 0;
    int64_t step = 4;
    for (; i + step <= numel; i += step) {
        out_ptr[i]     = a_ptr[i]     + b_ptr[i];
        out_ptr[i + 1] = a_ptr[i + 1] + b_ptr[i + 1];
        out_ptr[i + 2] = a_ptr[i + 2] + b_ptr[i + 2];
        out_ptr[i + 3] = a_ptr[i + 3] + b_ptr[i + 3];
    }

    // Handle the remaining elements
    for (; i < numel; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    return output;
}
```

**Optimized effect**: compiler is easier to identify and generate NEON vector commands, increasing performance 2-4 times.

**Key Difference**: ARM NEON for float32 is 4 in parallel, while x64 AVX is 8.

### 2.4 Elimination of data dependence (ARM-specific optimization)

**ARM feature**: The NEON directive usually requires multiple cycles, and if the next instruction uses the previous article ' s outcome repository, it will create a standstill.

**Simple approach**(data dependent):

```cpp
float sum_with_dependency(const float* data, int64_t size) {
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        sum += data[i];  // Every time you depend on the previous one. sum
    }
    return sum;
}
```

**Optimization**(elimination of dependency):

```cpp
float sum_no_dependency(const float* data, int64_t size) {
    // Use 4 independent loaders to eliminate data dependency
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int64_t i = 0;
    for (; i + 4 <= size; i += 4) {
        sum0 += data[i];        // Independent Thrust
        sum1 += data[i + 1];    // No dependency
        sum2 += data[i + 2];    // Complementary implementation
        sum3 += data[i + 3];
    }

    // Merge Results
    float sum = sum0 + sum1 + sum2 + sum3;

    // Handle the remaining elements
    for (; i < size; ++i) {
        sum += data[i];
    }

    return sum;
}
```

**Key Optimization**: Cyclops are used to avoid recycle-carrying dependency, allowing NEON pipeline to be executed in parallel.

### 2.5 Reduction Optimization

**Standard model**(adaptation NEON):

```cpp
torch::Tensor sum_reduction_optimized(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();

    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;

    torch::Tensor output;

    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        int64_t numel = input.numel();

        // 4 Thrusts (compatible NEON width)
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        int64_t i = 0;
        for (; i + 4 <= numel; i += 4) {
            sum0 += x_ptr[i];
            sum1 += x_ptr[i + 1];
            sum2 += x_ptr[i + 2];
            sum3 += x_ptr[i + 3];
        }

        float result = sum0 + sum1 + sum2 + sum3;

        // Disposal of surplus
        for (; i < numel; ++i) {
            result += x_ptr[i];
        }

        output = torch::tensor({result}, torch::kFloat32);
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        int64_t numel = input.numel();

        // 2 loaders (doule width in NEON 2)
        double sum0 = 0.0, sum1 = 0.0;

        int64_t i = 0;
        for (; i + 2 <= numel; i += 2) {
            sum0 += x_ptr[i];
            sum1 += x_ptr[i + 1];
        }

        double result = sum0 + sum1;

        for (; i < numel; ++i) {
            result += x_ptr[i];
        }

        output = torch::tensor({result}, torch::kFloat64);
    }

    if (need_convert) output = output.to(dtype);
    return output;
}
```

## 3. Cache Optimization

### 3.1 ARM Cache feature

Typical ARM structure (e.g. Apple M1):

- **L1 Cache**: 128-192 KB (data)+128-192 KB (directive)
- **L2 Cache**: 12-24 MB (shared)
- **Harmonized Memory Structure**: Apple Silicon uses Unified Memory, CPU and GPU sharing

### 3.2 Optimizing strategies

**PRINCIPLE**: Blank processing of big data to enhance cache reuse

```cpp
// matrix multiplication segment optimization (adaptation ARM cache)
torch::Tensor matmul_blocked(torch::Tensor A, torch::Tensor B) {
    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());
    auto a_ptr = A.data_ptr<float>();
    auto b_ptr = B.data_ptr<float>();
    auto c_ptr = C.data_ptr<float>();

    // Division size: Fit L1 Cache (usually 32-64)
    const int64_t BLOCK_SIZE = 32;

    for (int64_t i = 0; i < M; i += BLOCK_SIZE) {
        for (int64_t j = 0; j < N; j += BLOCK_SIZE) {
            for (int64_t k = 0; k < K; k += BLOCK_SIZE) {
                int64_t i_max = std::min(i + BLOCK_SIZE, M);
                int64_t j_max = std::min(j + BLOCK_SIZE, N);
                int64_t k_max = std::min(k + BLOCK_SIZE, K);

                // Block Count
                for (int64_t ii = i; ii < i_max; ++ii) {
                    for (int64_t jj = j; jj < j_max; ++jj) {
                        float sum = 0.0f;
                        for (int64_t kk = k; kk < k_max; ++kk) {
                            sum += a_ptr[ii * K + kk] * b_ptr[kk * N + jj];
                        }
                        c_ptr[ii * N + jj] += sum;
                    }
                }
            }
        }
    }

    return C;
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

    // Calculate exp after the maximum value (using 4 loaders)
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int64_t i = 0;
    for (; i + 4 <= numel; i += 4) {
        float exp0 = std::exp(x_ptr[i] - max_val);
        float exp1 = std::exp(x_ptr[i + 1] - max_val);
        float exp2 = std::exp(x_ptr[i + 2] - max_val);
        float exp3 = std::exp(x_ptr[i + 3] - max_val);

        out_ptr[i] = exp0;
        out_ptr[i + 1] = exp1;
        out_ptr[i + 2] = exp2;
        out_ptr[i + 3] = exp3;

        sum0 += exp0;
        sum1 += exp1;
        sum2 += exp2;
        sum3 += exp3;
    }

    float sum = sum0 + sum1 + sum2 + sum3;

    // Disposal of surplus
    for (; i < numel; ++i) {
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

### 4.2 Kahan Sumition

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

## 5. Full Optimization Example: ReLU

```cpp
torch::Tensor relu_optimized_arm(torch::Tensor x) {
    // 1. Ensuring continuity
    if (!x.is_contiguous()) x = x.contiguous();

    // 2. Type check and conversion
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;

    // 3. Creation of output
    torch::Tensor output = torch::zeros_like(input);

    // Optimized calculation logic (adaptation of ARM NEON)
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();

        // Looping 4 times (matching NEON float32 width)
        int64_t i = 0;
        for (; i + 4 <= numel; i += 4) {
            out_ptr[i]     = std::max(0.0f, x_ptr[i]);
            out_ptr[i + 1] = std::max(0.0f, x_ptr[i + 1]);
            out_ptr[i + 2] = std::max(0.0f, x_ptr[i + 2]);
            out_ptr[i + 3] = std::max(0.0f, x_ptr[i + 3]);
        }

        // Handle the remaining elements
        for (; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();

        // Looping 2 times (double width 2 in NEON)
        int64_t i = 0;
        for (; i + 2 <= numel; i += 2) {
            out_ptr[i]     = std::max(0.0, x_ptr[i]);
            out_ptr[i + 1] = std::max(0.0, x_ptr[i + 1]);
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

## 6. Apple Silicon Specific Optimization

### 6.1 Harmonized memory advantages

Apple M series chips use a unified memory architecture, CPU and GPU shared memory:

- **No copy of data required between CPU and GPU**
- **Large bandwidth**: Memory bandwidth up to 400-800 GB/s (M2 Pro/Max)

### 6.2 Core performance and efficiency

Apple Silicon has performance core (P-core) and efficiency core (E-core):

- **Optimization policy**: calculates the automated movement of intensive tasks to P-core
- **Compiler Options**: Auto-optimize using `-mcpu=native`

## 7. Performance debugging and analysis

### 7.1 Performance Check List

- [ ] Whether `-O3` optimization is enabled?
- [ ] Add `-mcpu=native` (not `-march`)?
- [ ] Is the cycle spread (4 times for float32, 2 times for float64)?
- [ ] Reduction uses a multi-gatherer (elimination of data dependence)?
- [ ] Access memory according to line priority?

### 7.2 Proposal for compilation options

```python
extra_cflags = [
    "-O3",                  # Maximum Optimization Level
    "-mcpu=native",         # For the current ARM CPU(Note: yes.) mcpu No, it's not. march)
    "-ftree-vectorize",     # AutovectorDilution
    "-ffast-math",          # Rapid mathematics (optional, sacrifice part)accuracy)
]
```

**Key differences**: ARM uses `-mcpu` instead of `-march`.

## 8. ARM vs x64 Optimized comparison

| Features | ARM (NEON) | x64 (AVX) |
|------|------------|-----------|
| SIMD width | 128 bits | 256 bits (AVX2), 512 bits (AVX-512) |
| Float32 Parallel | 4 | 8 (AVX2), 16 (AVX-512) |
| Float64 Parallel | 2 | 4 (AVX2), 8 (AVX-512) |
| Looping Multiplication (float32) | **4 times** | **8 times** |
| Looping Multiplication (float64) | **2 times** | **4 times** |
| Number of loaders (recommended) | 4 | 8 |
| Compile Options | `-mcpu=native` | `-march=native` |
| Data dependence sensitivity | **High**(needs special attention) | Medium |

## 9. Common optimization error zone

| Error | Annotations | Recommendations |
|------|------|------|
| Copy x64 Optimization | ARM and x64 have different parallels | Float32 Expand 4 times (not 8 times) |
| Ignore Data Dependence | ARM NEON command latency high, high impact dependency | Use multi-cumulator to eliminate dependency |
| Use `-march` | ARM should use `-mcpu` | Use `-mcpu=native` |
| Overexploited | Expand beyond NEON width is not helpful | Float32 up to 4 times |

## 10. Summary

### ARM Optimization of key principles

1. **compilerAutovectorDilution**:Use`-O3 -mcpu=native -ftree-vectorize`
2. **Recycling expansion**: Float32 expansion**4 times**, Float64 expansion**2 times**(compatibility of NEON width)
3. **Eliminate data dependency**: using**4 loaders**(Reduction operation)
4. **Cache friendly**: access by line priority, large matrix segment processing (block size 32-64)
5. **Stable value**: Softmax minus maximum value with significant cumulative use of Kahan algorithm

### ARM-specific note

- **Compiler Options**: Using `-mcpu=native` instead of `-march=native`
- **NEON width**: Float32 parallels 4 (not 8)
- **Data dependence**: NEON command latency height to avoid continuous repository dependence
- **Apple Silicon**: full utilization of unified memory and high bandwidth advantage

### References

- ARM NEON programming guide: https://developer.arm.com/documentation/den0018/latest/
- ARM C/C++ compiler Optimization: https://developer.arm.com/documentation/101458/latest/Optimize/Optimizing-C-C---code-with-Arm-SIMD--Neon-
