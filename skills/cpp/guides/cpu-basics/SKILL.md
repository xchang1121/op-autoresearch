---
name: cpu-basics
description: "CPU C++ operator Core Concept, Standard Structure Model, KernelBench code instruction and embedded extension method"
category: fundamental
version: "1.0.0"
metadata:
  backend: cpu
  dsl: cpp
  operator_patterns: "all"
  architecture: "x86_64, aarch64"
---

# CPU C++ Programming Base

## 1. Core concepts

### Kernel
- **Definition**: using the C++ function registered with `PYBIND11_MODULE`, compiled and executed on CPU
- **Characteristics**: Directly operating tensor data pointers to support multiple data types
- **Form**: with PyTorch C++ extension, dynamically compiled and loaded through `load_inline`

### tensor Processing
- **Consequence**: ensure continuity of the tensor RAM layout and avoid non-continuing access
- **System harmonization**: internal calculations use a uniform type (priority float32/float64/int32/int64) and final conversion back to the original type
- **Boundary check**: all arrays must check the border before visiting

### Memory management
- **Automandate**: PyTorch Automanaging tensor Memory Life Cycle
- **pointer operation**: direct operation data pointer for efficient calculation
- **Type security**: ensure that the type of pointer matches the type of tensor

## 2. Standard kernel structure (five-step model)

All CPU C++ kernels follow the same five-step structure model:

```cpp
torch::Tensor standard_kernel(torch::Tensor x) {
    // 1. Ensure that the input of tensor is continuous
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    // 2. Check data type to support multiple types
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64 &&
                        dtype != torch::kInt32 && dtype != torch::kInt64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;

    // Create output tensor
    torch::Tensor output = torch::zeros_like(input);

    // 4. Based on data type distribution
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);  // ReLU: max(0, x)
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt32) {
        auto x_ptr = input.data_ptr<int32_t>();
        auto out_ptr = output.data_ptr<int32_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt64) {
        auto x_ptr = input.data_ptr<int64_t>();
        auto out_ptr = output.data_ptr<int64_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0L, x_ptr[i]);
        }
    }

    // 5. Conversion back to original type
    if (need_convert) {
        output = output.to(dtype);
    }
    return output;
}
```

## 3. KernelBench Standard Code Format

**Important**: The resulting code must follow the KernelBench format specifications, using the**Python module embedded in the C++ code**.

### Full examples of templates

Reference Example position: `skills/cpp/docs/examples/torch_xxx_kernel.py`

```python
import torch
from torch.utils.cpp_extension import load_inline

# Inline C++ Extension Code
cpp_source = """
#include <torch/extension.h>

torch::Tensor op_name_kernel(torch::Tensor x) {
    // 1. Ensure that the input of tensor is continuous
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    // 2. Check data type to support multiple types
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64 &&
                        dtype != torch::kInt32 && dtype != torch::kInt64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;

    // Create output tensor
    torch::Tensor output = torch::zeros_like(input);

    // 4. Based on data type distribution
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            // Specific operator calculation logic
            out_ptr[i] = compute_logic(x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        // Same logic, but use doule type
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = compute_logic(x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt32) {
        auto x_ptr = input.data_ptr<int32_t>();
        auto out_ptr = output.data_ptr<int32_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = compute_logic(x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt64) {
        auto x_ptr = input.data_ptr<int64_t>();
        auto out_ptr = output.data_ptr<int64_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = compute_logic(x_ptr[i]);
        }
    }

    // 5. Conversion back to original type
    if (need_convert) {
        output = output.to(dtype);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("op_name_kernel", &op_name_kernel, "CPU op_name operator");
}
"""

# Dynamically load C++ extensions
op_name_module = load_inline(
    name="custom_op_name",
    cpp_sources=cpp_source,
    extra_cflags=["-O3"],
    verbose=True
)

# Python Interface Functions
def op_name(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu":
        x = x.cpu()
    return op_name_module.op_name_kernel(x)
```

### Key points

1. **Embedded C++ code**: contains the full C++ source code using a three-quote string
2. **Dynamic compile**: use `load_inline` to dynamic compile and load extensions
3. **PYBIND11 Registration**: must register operator with `PYBIND11_MODULE` Macro
4. **Python Interface**: Provides concise Python function packaging
5. **Does not contain the test code**: do not include any test code in the resulting code

## 4. Three basic programming modes

### 4.1 Element-level mode of operation

Applies to simple operations such as activation function, element-by-fact operation.

```cpp
// ReLU: max(0, x)
torch::Tensor relu_kernel(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;
    torch::Tensor output = torch::zeros_like(input);

    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0, x_ptr[i]);
        }
    }

    if (need_convert) output = output.to(dtype);
    return output;
}
```

### 4.2 Conclude operational modalities

Applies to sum, max, min.

```cpp
// Sum reduction: Summation along the specified dimensions
torch::Tensor sum_reduction_kernel(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;

    int64_t numel = input.numel();
    torch::Tensor output;

    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        float result = 0.0f;
        for (int64_t i = 0; i < numel; ++i) {
            result += x_ptr[i];  // Peace be with you.
        }
        output = torch::tensor({result}, torch::kFloat32);
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        double result = 0.0;
        for (int64_t i = 0; i < numel; ++i) {
            result += x_ptr[i];
        }
        output = torch::tensor({result}, torch::kFloat64);
    }

    if (need_convert) output = output.to(dtype);
    return output;
}
```

### 4.3 Border security disposal model

Ensure that all operations have proper border checks and error processing.

```cpp
torch::Tensor safe_operation_kernel(torch::Tensor x) {
    // 1. Examination of tensor validity
    TORCH_CHECK(x.numel() > 0, "Input tensor cannot be empty");
    TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");

    // 2. Ensuring continuity of tensor
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    // 3. Type checks and conversions
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;
    torch::Tensor output = torch::zeros_like(input);

    // 4. Secure data processing
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();

        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();

        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0, x_ptr[i]);
        }
    }

    if (need_convert) output = output.to(dtype);
    return output;
}
```

## 5. Core API Reference

### tensor Type Check & Convert

```cpp
// Type check
torch::ScalarType dtype = x.scalar_type();
bool is_float32 = (dtype == torch::kFloat32);
bool is_float64 = (dtype == torch::kFloat64);
bool is_int32 = (dtype == torch::kInt32);
bool is_int64 = (dtype == torch::kInt64);

// Type Conversion
torch::Tensor input = x.to(torch::kFloat32);  // Convert to float32
torch::Tensor output = result.to(dtype);      // Convert back to the original type
```

### Continuous inspection

```cpp
if (!x.is_contiguous()) {
    x = x.contiguous();
}
```

### Data pointer acquisition

```cpp
// Float32 Pointer
auto x_ptr = input.data_ptr<float>();
auto out_ptr = output.data_ptr<float>();

// Float64 Pointer
auto x_ptr = input.data_ptr<double>();
auto out_ptr = output.data_ptr<double>();

// Int32 Pointer
auto x_ptr = input.data_ptr<int32_t>();
auto out_ptr = output.data_ptr<int32_t>();

// Int64 Pointer
auto x_ptr = input.data_ptr<int64_t>();
auto out_ptr = output.data_ptr<int64_t>();
```

### tensor Creation and Properties

```cpp
// Create output tensor
torch::Tensor output = torch::zeros_like(input);  // The sameshapeniltensor
torch::Tensor output = torch::ones_like(input);   // The sameshapeUnitstensor
torch::Tensor output = input.clone();             // Cloningtensor

// tensor Properties
int64_t numel = input.numel();           // Total number of elements
int64_t dim = input.dim();               // Dimensions
torch::IntArrayRef shape = input.sizes(); // shape
```

### Border checks

```cpp
TORCH_CHECK(x.numel() > 0, "Input tensor cannot be empty");
TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");
```

## 6. Programming constraints and best practice

### Rules to be followed

1. **Boundary check**: all arrays must check the border before visiting
2. **Type security**: ensure that the type of pointer matches the type of tensor
3. **Consequence assurance**: ensuring continuity of tensor memory prior to processing
4. **Type support**: priority support for float32/float64/int32/int64, other types of automatic conversion

### Principles of kernel design

1. **Single function**: One thing for each function
2. **Parameters Simple**: Avoid complex data structure transfer
3. **Avoid dynamic distribution**: kernel avoid new/delete
4. **Clear note**: Add sufficient note to explain the calculation logic

### OpenMP Parallel Programming Constraint

1. **⚠️ Key Constraint**: OpenMPruntimeAPI call position limit
   - **No scenes allowed**:Don't be here.SIMDRegional, parallel regionsintervening codeMedium Call`omp_get_thread_num()`Wait.OpenMPruntimeAPI
   - **Correct use**: OpenMP API should be called in the normal code path in the parallel area, not in the restricted context of compiler
   - **Example of error**:
     ```cpp
     // ❌ error: Call OpenMP API in restricted context
     std::mt19937 gen(seed + omp_get_thread_num());  // Compiler error!
     ```
   - **Correct example**:
     ```cpp
     // ✅ Correct: Normal Call within Parallel Area
     #pragma omp parallel
     {
         int tid = omp_get_thread_num();  // Correct.
         std::mt19937 gen(seed + tid);
     }
     ```
2. **Language security**: ensuring an example of an independent random number generator for each thread
3. **Data competition**: Avoid multiple threads writing the same memory position at the same time

### Code Style Requirements

1. **Does not contain test code**: Generated code does not contain any test code
2. **Embedded C++ format**: C++ code must be written in three quote strings
3. **Keep format clear**: appropriate indentation and line breaks
4. **Descriptive naming**: using clear variable and function names

## 7. More Example References

For more complete examples of operator realization, please refer to:

- **Base document**: `skills/cpp/docs/basic_docs.md`
- **optimization recommendation**: `skills/cpp/docs/suggestion_docs.md`
- **API Handbook**: `skills/cpp/docs/api/api.md`
- **Code template**: `skills/cpp/docs/examples/torch_xxx_kernel.py`

These documents provide a complete guide and reference template for implementation.
