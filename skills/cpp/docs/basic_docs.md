# CPU C++ programming basic course

This document presents the core CPU C++ concept and standard programming model, which helps to understand how to construct an efficient kernel with detailed examples.

## 1. Core concepts

### Kernel
- **Definition**: using the C++ function registered with `PYBIND11_MODULE`, compiled and executed on CPU
- **Characteristics**: Directly operating tensor data pointers to support multiple data types

### tensor Processing
- **Consequence**: ensure continuity of the tensor RAM layout and avoid non-continuing access
- **System harmonization**: internal calculations use a uniform type, and final conversion back to the original type
- **Boundary check**: all arrays must check the border before visiting

### Memory management
- **Automandate**: PyTorch Automanaging tensor Memory Life Cycle
- **pointer operation**: direct operation data pointer for efficient calculation
- **Type security**: ensure that the type of pointer matches the type of tensor

## 2. Standard kernel structure

The CPU C++ cores follow the same five-step structure model:

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

    // 4. Based on data type manual calculations
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
    }

    // 5. Conversion back to original type
    if (need_convert) {
        output = output.to(dtype);
    }
    return output;
}
```

## 3. Programming Mode

### 3.1 Element-level operating modalities
Applies to simple operations such as activation function, element-by-fact operation.

```cpp
torch::Tensor elementwise_kernel(torch::Tensor x) {
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
            out_ptr[i] = std::max(0.0f, x_ptr[i]);  // ReLU: max(0, x)
        }
    }

    if (need_convert) output = output.to(dtype);
    return output;
}
```

### 3.2 Routine mode of operation
Applies to sum, max, min.

```cpp
torch::Tensor reduction_kernel(torch::Tensor x) {
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
    }

    if (need_convert) output = output.to(dtype);
    return output;
}
```

## 4. Example of boundary processing

### tensor border check
```cpp
torch::Tensor safe_tensor_operation(torch::Tensor x) {
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
            // Border checks: ensure that the index is valid
            if (i < numel) {
                out_ptr[i] = std::max(0.0f, x_ptr[i]);
            }
        }
    }

    if (need_convert) output = output.to(dtype);
    return output;
}
```