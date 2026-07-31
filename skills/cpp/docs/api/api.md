# CPU C++ API reference manual

This document provides detailed references for the CPU C++ core API, including functional signatures, parameter descriptions and examples of how to use them.

## 1. PyTorch C++ Extension API

### PYBIND11_MODULE Macro
```cpp
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("op_name", &op_function, "Description");
}
```
- **Use**: register custom operator to PyTorch
- **Parameter**: Module Object, operator Name, Function Pointer, Description
- **Request**: This macro must be used to register operator
- **Note: `TORCH_EXTENSION_NAME` is a predefined macro

### tensor type check
```cpp
torch::ScalarType dtype = x.scalar_type();
bool is_float32 = (dtype == torch::kFloat32);
bool is_float64 = (dtype == torch::kFloat64);
bool is_int32 = (dtype == torch::kInt32);
bool is_int64 = (dtype == torch::kInt64);
```
- **Use**: Check the data type
- **Support type**: kFloat32, kFloat64, kInt8, kInt16, kInt32, kInt64
- **Note: It is recommended that priority be given to float32/float64 and int32/int64

### data type conversion
```cpp
torch::Tensor input = x.to(torch::kFloat32);  // Convert tofloat32
torch::Tensor output = result.to(dtype);      // Convert back to the original type
```
- **Use**: Unified data type processing
- **Recommendation**: Internal calculations use float32, eventually reverting to the original type

### tensor Continuous inspection
```cpp
if (!x.is_contiguous()) {
    x = x.contiguous();
}
```
- **Pilot**: Ensure continuous memory layout of tensor
- **Importance**: Avoiding problems with non-continuous memory access

## 2. data type pointer fetch
```cpp
// Float32 pointer
auto x_ptr = input.data_ptr<float>();
auto out_ptr = output.data_ptr<float>();

// Float64 pointer
auto x_ptr = input.data_ptr<double>();
auto out_ptr = output.data_ptr<double>();

// Int32 pointer
auto x_ptr = input.data_ptr<int32_t>();
auto out_ptr = output.data_ptr<int32_t>();

// Int64 pointer
auto x_ptr = input.data_ptr<int64_t>();
auto out_ptr = output.data_ptr<int64_t>();
```
- **Turn**: Get tensor data pointer for direct operations
- **Note: It is recommended that priority be given to float32/float64 and int32/int64 to ensure that tensor types match

## 3. tensor Operation API

### Create tensor
```cpp
torch::Tensor output = torch::zeros_like(input);  // Create Sameshapeniltensor
torch::Tensor output = torch::ones_like(input);   // Create SameshapeUnitstensor
torch::Tensor output = input.clone();             // Cloningtensor
```

### tensor Properties
```cpp
int64_t numel = input.numel();           // Total number of elements
int64_t dim = input.dim();               // Dimensions
torch::IntArrayRef shape = input.sizes(); // shape
torch::Device device = input.device();   // device
```

### tensor Authentication
```cpp
TORCH_CHECK(x.numel() > 0, "Input tensor cannot be empty");
TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");
```

## 4. Mathematics API

### Basic Operations
```cpp
// Maximum value
float result = std::max(0.0f, x_ptr[i]);
double result = std::max(0.0, x_ptr[i]);

// Absolute value [u]
float result = std::abs(x_ptr[i]);

// Square root
float result = std::sqrt(x_ptr[i]);
```

### Conditional Operations
```cpp
// Triple Operators
float result = (x_ptr[i] > threshold) ? x_ptr[i] : 0.0f;

```

## 5. Type Conversion API

### Visible Type Conversion
```cpp
// Float 32 to float 64
double val = static_cast<double>(float_val);

// Int32 to float32
float val = static_cast<float>(int_val);

// Float 32 turn 32.
int32_t val = static_cast<int32_t>(float_val);
```

### data type judgement
```cpp
bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64 &&
                    dtype != torch::kInt32 && dtype != torch::kInt64);
```
