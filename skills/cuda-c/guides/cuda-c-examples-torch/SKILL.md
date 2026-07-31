---
name: cuda-c-examples-torch
description: "PyTorch + CUDA C full integration example code"
category: example
version: "1.0.0"
metadata:
  backend: cuda
  dsl: cuda_c
  framework: torch
  examples: "vector_add, relu, matmul, softmax, layernorm, double_kernel"
---

# PyTorch + CUDA C Example Code

This Skill contains a full runable example code showing how to use CUDA C in PyTorch to write high performance kernel, compiled and integrated through `load_inline` JIT.

## Integrated Mode

All CUDA C kernels are integrated with PyTorch in the following mode:

```python
import torch
from torch.utils.cpp_extension import load_inline

# CUDA source code (kernel definition + call function)
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(const float* input, float* output, int size) {
    // kernel realization
}

torch::Tensor my_kernel_call(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    my_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

# 2. C++ Function Declaration
cpp_source = "torch::Tensor my_kernel_call(torch::Tensor input);"

# 3. JIT compilation
module = load_inline(
    name="my_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["my_kernel_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# 4. Calls
def my_op(x):
    return module.my_kernel_call(x)
```

## Example List

### 1. vector Add
**operator type**: Element-wise
**Key points**:
- Simplest CUDA C kernel example
- 1-D index and border check
- Standard five-step model

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add_call(torch::Tensor a, torch::Tensor b) {
    auto n = a.numel();
    auto c = torch::empty_like(a);
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    vector_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), n);
    return c;
}
"""

cpp_source = "torch::Tensor vector_add_call(torch::Tensor a, torch::Tensor b);"

module = load_inline(
    name="vector_add_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["vector_add_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def vector_add(a, b):
    return module.vector_add_call(a, b)
```

### 2. ReLU Activate Function
**operator type**: Element-wise
**Key points**:
- Use `fmaxf` built-in functions
- A simple element-by-element operation

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

torch::Tensor relu_call(torch::Tensor input) {
    auto n = input.numel();
    auto output = torch::empty_like(input);
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

cpp_source = "torch::Tensor relu_call(torch::Tensor input);"

module = load_inline(
    name="relu_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["relu_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def relu(x):
    return module.relu_call(x)
```

### 3. matrix multiplication (MatMul)
**operator type**: MatMul
**Key points**:
- 2D Thread Configuration `dim3`
- shared memory Optimization (Bulk Loading)
- `__syncthreads()` Sync

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_call(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K);
    return C;
}
"""

cpp_source = "torch::Tensor matmul_call(torch::Tensor A, torch::Tensor B);"

module = load_inline(
    name="matmul_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def matmul(A, B):
    return module.matmul_call(A, B)
```

### 4. Softmax
**operator type**: Reduce + Element-wise
**Key points**:
- Line processing (one block per line)
- Numerical stability (maximum reduction first)
- shared memory Convention
- Grid-stride loop processing lines

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void softmax_kernel(
    const float* input, float* output, int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // 1. Looking for maximum value
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_in[i]);
    }
    // Warp Return
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // 2. Calculate exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += __expf(row_in[i] - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    // 3. Normalization
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_out[i] = __expf(row_in[i] - max_val) / sum;
    }
}

torch::Tensor softmax_call(torch::Tensor input) {
    auto sizes = input.sizes();
    int rows = sizes[0];
    int cols = sizes[1];
    auto output = torch::empty_like(input);

    softmax_kernel<<<rows, 256>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

cpp_source = "torch::Tensor softmax_call(torch::Tensor input);"

module = load_inline(
    name="softmax_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def softmax(x):
    return module.softmax_call(x)
```

### 5. Layer Norm
**operator type**: Reduce + Element-wise
**Key points**:
- Multiple scans (average → variance → unified)
- Warp-level Convention + shared memory
- `rsqrtf` Quick Calculating Standard Difference Last

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layer_norm_kernel(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // 1. Calculation of averages
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += row_in[i];
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / cols;
    __syncthreads();
    float mean = s_mean;

    // 2. Differences in calculation
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_in[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(var_sum / cols + eps);
    __syncthreads();
    float rstd = s_rstd;

    // 3. Normalization
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float normalized = (row_in[i] - mean) * rstd;
        row_out[i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor layer_norm_call(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps
) {
    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty_like(input);

    layer_norm_kernel<<<rows, 256>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        rows, cols, eps);
    return output;
}
"""

cpp_source = """
torch::Tensor layer_norm_call(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);
"""

module = load_inline(
    name="layernorm_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["layer_norm_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def layer_norm(x, gamma, beta, eps=1e-5):
    return module.layer_norm_call(x, gamma, beta, eps)
```

### 6. Double ReLU + Square
**operator type**: Multi Kernel group
**Key points**:
- Call multiple Kernels in one function
- Intermediate results passed by tensor
- A separate configuration grid for each kernel

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

__global__ void square_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    }
}

torch::Tensor relu_square_call(torch::Tensor input) {
    auto n = input.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // First kernel: ReLU
    auto intermediate = torch::empty_like(input);
    relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), intermediate.data_ptr<float>(), n);

    // Second Kernel: Square
    auto output = torch::empty_like(input);
    square_kernel<<<num_blocks, block_size>>>(
        intermediate.data_ptr<float>(), output.data_ptr<float>(), n);

    return output;
}
"""

cpp_source = "torch::Tensor relu_square_call(torch::Tensor input);"

module = load_inline(
    name="relu_square_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["relu_square_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def relu_square(x):
    return module.relu_square_call(x)
```

## General model summary

### CUDA source code Structure
```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

// 1. Definition of kernel
__global__ void kernel_name(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = /* Calculate */;
    }
}

// 2. Call function (return to Torch:::Tensor)
torch::Tensor kernel_call(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    kernel_name<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
```

### Python Integrated Structure
```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = "..."  # CUDA source codeString
cpp_source = "torch::Tensor kernel_call(torch::Tensor input);"

module = load_inline(
    name="op_name_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["kernel_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def op_function(x):
    return module.kernel_call(x)
```

## Key note

### 1. Must use load_inline
The CUDA C kernel must be compiled by JIT through `torch.utils.cpp_extension.load_inline`, with CUDA C code embedded in Python.

### 2. Output tensor Create
```cuda
// ✅ Recommendations: create output using the torch function
auto output = torch::empty_like(input);
auto output = torch::zeros_like(input);
auto output = torch::empty({M, N}, input.options());
```

### 3. Data pointer acquisition
```cuda
// Select template parameters according to data type
input.data_ptr<float>()     // float32
input.data_ptr<double>()    // float64
input.data_ptr<at::Half>()  // float16
```

### 4. ⚠ ️ prohibition
- **Test code banned**: does not contain `main()` functions or test clips
- **Ban printing**: not using `printf()`
- **Ban on anomalies**: not using `throw std::runtime_error()`

## Validate correctness

```python
# Compare to PyTorch Native
x = torch.randn(128, 256, device='cuda', dtype=torch.float32)

output_cuda = module.kernel_call(x)
output_torch = torch_reference(x)

diff = (output_cuda - output_torch).abs().max()
print(f"Max difference: {diff.item()}")
assert diff < 1e-5, "Results mismatch!"
```
