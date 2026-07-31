---
name: cuda-c-basics
description: "CUDA C Core Concept, kernel structure and standard programming model"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: cuda_c
  operator_patterns: "all"
---

# CUDA C Programming Base

## 1. Core concepts

### Kernel
- **Definition**: A C/C++ function modified by `__global__`, executed in parallel with GPU
- **Feature**: A subset of data processed for each kernel example, distinguished by a linear index
- **Call**: Start from host code using `<<<grid_size, block_size>>>` syntax

### Grid (Grid) and Block (Block)
- **Grid**: Parallel dimensions configuration at kernel start-up, e. g. `(num_blocks_x, num_blocks_y)`
- **Block**: Number of threads contained in each thread block, e.g. `block_size = 256`
- **Relationship**: `grid_size = ceil(total_elements / block_size)`
- **Restrict**: maximum 1024 threads per block (most GPU)

### Thread Level
- **Grid**: collection of all threads
- **Block**: a group of collaborative threads (shared memory, Sync)
- **Warp**: 32 threads for a set of parallel implementations (SIMT Implementation Model)
- **Thread**: minimum implementation module

### Memory Level
- **global memory (Global Memoory)**: All threads accessible, latency high, high capacity
- **shared memory (Shared Memoory)**: Block interior distance sharing, latency low, limited capacity (usually 48-164 KB/SM)
- **Registers**: private, quickest access for each thread
- **Constant Memory**: read-only, cache optimization
- **Texture Memory**: read-only, space localized optimization

## 2. Standard kernel structure (five-step model)

All CUDA C kernels follow the same five-step structure model:

```cuda
__global__ void standard_kernel(
    float* output, float* input, int n_elements
) {
    // 1. Calculation of global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Border inspections
    if (idx < n_elements) {
        // 3. Loading data
        float data = input[idx];

        // 4. Implementation calculations
        float result = compute_function(data);

        // 5. Storage results
        output[idx] = result;
    }
}
```

### kernel startup mode

```cuda
void launch_kernel(float* input, float* output, int n_elements) {
    const int block_size = 256;
    const int num_blocks = (n_elements + block_size - 1) / block_size;

    kernel<<<num_blocks, block_size>>>(output, input, n_elements);
}
```

## 3. Global index calculation

### One-dimensional data processing
```cuda
int global_index = blockIdx.x * blockDim.x + threadIdx.x;
```

### Two-dimensional data processing
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Three-dimensional data processing
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

### Grid Configuration

```cuda
// One-dimensional grid
int block_size = 256;
int num_blocks = (n_elements + block_size - 1) / block_size;
kernel<<<num_blocks, block_size>>>(...);

// 2D grid (matrix operation)
dim3 block_size(16, 16);
dim3 grid_size((N + 15) / 16, (M + 15) / 16);
kernel<<<grid_size, block_size>>>(...);

// 3D grid (volume data)
dim3 block_size(8, 8, 8);
dim3 grid_size((X + 7) / 8, (Y + 7) / 8, (Z + 7) / 8);
kernel<<<grid_size, block_size>>>(...);
```

## 4. Border processing

### Basic border checks
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n_elements) {
    // Security access input [idx]
    output[idx] = input[idx];
}
```

### 2D border check
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
    // Security access matrix elements
    output[row * N + col] = input[row * N + col];
}
```

## 5. Memory Management Mode

### host-device data transfer
```cuda
// Allocation of device memory
float* d_input, *d_output;
cudaMalloc(&d_input, size * sizeof(float));
cudaMalloc(&d_output, size * sizeof(float));

// Copy data to device
cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

// Launch kernel.
kernel<<<grid, block>>>(d_output, d_input, size);

// Copy results back to host
cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

// Release Memory
cudaFree(d_input);
cudaFree(d_output);
```

## 6. PyTorch Integration

### ⚠ ️ Important: Compile with Python JIT

When generating operator, the CUDA C code must be embedded in the Python module, using `torch.utils.cpp_extension.load_inline` for JIT compilation:

```python
import torch
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
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

cpp_src = "torch::Tensor my_kernel_call(torch::Tensor input);"

kernel_module = load_inline(
    name="my_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["my_kernel_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)
```

## 7. Summary of best practice

### Programming principles
- **Single function**: Each kernel does one thing only.
- **Parameters Simple**: Avoid complex data structure transfer
- **Boundary check**: all arrays must check the border before visiting
- **Memorial Locality**: access to adjacent memory as much as possible

### Common error avoidance
1. **Cross-border visits**: forget that border checks lead to runtime errors or abnormal results
2. **Memory leak**: Forget to call `cudaFree` Release device memory
3. **Sync error**: shared memory operation missing `__syncthreads()`
4. **Type mismatch**: host and device side data type inconsistent
5. **device memory is insufficient**: block size or data exceeding GPU memory

### ⚠ ️ note.
- kernel code generated**do not contain any test code clips**
- **Ban**Print/ Unusual Statements with `printf()`, `throw std::runtime_error()`, etc.
- kernel**does not use**`malloc` / `new` for dynamic distribution
