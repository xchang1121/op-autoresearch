---
name: cuda-c-api
description: "CUDA C programming interface complete reference manual"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: cuda_c
---

# CUDA C API Reference Manual

This document provides detailed references to the CUDA C core programming interface, including functional signatures, parameter descriptions and examples of use.

## 1. Function Modifiers

### __global__
```cuda
__global__ void kernel_function(List of parameters);
```
- **Function**: Mark as kernel to execute on GPU
- **Call**: Only from host code, started using `<<<>>>` syntax
- **return value**: must be `void`

### __device__
```cuda
__device__ float device_function(List of parameters);
```
- **Function**: Mark to device function performed on GPU
- **Call**: Only from other `__device__` or `__global__` functions
- **Use**: Auxiliary functions within the kernel

### __host__
```cuda
__host__ void host_function(List of parameters);
```
- **Function**: Mark as a function executed on CPU (default)
- **Call**: Only from host code

### __host__ __device__
```cuda
__host__ __device__ float utility_function(float x);
```
- **Function**: also available on CPU and GPU
- **Practice**: Common tool function

## 2. Memory Type Modifier

### __shared__
```cuda
__shared__ float shared_memory[256];
```
- **Function**: Declaration of memory shared within line blocks
- **Life cycle**: Same as the thread block
- **Arrival**: All threads within a block readable
- **Capacity**: normal 48-164 KB/SM

### __constant__
```cuda
__constant__ float constant_data[64];
```
- **Function**: Declaration of read-only constant memory
- **Feature**: Cache Optimization for Broadcast Reader
- **Settings**: through `cudaMemcpyToSymbol` from host side Settings

### extern __shared__
```cuda
extern __shared__ float dynamic_shared[];
```
- **Function**: dynamically allocated shared memory
- **Size**: specify by third parameter on inner core startup
- **Start**: `kernel<<<grid, block, shared_mem_bytes>>>(args)`

## 3. kernel startup syntax

### Basic Syntax:
```cuda
kernel_name<<<grid_size, block_size>>>(List of parameters);
kernel_name<<<grid_size, block_size, shared_mem_bytes>>>(List of parameters);
kernel_name<<<grid_size, block_size, shared_mem_bytes, stream>>>(List of parameters);
```
- **grid_size**: grid size (`int` or `dim3`)
- **block_size**: Thread block size (`int` or `dim3`)
- **shared_mem_bytes**: Dynamic shared memory Size (optional, default 0)
- **stream**: CUDA streams (optional, default 0)

### dim3 type
```cuda
dim3 grid_size(blocks_x, blocks_y, blocks_z);
dim3 block_size(threads_x, threads_y, threads_z);
```
- **Pilot**: multi-dimensional grid and line block configuration
- **Default**: Unspecified dimensions default to 1

## 4. Thread and block indexing system

### Block Index Variables
```cuda
int bx = blockIdx.x;   // X Direction block index
int by = blockIdx.y;   // Y Direction block index
int bz = blockIdx.z;   // Z Direction block index
```
- **Type**: `uint3`
- **Turn**: determine the position of the current thread in the grid

### Thread Index Variables
```cuda
int tx = threadIdx.x;  // X Directional Line Index
int ty = threadIdx.y;  // Y Directional Line Index
int tz = threadIdx.z;  // Z Directional Line Index
```
- **Type**: `uint3`
- **Pilot**: Determines the position of the current thread online block

### Block dimension information
```cuda
int bdx = blockDim.x;  // X Number of directional steps
int bdy = blockDim.y;  // Y Number of directional steps
int bdz = blockDim.z;  // Z Number of directional steps
```
- **Type**: `dim3`
- **Use**: Get the dimensions of a thread block

### Grid dimension information
```cuda
int gdx = gridDim.x;   // X Number of heading blocks
int gdy = gridDim.y;    // Y Number of heading blocks
int gdz = gridDim.z;   // Z Number of heading blocks
```
- **Type**: `dim3`
- **Use**: Get the dimensions of the grid

## 5. Memory Management API

### cudaMalloc(devPtr, size)
```cuda
float* d_data;
cudaMalloc(&d_data, n * sizeof(float));
```
- **Parameter**: address of device pointer, distribution bytes
- **Return**: `cudaError_t` error status code
- **Function**: Allocation of global memory on GPU

### cudaFree(devPtr)
```cuda
cudaFree(d_data);
```
- **Parameter**: device memory pointer
- **Return**: `cudaError_t` error status code
- **Function**: Release GPU global memory

### cudaMemcpy(dst, src, count, kind)
```cuda
// host → device.
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
// device→host
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
// device → device.
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
```
- **Parameter**: Target Pointer, Source Pointer, Bytes, Transfer Direction
- **Return**: `cudaError_t` error status code
- **Transfer direction**:
  - `cudaMemcpyHostToDevice`: H2D
  - `cudaMemcpyDeviceToHost`: D2H
  - `cudaMemcpyDeviceToDevice`: D2D

### cudaMemset(devPtr, value, count)
```cuda
cudaMemset(d_data, 0, n * sizeof(float));
```
- **Parameter**: device pointer, fill value (bytes), bytes
- **Function**: Set device memory to a specified value

## 6. Thread Synchronization Mechanism

### __syncthreads()
```cuda
__syncthreads();
```
- **Function**: Waiting for all threads in the block to reach sync point
- **Use**: Ensure consistency of shared memory data
- **⚠ ️ Note: All threads within the block must be executed to this synchronous point, otherwise it will result in a dead lock

### __threadfence()
```cuda
__threadfence();
```
- **Function**: ensure that the memory of the current thread is written in a global view of all threads
- **Pilot**: Data Visibility Assurance for Cross-line Blocks

### __threadfence_block()
```cuda
__threadfence_block();
```
- **Function**: ensure that the memory of the current thread is written to be visible to the same thread
- **Use**: Memory Visibility Assurance in Blocks

## 7. Atomic Operation API

### atomicAdd(address, val)
```cuda
atomicAdd(&output[0], local_sum);
```
- **Function**: Thread Safety Plus Operation
- **Support type**: `int`, `unsigned int`, `float`, `double` (Compute Capitality 6.0+)

### atomicMax / atomicMin
```cuda
atomicMax(&max_val, local_max);
atomicMin(&min_val, local_min);
```
- **Function**: Maximum/Minimum Thread Security Update
- **Support type**: `int`, `unsigned int`

### atomicCAS(address, compare, val)
```cuda
int old = atomicCAS(&target, expected, desired);
```
- **Function**: Comparison-And-Swap
- **Return**: Original

### atomicExch(address, val)
```cuda
float old = atomicExch(&target, new_value);
```
- **Function**: Atomic Exchange Operations
- **Return**: Original

## 8. Mathematical Operations Functions

### Standard Mathematical Functions
```cuda
float max_val = fmaxf(a, b);      // Maximum value
float min_val = fminf(a, b);      // Min
float abs_val = fabsf(x);         // Absolute value [u]
float sqrt_val = sqrtf(x);        // Square root
float rsqrt_val = rsqrtf(x);     // Square root countdown
float exp_val = expf(x);          // Index
float log_val = logf(x);          // Natural log [N]
float pow_val = powf(base, exp);  // Logic Operations
float ceil_val = ceilf(x);        // Lift Up
float floor_val = floorf(x);      // Sweep Down
```

### Rapid Math function (accuracy slightly lower but faster)
```cuda
float fast_exp = __expf(x);       // Quick Index
float fast_log = __logf(x);       // Fast logarithm
float fast_sin = __sinf(x);       // QuickSine
float fast_cos = __cosf(x);       // Quick Cosine
float fast_pow = __powf(b, e);    // Quick Crunch Operations
```

### Type Conversion
```cuda
float f = __int2float_rn(i);      // int → float(Recent rounded)
int i = __float2int_rn(f);        // float → int(Recent rounded)
```

## 9. Warp Level Operations (Compute Capitalisation 3.0+)

### __shfl_sync / __shfl_down_sync
```cuda
// Data Interchange in Warp
float val = __shfl_sync(0xFFFFFFFF, src_val, src_lane);
// Warp Internal Convention
float val = __shfl_down_sync(0xFFFFFFFF, src_val, offset);
```
- **Function**: direct data interchange between Warp inner processes
- **Parameter**: mask (participated thread mask), source value, source light/ offset

## 10. PyTorch Integration API

### Torch::Tensor operation
```cuda
// Use PyTorch interface in CUDA source code
torch::Tensor my_op(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    my_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    return output;
}
```

### Common torch::Tensor method
```cuda
input.numel()              // Total number of elements
input.size(dim)            // Specify the dimension size
input.data_ptr<float>()    // Get Data Pointer
torch::zeros_like(input)   // Create Sameshapeniltensor
torch::empty_like(input)   // Create SameshapeNot initializedtensor
```

## Use recommendations

1. **Boundary check**: all arrays must check the border before visiting
2. **Atomic operations**: use only when necessary, performance costs
3. **Rapid Mathematical Functions**: accuracy requires acceleration using `__expf`, etc.
4. **shared memory**: `__syncthreads()` Synchronize before and after use
5. **Error check**: Check return value after every CUDA API call
6. **Memory alignment**: ensure that data are aligned to the appropriate boundary (128 bytes)
