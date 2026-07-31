---
name: cuda-c-patterns
description: "CUDA C three major programming modes: vector operation, contract return, matrix multiplication"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: cuda_c
  operator_patterns: "elementwise, reduce, matmul"
---

# CUDA C Programming Mode

## 1. vector operating mode

For element-level operations: Adding, Multiplication, Activation Functions, etc.

### Standard code structure

```cuda
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Apply operator
- Algorithmic Operations: add, Mul, sub, div
- Activate function: relu, sigmoid, tan, gelu, silu
- Mathematical functions: exp, log, sqrt, pow, abs
- Type conversion:cast
- Broadcast operation: Broadcast

### Key points
- Use 1-D index `blockIdx.x * blockDim.x + threadIdx.x`
- Border check `if (idx < n_elements)`
- Simple direct data stream: Load → calculate → storage
- Recommended block size: 256 or 512

### ReLU Example

```cuda
__global__ void relu_kernel(
    const float* input, float* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}
```

### GELU Example

```cuda
__global__ void gelu_kernel(
    const float* input, float* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // ApproximationGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}
```

### Multi-Input Element-by-Element Operations

```cuda
__global__ void fused_multiply_add_kernel(
    const float* a, const float* b, const float* c,
    float* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] * b[idx] + c[idx];
    }
}
```

## 2. Reunification Mode

Applies to sum, max, min.

### Standard code structure (conscription of shared memory)

```cuda
__global__ void reduction_sum_kernel(
    const float* input, float* output, int n_elements
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data to shared memory
    sdata[tid] = (idx < n_elements) ? input[idx] : 0.0f;
    __syncthreads();

    // Consistency of blocks (regulating of trees)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // First thread to write a block result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
```

### Apply operator
- Som, mean, max, min, prod
- Normalize: softmax, logsoftmax, playnorm, watchnorm, rmsnorm
- Statistics: varance, std
- Search: argmax, argmin

### Key points
- Use `extern __shared__` to declare dynamic shared memory
- Tree return (half times each)
- `__syncthreads()` to ensure data consistency
- `atomicAdd` for global return across block
- ⚠ ️, the synchronous point must be reached by all threads.

### Softmax example (value stabilization version)

```cuda
__global__ void softmax_kernel(
    const float* input, float* output, int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // 1. Search for maximum value (stable value)
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_input[i]);
    }

    // Warp-incorporated maximum value
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }

    // Cross-wrap by shared memory
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // 2. Calculate exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += __expf(row_input[i] - max_val);
    }

    // Reunification Sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    // 3. Calculation of softmax
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_output[i] = __expf(row_input[i] - max_val) / sum;
    }
}
```

### LayerNornm Example

```cuda
__global__ void layer_norm_kernel(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // 1. Calculation of averages
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += row_input[i];
    }
    // Warp Return
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
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }
    __shared__ float s_var;
    if (threadIdx.x == 0) s_var = var_sum / cols;
    __syncthreads();
    float rstd = rsqrtf(s_var + eps);

    // 3. Normalization
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float normalized = (row_input[i] - mean) * rstd;
        row_output[i] = normalized * gamma[i] + beta[i];
    }
}
```

## 3. matrix multiplication mode

For multi-dimensional block calculations such as matrix multiplication.

### Standard code structure (in plain form)

```cuda
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### shared memory Optimization

```cuda
#define TILE_SIZE 16

__global__ void matmul_shared_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load to shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        // Calculated partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Apply operator
- Matrix operation: matmul, bmm (batch matmul), linear
- Volume: conv2d, conv3d
- Attention:

### Key points
- **2D Grid**: Configure 2D grids and thread blocks using `dim3`
- **shared memory Optimization**: segment load reduced global memory access
- **Sync**: `__syncthreads()` after each segment load
- **Boundary processing**: cross-border location fill 0
- **Block size**: commonly used 16x16 or 32x32

### Host Side Start

```cuda
// PARK is on.
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);

// shared memory release activated.
dim3 block(TILE_SIZE, TILE_SIZE);
dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
matmul_shared_kernel<<<grid, block>>>(A, B, C, M, N, K);
```

## Mode Selection Guide

| operator Type | Recommended Mode | Key features | Block Size |
|---------|---------|---------|-------|
| Element-wise | vector Operations | Element-by-Element calculation | 256/512 |
| Reduction | Reunification Mode | Multiple values need to be aggregated | 256 |
| MatMul/Conv | matrix multiplication | Multi-dimensional block calculations, 2D Grid | 16x16/32x32 |
| Softmax/Norm | Reunification+Element Operations | Line attribution + element by element | 256 |
| Attention | Group Mode | MatMul + Softmax | It depends. |

## best practice

1. **Select the appropriate mode**: Select the base mode based on operator characteristics
2. **Optimizing block size**: balancing parallelity and resource occupancy
3. **Note boundary**: use `if (idx < n)` to handle irregular shape
4. **Numerical stability**: maximum excretion pre-decreasing operator
5. **Memorial access**: ensure that access is combined, using shared memory cache
6. **Sync security**: `__syncthreads()` must reach all threads
7. **Decreasing atom operations**: pre-correspondence before global writing back
