---
name: sketch-design
description: "operator sketch design language specifications and generation guidance, including UnifiedSketch DSL syntax and operator sketch design methodology"
category: guide
version: "1.0.0"
metadata:
  role: designer
---

# UnifiedSketch Design

## Objectives and principles

### Objective
Expressing the design intent of operator with a minimum DSL to facilitate LLM understanding and Coder ' s realization.

### Principles
- **Pretty original**: only a few core operations (alloc/load/store/compute/...)
- **Uniform syntax**: all operations are function call styles, no syntax differences
- **Standard Control Flow**: no new syntax invented using Python for/range
- **hint separation**: complex optimization expressed in hint without prejudice to master logic clarity

## Core Syntax Elements

### Structure statement
```python
sketch <op_name> {
  symbols: M, N, K;                    # Symbol variable declaration
  tensors: A[M, K]: f16; B[K, N]: f16; C[M, N]: f32;  # tensorStatement
  constexpr: m0, k0, n0
}
```

### @llm_hint Decorator Details

#### Basic Syntax:
`@llm_hint` is used to provide an optimised hint to LLM to help coder select the best implementation strategy.

```python
@llm_hint("optimization_type")              # Single Hint
@llm_hint("optimization_type", "context")   # Take the following hints.
@llm_hint("opt1", "opt2", "opt3")          # Multiple hints
```

#### Optimization Type
- `"parallel"` - Parallelize this cycle
- `"pipeline"` - pipeline Optimization
- `"vectorize"` - vector
- `"unroll"` - Looping

#### Hardware Contexttips
- `"grididx"` - GPU Grid Level Parallel (corresponding to blockIdx)
- `"threadidx"` - GPU thread parallel (corresponding to threadIdx)
- `"coreidx"` - NPU core Level Parallel
- `"warp"` - GPU warp level optimization
- `"simd"` - CPU/NPU Simd vector

### For circular expression
```python
# GPU style: parallel two levels: grid + thread
@llm_hint("parallel", "grididx.x")
for i in range(0, M, 128):                  # blockLevel
    @llm_hint("parallel", "threadidx.x")
    for j in range(0, N, 32):               # threadLevel
        @llm_hint("pipeline")
        for k in range(0, K, k_tile):
            # Calculating Logic

# NPU style: core level parallel
@llm_hint("parallel", "coreidx")
for core_idx in range(num_cores):
    @llm_hint("pipeline")
    for k in range(0, K, k_tile):
        # Calculation for each core

# CPU style: SIMDvector
@llm_hint("parallel")                       # OpenMPParallel
for i in range(0, M, tile_size):
    @llm_hint("vectorize", "simd")          # SIMDvectorDilution
    for j in range(tile_size):
        # vector calculator
```

### Core Operations
1. **alloc**- Memory Allocation
2. **load**- Data loading
3. **store**- Data storage
4. **compute Functions**- Calculating Operations

## Syntax: Overview

```python
sketch matmul {
  symbols: M, N, K;
  tensors: A[M, K]: f16; B[K, N]: f16; C[M, N]: f32;

  m0, k0, n0 = 128, 256, 256

  @llm_hint("parallel")
  for i_outer in range(0, ceil(M, m0)):
    @llm_hint("parallel")
    for j_outer in range(0, ceil(N, n0)):

      # Memory Allocation
      c_tile = alloc([m0, n0], llm_hint=["accumulator", "init_zero"])
      a_tile = alloc([m0, k0], llm_hint=["fast", "input_cache"])
      b_tile = alloc([k0, n0], llm_hint=["fast", "input_cache"])

      @llm_hint("pipeline")
      for k_outer in range(0, ceil(K, k0)):
        # Data Migration
        load(A[i_outer:i_outer+m0, k_outer:k_outer+k0] -> a_tile)
        load(B[k_outer:k_outer+k0, j_outer:j_outer+n0] -> b_tile)

        # Calculator Operations
        gemm(a_tile, b_tile, dst=c_tile)

      # Data Writeback
      store(c_tile -> C[i_outer:i_outer+m0, j_outer:j_outer+n0])
}
```

## Memory management system

### alloc() Syntax:
```python
tile = alloc([shape], llm_hint=["Storage Requirements", "Description of use", "Performance Requirements"])
```

### hint design principles
**Semantic description to allow LLM to select the hardware document to achieve it**

#### Storage requirements (performance level)
- `"fastest"` - Maximum access speed, small capacity (let LLM select register/L0 etc.)
- `"fast"` - Quick Access, Medium Capacity (LLM selects Shared/L1 etc.)
- `"medium"` - Medium Speed, Large Capacity (LLM select L2/cache etc.)
- `"slow"` - Slow but large (let LLM choose global/DR etc.)

#### Purpose statement (to help LLM understand intent)
- `"accumulator"` - Composer with frequent reading and writing requirements
- `"input_cache"` - Input Data Cache, Main Read
- `"output_buffer"` - Output buffer, mainly written
- `"temp_workspace"` - Temporary workspace
- `"shared_between_threads"` - Serial Sharing Data

#### Initialization requirements
- `"init_zero"` - Initialize to 0
- `"no_init"` - Not Initialized (Default)

### Example:
```python
# Semantic hint approach (recommended)
c_acc = alloc([128, 128], llm_hint=["fastest", "accumulator", "init_zero"])
a_cache = alloc([128, 256], llm_hint=["fast", "input_cache"])
temp = alloc([128], llm_hint=["fast", "temp_workspace"])

# LLM will map it from the hardware document to:
# NPU: fastest→L0, fast→L1_buffer
# GPU: fastest→register, fast→shared_memory
# CPU: fastest→register, fast→L1_cache
```

## Data Moving Operation

### load() Syntax:
```python
load(tensor[slice] -> tile)
```

### store() Syntax:
```python
store(tile -> tensor[slice])
```

### Slice expression
```python
# Basic Slice
A[i:i+128, k:k+256]           # A 2D slice.
X[start:end]                  # One-dimensional slices.

# Full tile
A[i_outer:i_outer+m0, k_outer:k_outer+k0]
```

### Example:
```python
load(A[0:128, 0:256] -> a_tile)              # LoadAThe pieces are here.a_tile
store(result_tile -> C[i:i+128, j:j+128])    # Write Results BackCThe pieces.
```

## Calculating Operator Library

### Basic Operations
```python
add(src1, src2, dst)          # dst = src1 + src2
mul(src1, src2, dst)          # dst = src1 * src2
sub(src1, src2, dst)          # dst = src1 - src2
div(src1, src2, dst)          # dst = src1 / src2
max(src1, src2, dst)          # dst = max(src1, src2)
min(src1, src2, dst)          # dst = min(src1, src2)
...
```

### Math Functions
```python
exp(src, dst)                 # dst = exp(src)
log(src, dst)                 # dst = log(src)
sqrt(src, dst)                # dst = sqrt(src)
abs(src, dst)                 # dst = abs(src)
tanh(src, dst)                # dst = tanh(src)
sigmoid(src, dst)             # dst = sigmoid(src)
...
```

### Linear algebra
```python
gemm(a, b, dst)               # dst += a @ b (matrix multiplication)
dot(a, b, result)             # result = dot(a, b) (vectorPoint)
reduce_sum(src, axis, dst)    # dst = sum(src, axis=axis) (AllowaxisaslistIt means multiple axes at the same timereduce)
reduce_max(src, axis, dst)    # dst = max(src, axis=axis)
...
```

### Composite Functions
```python
relu(src, dst)                # dst = max(0, src)
gelu(src, dst)                # dst = gelu(src)
silu(src, dst)                # dst = silu(src) = src * sigmoid(src)
softmax(src, dst)             # dst = softmax(src)
...
```

## Parallel and Optimizing Tips

### Multiple Parameters@llm_hint

#### Parallel mode for different hardware
```python
# GPU: Use two parallels of grid + thread
@llm_hint("parallel", "grididx")      # Correspond blockIdx.x/y/z
@llm_hint("parallel", "threadidx")    # Correspond threadIdx.x/y/z

# NPU: Use core level parallel
@llm_hint("parallel", "coreidx")      # Correspond ai_core Parallel

# CPU: Use thread parallel + SIMD
@llm_hint("parallel")                 # Correspond OpenMP/TBB
@llm_hint("vectorize", "simd")        # Correspond AVX/NEON
```

#### Combine Usage Policy
```python
# Full example of GPU
@llm_hint("parallel", "grididx")
for block_i in range(M_blocks):
    @llm_hint("parallel", "threadidx")
    for thread_j in range(threads_per_block):
        @llm_hint("pipeline")
        for k in range(k_blocks):
            # Calculating Logic

# Full example of NPU
@llm_hint("parallel", "coreidx")
for core_idx in range(num_cores):
    @llm_hint("pipeline")
    for k in range(k_tiles):
        @llm_hint("vectorize")
        for i in range(vector_size):
            # vector calculation
```

## Example of common pattern

### MatMul (e.g. grammatical overview above)

### Elementwise - ReLU
```python
sketch relu {
  symbols: N;
  tensors: X[N]: f32; Y[N]: f32;

  tile_size = 1024

  @llm_hint("parallel")
  for i in range(0, ceil(N, tile_size)):
    x_tile = alloc([tile_size], llm_hint="l1_buffer")
    y_tile = alloc([tile_size], llm_hint="l1_buffer")

    load(X[i:i+tile_size] -> x_tile)
    relu(x_tile, y_tile)
    store(y_tile -> Y[i:i+tile_size])
}
```

### Reduction - Softmax
```python
sketch softmax {
  symbols: B, N;
  tensors: X[B, N]: f32; Y[B, N]: f32;

  @llm_hint("parallel")
  for b in range(B):
    x_row = alloc([N], llm_hint="l1_buffer")
    y_row = alloc([N], llm_hint="l1_buffer")
    max_val = alloc([1], llm_hint="l0c")
    sum_val = alloc([1], llm_hint="l0c")

    load(X[b, 0:N] -> x_row)

    # 3-stage softmax
    reduce_max(x_row, axis=0, max_val)
    sub(x_row, max_val, x_row)        # x = x - max
    exp(x_row, y_row)                 # y = exp(x)
    reduce_sum(y_row, axis=0, sum_val)
    div(y_row, sum_val, y_row)        # y = y / sum

    store(y_row -> Y[b, 0:N])
}
```

### Composite operator - GELU
```python
sketch gelu {
  symbols: N;
  tensors: X[N]: f32; Y[N]: f32;

  tile_size = 512

  @llm_hint("parallel")
  for i in range(0, ceil(N, tile_size)):
    x_tile = alloc([tile_size], llm_hint="l1_buffer")
    y_tile = alloc([tile_size], llm_hint="l1_buffer")

    load(X[i:i+tile_size] -> x_tile)
    gelu(x_tile, y_tile)              # Jean.coderDecision on how to achieve
    store(y_tile -> Y[i:i+tile_size])
}
```

## best practice

### order of preparation
1. **Base structure first**: symbols, tensors, main cycle framework
2. **Additional memory management**: aloc suitable tile
3. **And then add the data stream**: load - > data - > store
4. **Finally optimize hint**: @llm_hint decorator

### Tile Size Settings
- Consider hardware memory constraints (e.g. NPU UB size, GPU shared memory limits)
- Priority 2 bets (128, 256, 512, 1024)
- Ensure data alignment requirements

### Error avoidance
- **Do not mix abstract levels**: use either advanced functions (gelu) or base operations (add+mul)
- **Clear data flow**: each load has a corresponding compute and each compute has a corresponding store
- **Rational use of hint**: do not over-optimize, make sure the logic is correct
