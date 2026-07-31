---
name: triton-cuda-api
description: "Triton CUDA API complete reference manual, including tl.load/store, tl.reduce, tl.dot, tl.atomic, etc. Signature, parameters and examples of use of core functions. This applies to any Triton CUDA nuclear code scenario that requires access to specific API usages to understand the meaning of function parameters"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# Triton CUDA API Reference Manual

This document provides detailed references to the Triton Core API, including functional signatures, parameter descriptions and examples of use.

## 1. kernel decorator

### @triton.jit
```python
@triton.jit
def kernel_function(...):
    pass
```
- **Fact**: Compile the Python function into the GPU kernel
- **Constraint**: `return`, `break`, `continue` statements cannot be used inside the function

## 2. Program ID and Grid API

### tl.program_id(axis)
```python
pid = tl.program_id(axis)  # axis: 0, 1, or 2
```
- **Parameters**: `axis` - dimensional axis (0, 1, 2)
- **Return**: Current program ID on this axis
- **Use**: Determine the data range of the current block

### tl.num_programs(axis)
```python
num_pids = tl.num_programs(axis)  # axis: 0, 1, or 2
```
- **Parameters**: `axis` - dimensional axis (0, 1, 2)
- **Return**: total number of programs on this axis
- **Use**: Calculate grid size and boundary conditions

### triton.cdiv(a, b)
```python
grid_size = triton.cdiv(total_elements, block_size)
```
- **Parameters**: `a`, `b` - divided and divided
- **Return**: remove the division result upwards
- **Use**: host side usage, calculation of startup grid size

## 3. Memory Operation API

### tl.load(pointer, mask=None, other=None, boundary_check=None)
```python
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```
- **Parameters**:
  - `pointer`: Memory pointer
  - `mask`: Boolean mask, True indicates a valid position
  - `other`: Default value when mask is False
  - `boundary_check`: Border check dimensions (0, 1) or None
- **Return**: tensor data loaded
- **Use**: Load data from global memory

### tl.store(pointer, value, mask=None, boundary_check=None)
```python
tl.store(ptr + offsets, result, mask=mask)
```
- **Parameters**:
  - `pointer`: Memory pointer
  - `value`: Value to store
  - `mask`: Boolean mask, True indicates a valid position
  - `boundary_check`: Border check dimensions (0, 1) or None
- **Use**: Store data to global memory

### tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)
```python
block_ptr = tl.make_block_ptr(
    base=ptr,                    # Base pointer
    shape=(M, N),                # Full Matrixshape
    strides=(stride_m, stride_n), # Step length
    offsets=(start_m, start_n),   # Current block offset
    block_shape=(BLOCK_M, BLOCK_N), # Blocksshape
    order=(1, 0)                 # Memory Layout Order
)
```
- **Parameters**:
  - `base`: Base memory pointer
  - `shape`: shape full of tensor
  - `strides`: The length of each dimension
  - `offsets`: Initial offset of current block
  - `block_shape`: Size of current block
  - `order`: Memory layout order (1,0) for line main order
- **Return**: block refers to the elephant
- **Use**: Efficient access to 2D data blocks

### tl.advance(ptr, offsets)
```python
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
```
- **Parameters**:
  - `ptr`: Block pointer
  - `offsets`: Offset of dimensions
- **Return**: block pointer after moving
- **Pilot**: Move Block Pointer to Next Location

## 4. tensor Create & Operation API

### tl.arange(start, end)
```python
offsets = tl.arange(0, BLOCK_SIZE)
```
- **Parameters**: `start`, `end` - Start and end values
- **Return**: Continuous integer series
- **Use**: Create Index Sequence

### tl.zeros(shape, dtype)
```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```
- **Parameters**:
  - `shape`: tensorZ1XQ
  - `dtype`: data type
- **Return**: Zero tensor

### tl.full(shape, value, dtype)
```python
ones = tl.full((M, N), 1.0, dtype=tl.float32)
```
- **Parameters**:
  - `shape`: tensorZ1XQ
  - `value`: Filling value
  - `dtype`: data type
- **Return**: fill tensor with specified values

### tl.cast(input, dtype)
```python
float_data = tl.cast(int_data, tl.float32)
```
- **Parameters**:
  - `input`: Enter tensor
  - `dtype`: Target data type
- **Return**: tensor after type conversion

## 5. Mathematics Operations API

### tl.dot(a, b, acc=None, allow_tf32=True)
```python
result = tl.dot(a, b, acc=accumulator)
```
- **Parameters**:
  - `a`, `b`: Enter Matrix
  - `acc`: Composer (optional)
  - `allow_tf32`: Allow TF32 accuracy (CUDA-specific, Ampere+GPU)
- **Return**: matrix multiplication result
- **Pilot**: Core matrix multiplication operation, accelerated using Tensor Core

### tl.sum(x, axis)
```python
block_sum = tl.sum(data, axis=0)
```
- **Parameters**:
  - `x`: Enter tensor
  - `axis`: Axis of Return
- **Return**: Return result

### tl.max(x, axis)
```python
max_val = tl.max(data, axis=0)
```
- **Parameters**:
  - `x`: Enter tensor
  - `axis`: Axis of Return
- **Return**: max

### tl.min(x, axis)
```python
min_val = tl.min(data, axis=0)
```
- **Parameters**:
  - `x`: Enter tensor
  - `axis`: Axis of Return
- **Return**: Minimal value

### tl.where(condition, x, y)
```python
result = tl.where(mask, data, 0.0)
```
- **Parameters**:
  - `condition`: Condition tensor
  - `x`, `y`: Select Value
- **returned**: value selected according to condition
- **Practice**: SIMD friendly choice of conditions

### tl.exp(x) / tl.log(x) / tl.sqrt(x)
```python
exp_val = tl.exp(x)
log_val = tl.log(x)
sqrt_val = tl.sqrt(x)
```
- **Practice**: Basic Math Functions, CUDA backend directly supported

### tl.sigmoid(x)
```python
sigmoid_val = tl.sigmoid(x)
```
- **Pilot**: Sigmoid Activation Function

### tl.extra.cuda.libdevice.tanh(x)
```python
tanh_val = tl.extra.cuda.libdevice.tanh(x)
```
- **Practice**: hyperbolic tangent function
- **Note: CUDA backend does not have `tl.tanh` or `tl.math.tanh`, must use `tl.extra.cuda.libdevice.tanh`

### tl.math.exp2(x) / tl.math.log2(x)
```python
exp2_val = tl.math.exp2(x)
log2_val = tl.math.log2(x)
```
- **Use**: Index/ logarithmic at 2

### tl.cumsum(input, axis=0, reverse=False, dtype=None)
```python
cumulative_sum = tl.cumsum(data, axis=0)
reverse_cumsum = tl.cumsum(data, axis=1, reverse=True)
```
- **Parameters**:
  - `input`: Enter tensor
  - `axis`: Axis of Accumulation Sum (default 0)
  - `reverse`: Whether to accumulate backwards (default is False)
  - `dtype`: Output data type (optional, default is the same as input)
- **Return**: cumulative sum result tensor
- **Use**: Calculate accumulation along assigned axes, often for prefixing and calculation

### tl.cumprod(input, axis=0, reverse=False)
```python
cumulative_prod = tl.cumprod(data, axis=0)
```
- **Parameters**:
  - `input`: Enter tensor
  - `axis`: Axis of cumulative product (default 0)
  - `reverse`: Whether to accumulate backwards (default is False)
- **Return**: cumulative product tensor

## 6. Atomic Operation API

### tl.atomic_add(pointer, value)
```python
tl.atomic_add(output_ptr, block_sum)
```
- **Parameters**:
  - `pointer`: Target memory pointer
  - `value`: Value to add
- **Pilot**: Line secure plus operation

### tl.atomic_max(pointer, value)
```python
tl.atomic_max(max_ptr, local_max)
```
- **Parameters**:
  - `pointer`: Target memory pointer
  - `value`: Value to compare
- **Pilot**: maximum value update for thread security

### tl.atomic_min(pointer, value)
```python
tl.atomic_min(min_ptr, local_min)
```
- **Parameters**:
  - `pointer`: Target memory pointer
  - `value`: Value to compare
- **Pilot**: minimum value update for thread security

### tl.atomic_cas(pointer, cmp, val)
```python
old = tl.atomic_cas(ptr, expected, desired)
```
- **Parameters**:
  - `pointer`: Target memory pointer
  - `cmp`: Expectations
  - `val`: New value
- **Return**: Original
- **Use**: Compare and exchange operations

### tl.constexpr
```python
BLOCK_SIZE: tl.constexpr = 1024
```
- **Practice**: Mark constant parameters at time of compilation
- **Constraint**: must be stated in a function signature

## 7. Block Distribution Optimization API

### tl.swizzle2d(i, j, size_i, size_j, group_size)
```python
task_i, task_j = tl.swizzle2d(block_i, block_j, NUM_BLOCKS_I, NUM_BLOCKS_J, GROUP_SIZE)
```
- **Parameters**:
  - `i`, `j`: Raw block index
  - `size_i`, `size_j`: Total number of blocks
  - `group_size`: Group Size (usually 2/4/8)
- **Return**: Renumbered block index (task_i, task_j)
- **Pilot**: 2D block reset to increase L2 cache locality
- **Applicable scene**: matrix multiplication multi-dimensional block calculations to improve data reuse

## Use recommendations

1. **Memory operation**: Prefer to `tl.make_block_ptr` for processing 2D data
2. **Boundary check**: always use `mask` or `boundary_check` to prevent border crossing
3. **Atomic operations**: use only when necessary, performance costs
4. **data type**: Note type conversion, use `tl.cast` visible conversion
5. **Compiler optimization**: use `tl.constexpr` tag constant parameters
6. **Tensor Core**: The MatMul type operation uses `allow_tf32=True` with Tensor Core
