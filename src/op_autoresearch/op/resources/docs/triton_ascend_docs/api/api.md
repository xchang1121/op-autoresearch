# Triton API Reference Manual

This document provides detailed references to the Triton Core API, including functional signatures, parameter descriptions and examples of use.

## 1. kernel decorator

### @triton.jit
```python
@triton.jit
def kernel_function(...):
    pass
```
- **Fact**: Compile the Python function into the hardware kernel
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
- **Use**: host side to calculate start-up grid size

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

### tl.cdiv(a, b)
```python
result = tl.cdiv(offset, BLOCK_SIZE)
```
- **Parameters**: `a`, `b` - divided and divided
- **Return**: Take the division result up ⌈a/ b⌉
- **Practice**: internal use in Kernel, calculation of upward decomposition result, equivalent to `(a + b - 1) // b`

### tl.dot(a, b, acc=None, allow_tf32=True)
```python
result = tl.dot(a, b, acc=accumulator)
```
- **Parameters**:
  - `a`, `b`: Enter Matrix
  - `acc`: Composer (optional)
  - `allow_tf32`: Allow TF32 accuracy
- **Return**: matrix multiplication result
- **Practice**: Core matrix multiplication Operations

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


### tl.where(condition, x, y)
```python
result = tl.where(mask, data, 0.0)
```
- **Parameters**:
  - `condition`: Condition tensor
  - `x`, `y`: Select Value
- **returned**: value selected according to condition
- **Practice**: SIMD friendly choice of conditions

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
reverse_cumprod = tl.cumprod(data, axis=1, reverse=True)
```
- **Parameters**:
  - `input`: Enter tensor
  - `axis`: Axis of cumulative product (default 0)
  - `reverse`: Whether to accumulate backwards (default is False)
- **Return**: cumulative product tensor
- **Use**: Calculating cumulative multipliers along specified axes, often used in probability calculations and sequence processing

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
- **Use**: 2D block reset to increase cache locality
- **Applicable scene**: matrix multiplication multi-dimensional block calculations to improve data reuse
- **Note: Priority (i) grouping is supported only, and priority is manually achieved

## 8. Slice Extension API

### tl.extra.cann.extension.extract_slice(ful, offsets, sizes, strides)
```python
sub_tensor = tl.extra.cann.extension.extract_slice(tensor, [0], [32], [1])
```
- **Activation**: Extracts a slice from the input tensor by offset, size and step.
- **Parameters**:
  - `ful`: Source tensor to extract slices
  - `offsets`: Initial offset of slices on all dimensions
  - `sizes`: Size of slices on all dimensions
  - `strides`: Length of slices in all dimensions
- **Return**: after extract tensor
### tl.extra.cann.extension.insert_slice(ful, sub, offsets, sizes, strides)
```python
output = tl.extra.cann.extension.insert_slice(output, output_sub, [offset], [size], [1])
```
- **Activation**: Insert sub-tensor at the specified position of target tensor by offset, size and step.
- **Parameters**:
  - `ful`: Target for receiving the insertion result tensor
  - `sub`: Subs tensor to insert
  - `offsets`: Initial offset on all dimensions of the insert area
  - `sizes`: Size of the insert area on all dimensions
  - `strides`: The length of the insertion area on all dimensions
- **Return**: new tensor after insertion of sub-tensor

## API does not exist in the current version

### tl.extract_slice(ful, offsets, sizes, strides)
This api doesn't exist in the current version.

### tl.insert_slice(ful, sub, offsets, sizes, strides)
This api doesn't exist in the current version.

