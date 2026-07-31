### tl.insert_slice(ful, sub, offsets, sizes, strides)
```python
output = tl.insert_slice(output, output_sub, [offset], [size], [1])
```
- **Activation**: Insert sub-tensor at the specified position of target tensor by offset, size and step.
- **Parameters**:
  - `ful`: Target for receiving the insertion result tensor
  - `sub`: Subs tensor to insert
  - `offsets`: Initial offset on all dimensions of the insert area
  - `sizes`: Size of the insert area on all dimensions
  - `strides`: The length of the insertion area on all dimensions
- **Return**: new tensor after insertion of sub-tensor
