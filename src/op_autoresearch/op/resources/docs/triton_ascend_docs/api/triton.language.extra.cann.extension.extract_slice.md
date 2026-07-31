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
