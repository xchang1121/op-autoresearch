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

