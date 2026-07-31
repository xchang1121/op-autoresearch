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

