### tl.where(condition, x, y)
```python
result = tl.where(mask, data, 0.0)
```
- **Parameters**:
  - `condition`: Condition tensor
  - `x`, `y`: Select Value
- **returned**: value selected according to condition
- **Practice**: SIMD friendly choice of conditions

