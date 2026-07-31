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

