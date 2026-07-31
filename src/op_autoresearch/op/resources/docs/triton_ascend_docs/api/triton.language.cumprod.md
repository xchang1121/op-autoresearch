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

