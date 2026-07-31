### @triton.jit
```python
@triton.jit
def kernel_function(...):
    pass
```
- **Fact**: Compile the Python function into the hardware kernel
- **Constraint**: `return`, `break`, `continue` statements cannot be used inside the function

