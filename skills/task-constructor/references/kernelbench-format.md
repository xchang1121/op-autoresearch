# KernelBench Job Format

## File Structure

KernelBench task file is**a single Python file**containing the following four essential parts:

### 1. Imports Area

```python
import torch
import torch.nn as nn
# Only standard library and PyTorch-related packages are allowed
# Prohibit other documents in the Import project
```

### 2. Model Class

```python
class Model(nn.Module):
    def __init__(self, <init_params>):
        super(Model, self).__init__()
        # Save all initializing parameters

    def forward(self, <forward_inputs>) -> torch.Tensor:
        # Core calculation logic
        return output
```

### 3. `get_inputs()` function

```python
def get_inputs():
    """Back forward() ofinput parameterList"""
    return [torch.randn(batch_size, dim)]
```

### 4. `get_init_inputs()` function

```python
def get_init_inputs():
    """Back __init__() List of initialised parameters"""
    return [dim_value]
```

## Key constraints

| Constraints | Annotations |
|------|------|
| Self-include | All relying functions must be inlined into the file |
| Executable | `Model(*get_init_inputs()).forward(*get_inputs())` must run directly |
| Determination | Gives the same input, the output must be consistent |
| None NAN/Inf | Forward output cannot contain NN or Inf |
| Reasonable input | Get_inputs should provide input of reasonable size (not too small or too large) |
| Unanimously return | Return type/shape must be consistent with original |
