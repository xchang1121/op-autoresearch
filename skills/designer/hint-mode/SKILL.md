---
name: hint-mode
description: "Space Configuration Guide for Hint Mode Parameters in Sketch Design to extract range of parameters from the mission description and generate a modifiable parameter space configuration"
category: guide
version: "1.0.0"
metadata:
  role: designer
---

# Hint Model: Parameter Space Configuration Guide

## Overview

The Hint mode is used to identify parameters range constraints from mission descriptions, generate parameter spatial configurations (space_config), and support subsequent automatic emulations.

## Hint Syntax:

### Standard format
```python
# @hint: param in [val1, val2, ...]         → type='choice', values=[...]
# @hint: param in range(min, max, step=N)   → type='range', min=..., max=..., step=...
# @hint: param = value                      → type='fixed', value=...
# @hint: param in pow2(min_pow, max_pow)    → type='power_of_2', min_pow=..., max_pow=...
# @hint: param in pow of 2                  → type='power_of_2'
```

### Compatible Format
```python
# @range_hint("param", start=min, end=max)  → type='range', min=..., max=...
# @elemwise_hint("param", [val1, val2])     → type='choice', values=[...]
# @elem_hint("param", [val1, val2])         → type='choice', values=[...]
```

### Example:
```python
# @hint: batch_size in pow of 2             → {'type': 'power_of_2', 'min_pow': 3, 'max_pow': 6}  # [8, 16, 32, 64]
# @hint: dim in range(16, 65536)            → {'type': 'range', 'min': 16, 'max': 65536, 'step': 1}
# @hint: BLOCK_M in [64, 128, 256]          → {'type': 'choice', 'values': [64, 128, 256]}
```

## Hint extraction rules

1. **All hint**: including annotated hint (`# @range_hint`) should be identified and extracted
2. **Complete extraction**: if there are more than one hint, all must be extracted and cannot be omitted
3. **Format conversion**:
   - Decorator format (e. g. `@range_hint("param", st=8, ed=64)`) → to extract information in brackets
   - Comment format (e. g. `# @hint: param in range(16, 65536)`) → extract colon declaration
   - Harmonize to standard `SPACE_CONFIG` Dictionary format

## Parameter Space Configuration Template

```python
"""Parameter Space Configuration"""
import torch  # or import mindspore as ms

# Synchronization point.
SPACE_CONFIG = {
    'param1': {'type': 'choice', 'values': [val1, val2, ...]},
    'param2': {'type': 'range', 'min': min_val, 'max': max_val, 'step': step_val},
    'param3': {'type': 'power_of_2', 'min_pow': min_exp, 'max_pow': max_exp},
    # ...all parameters extracted by hint
}

# Synchronization point.
META_INFO = {
    'op_name': 'op_name',
    'framework': 'torch',  # or 'mindspore'
    'param_names': ['param1', 'param2', ...]  # List of parameter names, keep order!
}

# Synchronization point.
def create_inputs(param1, param2, ...):
    """
    Generate input based on parameters
    Arguments must be in order META_INFO['param_names'] Unanimously
    """
    ...
    return [tensor1, tensor2, ...]

# Synchronization point.
def get_init_inputs():
    """If this is the original code, copy it in its entirety"""
    return []  # or ["auto"]
```

## BLONK_SIZE Configuration Selection Principles

### Smaller range of parameters (e.g. M in [128, 256])
- Avoid too big BLONK_SIZE, recommended [32, 64, 128]
- Make sure the minimum can accommodate at least one BLONK

### Larger range of parameters (e.g. M inrange (128, 8192))
- Providing multiple options [64, 128, 256, 512]
- Use autotune to fit different sizes

## Border management strategy

### Option A: Use mask (recommended)
- Support for Any Shape
- In the sketch note, state: `Use mask in support of any shape '

### Option B: Do not use mask (more performance)
- Request Shape COMPLETE BLONK_SIZE
- Description in sketch notes: `M %64 = 0 (no mask) '

## Note on the scope of application of the design

A "design scope" note must be added to the sketch:

```python
# Design scope of application:
# - Parameter range: M in [128, 2048], N in [128, 4096]
# - Border processing: use mask / do not use mask (to be divided)
# - BLONK_SIZE Configuration: [64, 128, 256]
# - Conditional: if no mask, M %64 = 0
```

Important**:
- Whether sketch uses a few dimensions, the "design scope" comment must show a separate range for each dimension originally entered
- Do not write only the range of the total number of elements, write separately the range of each dimension
- If it's two, you need to mark it as well.
