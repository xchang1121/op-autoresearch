---
name: task-constructor
description: >
  The extraction of operator from the PyTorch/Triton code warehouse was achieved, and the standardized single file in KernelBench format was built to contain tasks.
  Supports code extraction, AST-dependent tracking, intra-functional connection, import clean-up, format validation and reference comparison testing.
  Use this Skill when the user needs to build a name_code from an existing code.
category: workflow
version: "1.0.0"
metadata:
  task_type: code_transformation
  target_format: kernelbench
  input_types: code,file,directory
---

# Standardised Task Build Workstream

## Apply scene

- User provides PyTorch/Triton code warehouse path, needs to extract operator and build task_code
- User provides code clips that require standardized tasks in KernelBench format
- The user specifies an internal torch function that needs to be extracted and decomposed

## Objective Format

The resulting file must be**single-inclusion Python file**:

```python
import torch
import torch.nn as nn

# All relying functions inline (not able external file)

class Model(nn.Module):
    def __init__(self, <params>):
        super(Model, self).__init__()

    def forward(self, <inputs>) -> torch.Tensor:
        return output

def get_inputs():
    return [input1, input2, ...]

def get_init_inputs():
    return [param1, ...]
```

## Guide for the use of tools

### Call `call_task_constructor`

This tool runs a full rect loop internally and automatically completes the following steps:

1. **Target code**: search target function
2. **Dependence on tracking**: AST analysis automatically discovers all dependencies (with file function + call from external module)
3. **Task configuration**: Select the best policy (exclusion/selective/complete embedding) to build a self-inclusion file
4. **Validation**: Format Validation (Practicalisation +Forward + NN/ Inf + Consistency)
5. **Reference**: Multigroup input compared to the original torch function

### Parameters

- `user_input`: Description of user needs (e. g. "Decomposition of xx extracted from pytoch warehouse")
- `source_path`: Optional, code warehouse/file path

### Back

- `task_code`: Standardized task code generated
- `task_code_path`: Path to code file
- `op_name`: operator name
- `summary`: Summary of the construction process

## Core rules

1. **Complicated functions are prohibited**: original functions can be run and reused directly, without change in any line
2. **return value must be consistent**: additional tensor returns back to tuple, uninterrupted
3. **Inline external functions are preceded by signature**: external call source module automatically detected by relying on tracking

## Scripts

1. `scripts/validate_kernelbench_task.py` - Verify whether the tag code matches KernelBench format (parameters: `--stdin --json`)

### Use Example

```
Think: Requires validation of generated task Is the code correct?
Action: execute_script(script_path="skills/task-constructor/scripts/validate_kernelbench_task.py", args="--stdin --json", stdin_input="<task Code>")
Observation: {"valid": true, "static_check": {...}, "runtime_check": {...}}
```

You can also verify the file directly:

```
Action: execute_script(script_path="skills/task-constructor/scripts/validate_kernelbench_task.py", args="/path/to/task.py --json")
```

## Reference Documents

- `references/kernelbench-format.md` - Format Specification
- `references/assembly-strategies.md` - assembly policy statement
