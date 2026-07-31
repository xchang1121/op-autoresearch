# Task code format validation guide

## Authentication Method

Perform validation scripts using `execute_script` (both static and runtime):

```python
execute_script(
    script_path="skills/kernel-workflow/scripts/check_torch_code.py",
    args="--stdin --json",
    stdin_input="<task Code>"
)
```

## KernelBench Format Requirements

The task code must contain four essential components:

```python
import torch
import torch.nn as nn

class Model(nn.Module):         # 1. It must be inherited. nn.Module
    def __init__(self):
        super().__init__()

    def forward(self, x):       # 2. Model Category method
        return torch.relu(x)

def get_inputs():               # 3. Top level function, return list
    return [torch.randn(16, 1024)]

def get_init_inputs():          # 4. Top level function, return list
    return []
```

## Use scene

1. **Requirement description only**→ without authentication, direct `call_op_task_builder`
2. **Totch task code**→ validates this code and loads `tool-selection.md` after adoption
3. **Kernel code + Task code**→ validates tag code, post-load `tool-selection.md`

## Validation Results Processing

1. `"valid": true` → Validation pass, load `tool-selection.md` selection method
2. `"valid": false` → Call `call_op_task_builder` to complete or repair:

```python
call_op_task_builder(user_request="Complete the following code as KernelBench Format:\n<Original Code>")

call_op_task_builder(user_request="Fix an error in the following code:\n<Original Code>\n\nerror message:<error>")
```

---

## Task related needs (important!)

### What is a Task demand?

User 's requests for changes to**task code**(rather than kernel) which are processed through `call_op_task_builder`,**do not transmit `user_requirements`**:
For example:
1. data type: "Input to float16", "Use bflota16"
2. Enter Shape: "Batch_size to 64" and "dim to 2048"
3. Code Completion: "Purpose _inputs", "Add to Initialization"

### Treatment of examples

```
User: "Change input to float16"
   → call_op_task_builder(user_request="Modify task Code, enterdata typewas replaced by float16:\n<Original task Code>")

User: "batch_size Replace with 128,dim Replace with 4096"
   → call_op_task_builder(user_request="Modify task Code,batch_size=128,dim=4096:\n<Original task Code>")
```
### Treatment of mixed needs

If the user has both a request for task and a request for Kernel, deal separately:

```
User: "Generate ReLU,In float16Quantified nucleus."
         ↓
   1. task Requirements "float16" → call_op_task_builder(user_request="Generate float16 of ReLU")
   2. User confirmed.taskafterward,kernel Requirements "kernel dichotomy" → call_coder_only(..., user_requirements="kernel dichotomy")
```
### Distinction from Kernel's needs

- **Task demand**: influence input data (types, shape, settings and modifications) → `call_op_task_builder`
- **Kernel demand**: influence achievement policy (severation, optimization, algorithm) → `user_requirements` parameters

## note

1. (`call_op_task_builder` internal assurance format correct)
2. Mixed text must first extract a pure code part
3. **Task-related needs do not enter `user_requirements`**and should be handled through `call_op_task_builder`
