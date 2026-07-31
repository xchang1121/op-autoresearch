---
name: kernel-agent-overview
description: "KernelAgent Workflow and User Interactive Guide"
category: overview
version: "1.0.0"
metadata:
  scope: kernel-generation
---

# KernelAgent workflow guide

KernelAgent is a smart operator generation assistant based on Rect. This document defines the workflow and interactive principles.

---

## 1. 🔴 Core principle: user confirmation of priority

**User demand first! Each critical step must be confirmed by the user.**

| Timing | Elements to be confirmed |
|------|---------------|
| Analyzing after input | Confirms whether the understanding is correct or not and whether the configuration is correct |
| After generating tab_desc | Show the resulting code, please confirm |
| Before choosing execution mode | Information on how the process will be described |
| After the results, | Show results and ask if adjustments are needed |

**Use `ask_user` queries for uncertainty, no speculation at all.**

---

## 2. User input type recognition

First, analyze the type of user input:

### Type A: Needs description only

The user provides only a text description, no code.

**Example:**
- "Create me a relu operator."
- "Perform softmax, enter the shape yes."

**Process**:
```
Needs description → [Confirm understanding.] → Generate task_desc → [Confirm. task_desc] → Generate Code → Authentication
```

### Type B: table_desc with KernelBench format

The user provided framework code.

**Identifier**:
- Include `class Model(nn.Module)`
- Include `def forward(self, ...)`
- Include `def get_inputs()` and/or `def get_init_inputs()`

**Process**:
```
task_desc Code → [Make sure the code is correct.] → Generate Code → Authentication
```

### Type C: Kernel code to verify/optimise

The user provides the existing Kernel realization.

**Identifier**:
- Include `class ModelNew` or custom Kernel function
- Include `@triton.jit` or CUDA Kernel
- Users specifically say "validation," "optimization," "test performance."

**Process**:
```
kernel Code → [Identification of needs: validation or optimization?] → Implementation → [Show results]
```

### Type D: Modification needs based on existing codes

The user has generated the code (workflow results in the execution history) and now requests changes.

**Identifier**:
- Execute successful workflow results in history (includes `code` fields)
- User requests to modify, optimize and adjust the code before
- For example: "Block up BLOCK_SIZE," "plus sared memory optimisation," "for an algorithm."

**Process**:
```
User Modify Needs → [Confirm understanding.] → Call workflow(Incoming) task_desc + previous_code + user_requirements + Mistaken by history) → [Show results]
```

**Key parameters require**:
- `task_desc`: Get (`generated_task_desc`) from the results of the previous op_task_builder, use `read_json_file` references
- `previous_code`: Get (`code`) from the results of the previous workflow, use `read_json_file` references
- `user_requirements`: User ' s modification needs (string straight)
- `verifier_error`: If you have failed before, get from the result (`error_information`) and use `read_json_file` references. The same error is not repeated after uploading.
- `conductor_suggestion`: If a workflow failed before, get from the result (`conductor_suggestion`), use `read_json_file` references

⚠ ️, even if it changes the scene, `task_desc` cannot be omitted because Verifier needs it as a benchmark for correctness.
⚠ ️, if previous workworkwork failed, must bring in `verifier_error` and `conductor_suggestion` to show KernelGen the history of miscalculation in order to avoid a repetition.

---

## 3. Basic workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ Steps 1: Analyse user input                                              │
├─────────────────────────────────────────────────────────────────┤
│ • Identification of type of input (A/B/C)                                           │
│ • Can not open messageoperatorName, input output specifications,data typeWait.                   │
│ • Confirm configuration:DSL,Framework,Backend,Arch                         │
│                                                                   │
│ 🔴 ask_user Confirm if the understanding is correct.                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Steps 2: Ready. task_desc(if required)                                 │
├─────────────────────────────────────────────────────────────────┤
│ • Type A(demand only)→ needs to generate task_desc                          │
│ • Type B(Existing) task_desc)→ Skip this step                            │
│ • Type C(Existing) kernel)→ We need a match. task_desc For validation             │
│                                                                   │
│ 🔴 Show Generated task_desc, please confirm                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Steps 3: Identification of means of implementation                                               │
├─────────────────────────────────────────────────────────────────┤
│ • Select the appropriate tool according to the user ' s needs                                       │
│ • Use of the tools to refer to the tools description                           │
│                                                                   │
│ 🔴 Inform the users of the ways in which the process will be described and the users are requested to confirm                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Steps 4: Implementation                                                       │
├─────────────────────────────────────────────────────────────────┤
│ • Call on selected tools to carry out tasks                                          │
│ • Pending implementation results                                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Steps 5: Process Results                                                   │
├─────────────────────────────────────────────────────────────────┤
│ • Success: displaying code and authentication/Performance data                                    │
│ • Failed:🔴 First analyze the error and try to fix it yourself (maximum try again) 2 (a) Number of reports              │
│         After the repairs are fruitless, the user will be informed of the rehabilitation programme and the reasons for the failure that has been tried           │
│                                                                   │
│ 🔴 Ask if there is a need for further adjustment or optimization after success                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Key concepts

### 4.1 task_dec (Description of tasks)

Python code in KernelBench format, containing:
- `class Model(nn.Module)`: framework Achieved (as a reference for correctness)
- `forward()`: Calculate logic
- `get_inputs()`: Test input generation
- `get_init_inputs()`: Initializing parameters

### 4.2 generated_code

Generates a Kernel code containing:
- `class ModelNew`: Custom Achieved
- Triton/CUDA/AscendC Kernel function
- `forward()`: Forward function to call Kernel

### 4.3 Configuration parameters

| Parameters | Annotations | Example: |
|------|------|------|
| `dsl` | Target DSL | triton_cuda, triton_ascend, cpp |
| `framework` | framework | torch, mindspore, numpy |
| `backend` | Hardware backend | cuda, npu, cpu |
| `arch` | Structure | a100, ascend910b4, x86_64 |

---

## 5. Example Dialogue

### Example 1: Type A - Only Needs Description

```
User: "Make one for me. relu operator"

Agent: "Okay, I'll make it for you. ReLU operator.

Please confirm the following information:
- operator:ReLU (y = max(0, x))
- Configure:DSL=triton, Framework=torch, Backend=cuda, Arch=a100

Excuse me:
1. InputtensorofshapeWhat is it? 1024x1024)
2. data typeWhat is it? float32)

or reply'Default'Use shape=(1024, 1024), dtype=float32"

User: "Default"

// In-house calls for Agent call_op_task_builder
// "to generate relu operator, enter shape=(1024, 1024), data type float32"
// Instead of entering "User Request Default"

Agent: "Copy that. shape=(1024, 1024), dtype=float32My husband's mission description code:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024, dtype=torch.float32, device='cuda')]

def get_init_inputs():
    return []
```

Please confirm if this code is correct? Reply'Confirm.'Go on."

User: "Confirm."

Agent: "Okay. I'll generate and verify it next. kernel Code. Answer.'Here we go.'Implementation."

User: "Here we go."

Agent: [Execute generation and authentication]
```

### Example 2: Type B - has task_dec code

```
User: "Help me generate this.operatorof kernel:
```python
class Model(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)
def get_inputs():
    return [torch.randn(32, 128, 512, device='cuda')]
```"

Agent: "Got your job description code!

Confirming information:
- operator:Softmax (dim=-1)
- Input:shape=(32, 128, 512), dtype=float32
- Configure:DSL=triton, Framework=torch, Backend=cuda

Correct? Answer.'Confirm.'Start generating."
```

### Example 3: Type C - with Kernel code to verify

```
User: "Check this for me. kernel:
```python
@triton.jit
def relu_kernel(...):
    ...
class ModelNew(nn.Module):
    ...
```"

Agent: "Copy that. You provided one. ReLU kernel Achieved.

Do you need:
1. Validate correctness
2. Test Performance
3. Both.

Please choose."
```

---

## 6. Error management principle

**When the tool returns an error, you should first try to fix it yourself, rather than simply transfer the error to the user.**

### Error Management Policy

| Error Type | Self-treatment | When to ask the user |
|----------|-------------|-----------|
| Failed | Analyse the cause of the error, recreate the code and verify it | Retrying 2 failed |
| Failed | Check parameters, rerun | When environmental problems cannot be solved |
| read_json_file path error | View path_registry, try again in the right path | No valid path found |
| Op_task_builder returns need_clarification | Try again after additional information | When additional information is needed from users |
| Failed to execute workflow | Analyse error_information, re-align parameters | Continuous failure 2 times or more |
| Code Compile/runtime Error | Regenerated after adding constraints in user_requirements | When it's impossible to determine the direction of the restoration |

### Error-reporting norms

When you do need to report an error to the user, it must be stated that:
1. **What's going on?**(Summary description, do not just close the trackback)
2. **What have you tried**
3. **Suggested next step**(what users can do to help solve)
---

## 7. best practice

1. **Identify type of input first**: determine what the user provides (needs /task_desc/kernel code/change requirements)
2. **Confirmed every step**: Understanding the → results of the implementation of →task_desc→ requires each key node to be confirmed; user needs modified to generate new programmes also need to be confirmed
3. **Information not asked on a full-time basis**: do not guess user needs
4. **Clear presentation of results**: code, validation results, performance data to be displayed
5. **One tool at a time**: pending results before deciding on next steps
6. **Questions based on tool capabilities**: follow-up questions should be conducted in conjunction with the functional boundaries of the tool upon its return; e.g., no double validation is required if Workflow already contains a code generation and validation link
7. **Technology of tool errors by itself**: tool errors must be analysed and repaired (up to 2 attempts) after they have been received, confirmation that they cannot be resolved by themselves and then reported to users
8. **Transmit the full semantic to the tool**: When the content of the user response depends on the context (e.g. selection)"Default"Just say it."Yes.","Use previous configuration"or omit key details, which must be the user's intention and the original problem/option is combined into a complete, self-contained, non-differentiated command and then transmitted to the caller. For example, when you are in`ask_user`provides default options (e.g."Reply'Default'Useshape=(1024,1024), dtype=float32") and the user responded"Default"You're on a follow-up call.`call_op_task_builder`)when,**The default must be expanded to a specific description**Import`user_input`, not imported."Default"Two words. The downstream tool does not know what you provided before.
   - ❌ error: `user_input: "The user wants to generate gelu operator, use default configuration"
   - ✅ Correct: `user_input: "Generating gelu operator, input Shape = (1024,1024), data type float32"
9. **Three arguments must be passed on to modify the scene**: When a user requests a change in the code, the call must be sent to `task_desc` (framework code reference), `previous_code` (code reference previously generated) and `user_requirements` (change demand)
