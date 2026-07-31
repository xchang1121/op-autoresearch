---
name: kernel-workflow
description: AI Kernel operator generates and optimizes workflows. Use this Skill when users need to generate, validate or optimize kernel operator. Supports a variety of backend DSL such as Triton, CUDA C, C++, TileLang.
---

# Kernel operator generation and optimization of workflow

## Summary of process

1. **Analyse user input**→ only request description or have task code?
2. **Certify/ Generate the task code**→ for details of `references/torch_task_verify.md`
3. **Choose how to generate/optimize**→ for more details on `references/tool-selection.md`
4. **Execute generation/optimization**→ Call sub-Agent (transmit `user_requirements` if additional user needs are required)
5. **Return result**→ with `finish`

**Note: `user_requirements` is optional.**

## Enter type judgement

1. **Only needs description**(no code) → Direct `call_op_task_builder`
2. **task Code**(includes `class Model(nn.Module)`) → Load `references/torch_task_verify.md` for validation
3. **Kernel code needs to be optimized**(including `@triton.jit` / `__global__`, etc.) → Validate the task code and load `references/tool-selection.md`

## Examples of basic processes

### Example 1: User only needs description

```
User: Generate one ReLU operator

Think: The user did not provide the code and generated it directly task.
Action: call_op_task_builder(user_request="Generate one ReLU operator")
Observation: {status: "READY", task_code: "...", op_name: "relu"}

Think: task Generated, Loaded tool-selection Directs the selection of the method of generation.
Action: read_file(file_path="skills/kernel-workflow/references/tool-selection.md")
...
```

### Example 2: User provides Torch task code

```
User: Help me generate this code. kernel: class Model(nn.Module)...

Think: The user provided the code and loaded the validation guide.
Action: read_file(file_path="skills/kernel-workflow/references/torch_task_verify.md")
Observation: <Validation guide>

Think: Validate the code format in accordance with the guidance.
Action: execute_script(script_path="skills/kernel-workflow/scripts/check_torch_code.py", args="--stdin --json", stdin_input="<Code>")
...
```

### Example 3: User provides the Kernel code for optimization

```
User: Help me optimize this. Triton kernel: @triton.jit def kernel(...): ...
      Corresponding task Yes.: class Model(nn.Module)...

Think: User provided kernel and taskCheck first. task.
Action: read_file(file_path="skills/kernel-workflow/references/torch_task_verify.md")
...After validation...
Action: read_file(file_path="skills/kernel-workflow/references/tool-selection.md")
...
```

## Reference Documents

1. `references/torch_task_verify.md` - task code validation and completion process
2. `references/tool-selection.md` - SubAgent Select and Reconfirm Process

## Scripts

1. `scripts/check_torch_code.py` - Authenticate tag code format (parameters: `--stdin --json`)

## Prohibition of conduct

1. Could not close temporary folder: %s
2. Autoselect when the generation method is not specified by the user
3. Ask generated before calling `call_op_task_builder`
4. No double confirmation before calling sub-Agent
