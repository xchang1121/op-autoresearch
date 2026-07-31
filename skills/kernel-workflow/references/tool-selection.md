# SubAgent Selection Guide

## Select Logic

1. **User specified**→ recognition by keyword
2. **User not specified**→ with `ask_user` query
3. **Called the previous second confirmation**→**has to**again `ask_user` confirms user selection

## Keyword Recognition

1. "Evilve," "Evolutionary Search," "High Performance," → `call_evolve`.
2. "With coder_only," "fast generation," → `call_coder_only`.
3. "Adaptive_search", "Search the Tree" → `call_adaptive_search`
4. "test performance," "profile," → `call_coder_only(task_type="profile")`.

## Ask User

Use this helper when the generation mode is unclear:

```python
ask_user(message="Please select the generation method:\n1. coder_only - Quick Generate\n2. evolve - Evolution search (high performance)\n3. adaptive_search - Tree Search")
```

## Reconfirmation (required)

**Before calling the sub-agent Agent, the user must be asked again to confirm:**

```python
ask_user(message="To be used [The way to choose] Generate kernelAre you sure about this?(y/n)")
```

- User reply "y"/ "yes"/ "confirm" → to execute call
- User reply "n"/ "no"/ "cancell" or other way → requiring selection

## SubAgent Arguments

1. `call_coder_only(task_code="...", op_name="relu", user_requirements="...")`
   - task_type: "precise_only" (default) or "profile"
   - Our_requirements: Kernel Optimizing Demand (optional; see note below)

2. `call_evolve(task_code="...", op_name="relu", user_requirements="...")`

3. `call_adaptive_search(task_code="...", op_name="relu", user_requirements="...")`

4. `call_op_task_builder(user_request="...")`

5. `call_kernel_verifier(kernel_code="...", task_code="...", op_name="relu")`

---

## Description of user_requirements parameters (important!)

### Need to pass on user_requirements

**Only "the requirements for kernel realization" need to be passed on to `user_requirements`**, which will be passed to son Agent, affecting the final kernel code:

1. Severation strategy: "Triple-separation in the core" and "Use the partition strategy."
2. Hardware characteristics: "Use tensor core", "use shared memory"


**Example:**
```
User: "Generate ReLU operatorWe're going to double-separate the core."
   → call_coder_only(task_code="...", op_name="relu", user_requirements="Retract the core.")
```

### Do not need to pass on user_requirements

**Users did not transmit the request for the kernel generation**
**"Dequisites for task code" does not transmit `user_requirements`**, but has been completed in front of the `call_op_task_builder` processing.

## Prohibition of conduct

1. Automatically select when the user has not specified
2. Incomplete code direct call sub-Agent
3. Do not double confirm direct call sub-Agent
4. **Show related needs (data type, Shape) into user_requirements**
