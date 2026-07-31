---
name: triton-ascend-basics
description: "Triton Ascend programming base, including core concepts (program_id, block, Grid), kernel structure, decorator usage and standard code model. Application of any internal nuclear code generation scenario using Triton Ascend that requires knowledge of basic syntax structures"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_patterns: "all"
---

# Triton Ascend Programming Base

## Standard kernel structure (stagger cycle)

```python
@triton.jit
def kernel(
    output_ptr, input_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, CORE_NUM):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)
```

## kernel boot template

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch
            import triton
            device = torch.npu.current_device()
            properties = triton.runtime.driver.active.utils.get_device_properties(device)
            self.VEC_CORE_NUM = properties.get("num_vectorcore", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, x):
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (self.VEC_CORE_NUM,)  # Ascend: Fixed to core
        kernel[grid](out, x, x.numel(), BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=self.VEC_CORE_NUM)
        return out
```

## Border processing

```python
offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)
result = tl.where(condition, true_val, false_val)
tl.store(out_ptr + offsets, result, mask=mask)
```

Hard rule:

- Any `tl.load` that may cross the Shape border must be accompanied by `mask` or block_ptr `boundary_check`.
- Any tail block `tl.store` must be accompanied by `mask` or block_ptr `boundary_check`, otherwise the last dissatisfied block will write bad output.
- 2D/3D block_ptr scenario preferred to `boundary_check=(0, 1)`; common pointer scene used visible mask.

## Multi-dimensional Indexing and Spelling

A multi-dimensional task can be equalised to a dimensional task, but it must clearly write the decomposition formula for each dimension and restore the whole of the world to the original shape / structure. Do not leave an ambiguous intermediate index in the code.

```python
task = tl.program_id(0)
b = task // (M * N)
rem = task - b * M * N
m = rem // N
n = rem - m * N
offset = b * stride_b + m * stride_m + n * stride_n
```

Hard rule:

- Each program default can only write its only responsible output area.
- Multiple programs may write the same output address using atomic operations such as `tl.atomic_add` / `tl.atomic_max`, or rewrite the schedule to be unique.
- When stretching a multi-dimensional task, write a formula for `task -> dim0/dim1/...`, then calculate a pointer input; do not mix different dimensions into a non-readable expression.

## Autotune Usage (static Shape only)

Autotune finds the optimal configuration and cache of the current hardware and data size by auto-benchmark multigroup configuration parameters, without manual referencing.

### Apply scene

- **Recommended**: input shape fixed or limited range of changes (static shape), e. g. MatMul for fixed bat size, Attention for fixed sequence length, etc.
- **Ban on use**: Enter Shape Frequent Changes (Dynamic Shape). Autotune best config based on `key` parameters cache, dynamic Shape triggers a full benchmark with a severe drag on chronic energy

### Mandatory rules

1. **Must write `restore_value`**: list all**output pointer parameters**for kernel. uututune benchmark will repeat kernel, `restore_value` will save and output a copy of tensor before each config and restore values after each traverse to prevent contamination of results between different configs.**Failure to write `restore_value` will result in certification failure.**
2. **Call without calling configs parameters**: autotransmittune.
3. **configs must be constexpr**: declared `PARAM: tl.constexpr` in Kernel.
4. **key Parameter**: reautonne when you specify which input dimensions change.
5. **Ascend does not support the modulation**: do not modify the num_warps, num_catas, num_stages, etc., which is currently not supported by Ascend backend.

### Standardized

```python
# Correct: There are current_value, Grid fixed core numbers
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['n_elements'],
    restore_value=['output_ptr'],  # ⚠ Must: list all output pointer parameter names
)
@triton.jit
def kernel(input_ptr, output_ptr, n_elements,
           BLOCK_SIZE: tl.constexpr):
    pass

# Ascend: grid fixed to core
grid = (VEC_CORE_NUM,)
kernel[grid](input_ptr, output_ptr, n_elements)
```

```python
# Error: Lack of resource_value → CodeChecker intercepts, authentication fails
@triton.autotune(
    configs=[...],
    key=[...],
)
@triton.jit
def kernel(input_ptr, output_ptr, ...):
    pass
```

### Autotune, key points
1. **grid must use lmbda**: `grid = lambda meta: (...)`
2. **Call without configs parameters**: autotone
3. **configs parameters must be constexpr**
4. **key Parameter**: reset autotune when specifying which dimensions change
5. **Ascend does not support adjustments**: num_warps / num_ctas / num_stagets

## Core number selection (important)

Ascend NPU has two kinds of cores that must be correctly selected according to the operator type:

- **VEC_CORE_NUM(vectorCore)**: For useelement-wise,reduce,softmaxNormalization, etc.**does not containtl.dot** ofoperator
- **CUBE_CORE_NUM (matmul, attaction, etc.)**operator containing tl.dot**

**Hard bound**: operator**, which relates to `tl.dot` / matrix multiplication calculations, must**use CUBE_CORE_NUM, and hybrid calculations (first matmul and then elementwise) also use CUBE_CORE_NUM. Nucleus access codes and detailed policies can be found in the Grid-config document.

## Output tensor Create

- Output tensor with `torch.empty`/ `torch.empty_like` (avoiding initial cost of `zeros`/`ones`)
- Default output sequence created by `torch.empty_like()`

## Ascend Triton does not support API

The following API exists in CUDA Triton, but is not supported in Ascend Triton**, and its use leads to an error of translation:

| Unsupported API | Alternatives |
|-------------|---------|
| `tl.any` / `tl.all` | `tl.sum(mask.to(tl.int32)) > 0` |
| `tl.histogram` | Manually achieve the barrel logic |
| `tl.sort` | Manual Sorting or Phased Comparison |
| `tl.gather` / `tl.scatter` (part) | `tl.load` / `tl.store` + index calculation |
| `num_warps` / `num_ctas` / `num_stages` (autotune parameters) | Ascend is not needed. Just ignore it. |
