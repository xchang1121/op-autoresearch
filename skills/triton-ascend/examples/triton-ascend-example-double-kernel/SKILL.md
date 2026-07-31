---
name: triton-ascend-example-double-kernel
description: "Triton Ascend is an example of a two-kernel mode of call. Shows two standard kernels in forward: the distribution of the intermediate result buffer zone, and two kernels. This applies to the integration of operator, which needs to be calculated in stages."
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  framework: torch
---

# Double Kernel (Biker Call) - Triton Ascend Example

When an operator requires a two-step calculation (e.g., a transformation before a contract is contracted), two Kernels can be started in `forward` in turn:

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
        intermediate = torch.empty_like(x)
        output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        grid = (self.VEC_CORE_NUM,)

        kernel_stage1[grid](x, intermediate, ..., CORE_NUM=self.VEC_CORE_NUM)
        kernel_stage2[grid](intermediate, output, ..., CORE_NUM=self.VEC_CORE_NUM)

        return output
```

**Element**
- Middle buffer with `torch.empty_like` or specified shape distribution
- Ensure stage1 to read after writing (Triton Ascend default hidden sync)
