---
name: ascendc-direct-invoke
description: "AscendC direct-invoke project contract: The job directory uses kernel.py + ascendc_op/, ModelNew calls Torch.ops.npu.*. The adaptor is responsible for copying the project, CMake construction and npu-arch Patch. operator generation and validation missions for dsl=ascendc."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC Direct-Invoke Works Contract

The `dsl=ascendc` task directory is not an old three-part string protocol, nor is it a fully registered customised operator project. The standard delivery form is a Python wrapper plus an AscendC project that can be constructed by CMake:

```text
task_dir/
  kernel.py
  ascendc_op/
    CMakeLists.txt
    op_kernel/
    op_host/
    op_extension/
    scripts/
```

## 1. A hard deal.

- `kernel.py` is exposed to only one open-entry category: `ModelNew`.
- `ModelNew.forward()` calls for compiled extensions via `torch.ops.npu.<op>(...)` or project-registered namespace.
- `ascendc_op/` saves the CMake project; do not embed the C++/AscendC source code in the Python string.
- `kernel.py` is not responsible for calling CMake, nor is it compiled at the Import stage.
- wrapper code does not call `.npu()`; verifier moves input to target device.
- `.so` has been lazied in `ModelNew._load()` or `forward()` to avoid the side effects of the Import phase.

## 2. Python Wrapper Template

```python
from __future__ import annotations

from pathlib import Path

import torch
import torch_npu


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        build_dir = Path(__file__).with_name("ascendc_op") / "build"
        so_files = sorted(build_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError(f"no AscendC extension found under {build_dir}")
        torch.ops.load_library(str(so_files[0]))
        self._loaded = True

    def forward(self, x, y):
        self._load()
        return torch.ops.npu.my_op(x, y)
```

If there may be multiple `.so`s under `build/`, priority is given to sifting by a stable filename, such as `libmy_op_ops.so`; the first result after sorting is finally used.

## 3. CMake Project Structure

The common project consists of two parts:

- Optional standalone executable for local Kernel launch and debugging.
- PyTorch shared library, registered operator through `TORCH_LIBRARY_FRAGMENT`/`TORCH_LIBRARY_IMPL`.

Recommended file layout:

```text
ascendc_op/
  CMakeLists.txt
  op_kernel/<op>_kernel.asc
  op_kernel/<op>_tiling.h
  op_host/<op>.asc
  op_extension/<op>_torch.cpp
  op_extension/register.cpp
  op_extension/ops.h
```

PyTorch Extension Registration Example:

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("my_op(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("my_op", TORCH_FN(ascend_kernel::my_op_torch));
}
```

## 4. Host Launch Duty

Host bridge code needs to be completed:

- Distributes output on the input device.
- From Shape, dtype, stride extrapolating tilling.
- Copy tilling data to a readable memory or tensor on the device side.
- Get the current NPU stream through `c10_npu::getCurrentNPUStream()`.
- Call launcher in the order of kernel entry.

The Meta function must return the exact output Shape and dtype. The wrong Meta will cause verifier to read the error Shape and then judge the correct Kernel as a failure.

## 5. CMake engagement

The adaptor will transfer to CMake:

```text
NPU_ARCH
ASCENDC_NPU_ARCH
ASCEND_HOME_PATH
Python_EXECUTABLE
Python3_EXECUTABLE
```

The new project gives preference to `${NPU_ARCH}`. If a hard code such as `--npu-arch=dav-2201` or `--npu-arch=dav-3510` exists in an old project, the buildup of a suitable layer will try to catch, but do not continue writing death in the new code.

The project must generate loadable `.so` under `ascendc_op/build/`; the search is retrievable, but the stability of the file name reduces the risk of misloading.

## 6. Vector-type operator conversion path

A single Vector template can be used:

- `op_kernel/<op>_tiling.h`: host/kernel shared tiling story.
- `op_kernel/<op>_kernel.asc`: `KernelXxx` class, consisting of `Init`, `CopyIn`, `Compute`, `CopyOut`, `Process`.
- `op_extension/<op>_torch.cpp`: PyTorch Bridge, Tiling Calculating, Steam Selection, Kirnel lanch.
- `register.cpp`: `torch.ops.npu.<op>` registration and Meta realization.

When you migrate an existing template, replace the name operator with a global name and change only the semantic point: number of inputs/outputs, tilling fields, calculation expression, output Shape, dtype check, CMake target name.

## 7. Prevent Generating Forms

Do not generate:

- `host_tiling_src = """..."""` such Python embedded source code.
- `kernel_src = """..."""` such as runtime collating source code.
- Only depend on `run.sh` for a one-time workflow.
- The iport phase compiles and iport phase selects device.
- Hard-coding local data id.

The mission product should be re-engineered, capable of being taken over by a certification link, rather than a single local script.
