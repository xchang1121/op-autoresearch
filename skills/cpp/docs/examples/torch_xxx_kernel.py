import torch
from torch.utils.cpp_extension import load_inline

# Inline C++ Extension Code
cpp_source = """
#include <torch/extension.h>

torch::Tensor op_name_kernel(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    //Specific code is achieved!
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("op_name_kernel", &op_name_kernel, "CPU op_name operator");
}
"""

# Dynamically load C++ extensions
op_name_module = load_inline(
    name="custom_op_name",
    cpp_sources=cpp_source,
    extra_cflags=["-O3"],
    verbose=True
)

# Python Interface Functions


def op_name(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu":
        x = x.cpu()
    return op_name_module.op_name_kernel(x)
