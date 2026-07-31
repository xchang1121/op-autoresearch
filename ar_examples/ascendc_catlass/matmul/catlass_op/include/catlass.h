#ifndef CATLASS_OP_INCLUDE_CATLASS_H_
#define CATLASS_OP_INCLUDE_CATLASS_H_

#include <torch/extension.h>

namespace catlass_torch {

at::Tensor basic_matmul(const at::Tensor& A, const at::Tensor& B);
at::Tensor optimized_matmul(const at::Tensor& A, const at::Tensor& B);
at::Tensor splitk_matmul(const at::Tensor& A, const at::Tensor& B);

}

#endif // CATLASS_OP_INCLUDE_CATLASS_H_
