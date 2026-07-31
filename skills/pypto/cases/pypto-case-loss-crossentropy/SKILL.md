---
name: pypto-case-loss-crossentropy
description: "Example D: Loss - CrossEntropyLoss, display multiple input kernel, two slots file, softmax+gather+sum, scalar output"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "loss,reduction,gather,softmax"
---

# Mode D: Los — CrossEntropylos

```python
def create_cross_entropy_kernel(batch, num_classes):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        predictions: pypto.Tensor((batch, num_classes), pypto.DT_FP32),
        targets: pypto.Tensor((batch,), pypto.DT_INT64),
    ) -> pypto.Tensor((1,), pypto.DT_FP32):
        output = pypto.tensor([1], pypto.DT_FP32)
        # Phase 1: per-sample softmax + gather
        pypto.set_vec_tile_shapes(1024, 16)
        log_probs = pypto.log(pypto.softmax(predictions, dim=1))
        targets_i32 = pypto.cast(targets, pypto.DT_INT32)
        idx = pypto.unsqueeze(targets_i32, 1)
        picked = pypto.gather(log_probs, dim=1, index=idx)
        neg_picked = pypto.mul(picked, -1.0)
        # Phase 2: batch reduction
        pypto.set_vec_tile_shapes(2048, 8)
        total = pypto.sum(neg_picked, dim=0, keepdim=False)
        output[:] = total / batch
        return output
    return kernel
```

Forward: Assert → contiguous → to Kernel → `reshape(1,)`

## Elements of a model
- **Two paragraphs file**: different tile configuration at different stages of calculation
- `pypto.cast(targets, DT_INT32)` — INT64 input needs to rotate INT32
- `pypto.unsqueeze` + `pypto.gather` - Press index to extract elements
- `pypto.mul(x, -1.0)` - Counterfeiting standard formulation (application of rule R2)
- scalar output: `pypto.tensor([1], ...)` + `output[:] = scalar`
- Element-by-Element loss(MSE/ Huber, etc.) simpler: all input `reshape(-1)` with 1D Kernel
