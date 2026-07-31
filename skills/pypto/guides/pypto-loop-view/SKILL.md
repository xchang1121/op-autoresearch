---
name: pypto-loop-view
description: "The correct writing for pypto.loop + pypto.view: View Shape must be a constant compilation period for all loop scenes such as matmul, nom, elementwise"
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "matmul,norm,elementwise,reduction,loop,view"
---

# Loop + View: Compiler Constant Rule

## Fatal error: `ValueError: Not concrete value`

**The most common first-time generation error**. All `pypto.view` parameters must be**compilation period constant**(physical or closed-pack variables).

```python
# WRONG — min() contains loop variable offset and is runtime value
for idx in pypto.loop(0, num_iters, 1, ...):
    offset = idx * BASIC_BATCH
    current = min(BASIC_BATCH, total_size - offset)      # runtime!
    chunk = pypto.view(x, [current, n], [offset, 0])     # ValueError!
```

**Any expression with loop idx is a runtime value**and cannot be used for viewshape.

## Common correct writing

### Method A: Ensure completeness (recommended)

Select BASIC_BATCH to enable total_size to be severed, or assert to be severed in forward:

```python
def create_kernel(total_rows, cols, basic_batch):
    assert total_rows % basic_batch == 0
    num_iters = total_rows // basic_batch  # Closed constant

    @pypto.frontend.jit(...)
    def kernel(x: pypto.Tensor((total_rows, cols), ...)) -> ...:
        output = pypto.tensor([total_rows, cols], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 8192)
        for idx in pypto.loop(0, num_iters, 1, name="LOOP", idx_name="idx"):
            off = idx * basic_batch
            chunk = pypto.view(x, [basic_batch, cols], [off, 0])
            result_chunk = chunk * 2.0  # Example Operations
            pypto.assemble(result_chunk, [off, 0], output)
        return output
    return kernel

class ModelNew(torch.nn.Module):
    def forward(self, x):
        total_rows = x.shape[0] * x.shape[1]
        # Interference by loop_count space driver: try 16/32 first and then reverse Basic_batch
        target_loop_count = 16
        assert total_rows % target_loop_count == 0
        basic_batch = total_rows // target_loop_count
        ...
```

### Method B: Main Loop + End

When there is no guarantee that the whole will be divided:

```python
def create_kernel(total_rows, cols, basic_batch):
    full_iterations = total_rows // basic_batch
    tail = total_rows % basic_batch
    tail_offset = full_iterations * basic_batch
    # Full_actions, tail, tail_offset are all closed constants.

    @pypto.frontend.jit(...)
    def kernel(x: pypto.Tensor((total_rows, cols), ...)) -> ...:
        output = pypto.tensor([total_rows, cols], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 8192)

        for idx in pypto.loop(0, full_iterations, 1, name="LOOP", idx_name="idx"):
            off = idx * basic_batch
            chunk = pypto.view(x, [basic_batch, cols], [off, 0])
            pypto.assemble(chunk * 2.0, [off, 0], output)

        if tail > 0:  # Compiler-time request (%1)tail It's a closed constant.
            tail_chunk = pypto.view(x, [tail, cols], [tail_offset, 0])
            pypto.assemble(tail_chunk * 2.0, [tail_offset, 0], output)
        return output
    return kernel
```

## 1D Loop (global return/large vector)

```python
LOOP_CHUNKS = 8

def create_frobenius_kernel(flat_size):
    chunk_size = flat_size // LOOP_CHUNKS  # Closed constant

    @pypto.frontend.jit(...)
    def kernel(x: pypto.Tensor((flat_size,), ...)) -> ...:
        output = pypto.tensor([flat_size], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(16384)
        acc = pypto.zeros([1], dtype=pypto.DT_FP32)
        for i in pypto.loop(0, LOOP_CHUNKS, 1, name="LOOP_ACC", idx_name="i"):
            x_chunk = pypto.view(x, [chunk_size], [i * chunk_size])
            part = pypto.sum(x_chunk * x_chunk, dim=0, keepdim=True)
            acc[:] = acc + part
        norm = pypto.sqrt(acc)
        output[:] = x / norm
        return output
    return kernel

class ModelNew(torch.nn.Module):
    def forward(self, x):
        x_flat = x.reshape(-1)
        assert x_flat.numel() % LOOP_CHUNKS == 0
        ...
```

## Key principles

1. **view Shape can only be used as a word or closed variable**. Loop idx and its expression with idx is runtime values.
2. **`if tail > 0:` is requested during the compilation period**(tail is a closed-pack constant), not a branch of runtime.
3. **Priority is given to ensuring completeness**and avoiding the complexity of tail processing.
4. **Matmul does not fix BASIC_BATCH**. First in `loop_count` Space Selection (`1~128` usually starts from `16/32`), then invert `BASIC_BATCH`; if necessary, expand to `8/64` and lastly fill the endpoint.
