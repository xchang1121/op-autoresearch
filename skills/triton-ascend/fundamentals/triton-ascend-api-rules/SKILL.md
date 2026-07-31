---
name: triton-ascend-api-rules
description: "Triton Ascend hard API restrictions and forbidden syntax. MUST-follow rules that apply to every kernel: forbidden control flow (return/break/continue/lambda/while), tensor slice/index restrictions, scalar conversion rules, BLOCK_SIZE upper bound. Violating any of these produces a compile or runtime error on Ascend."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3, Atlas A5"
---

# Triton Ascend API Hard Rules (MUST follow)

This file lists rules whose violation causes a compile or runtime error
on Ascend. They apply to every kernel in this DSL — no exceptions.

## Syntax: No. of use

- `return` / `break` / `continue` → Use mask control
- lmbda → inline function or tl.where
- Chain-based boolean calculation → step-by-step calculation
- tensor Direct Index → tl.load / tl.store
- Negative offset in if-else → tl.maxum (offset, 0)
- Complex tl. where → if-else
- Disables disassembly from `tl.where(cond, ptr_a, ptr_b)` selection pointer, address, or memory offset → to static if/else branch, or create load/store after mask
- Prohibits replacing with → for while recycling
- Start/stop forbids the mix of runtime variables and constexpr → with the full range+ cycle runtime if
- Reduction scalar results are not available for `[0]` indexes, e. g. using `result = tl.sum(x, axis=0)` directly after `result`

## Wheel Cycle Substitution (Ascend)

**Static cap**(converted constant): Direct `for i in range(N_ITERS)`

**Dynamic cap**(runtime parameter):
```python
@triton.jit
def kernel(ptr, n_iters, TILE: tl.constexpr, MAX_ITERS: tl.constexpr):
    for i in range(MAX_ITERS):
        if i < n_iters:
            offset = i * TILE + tl.arange(0, TILE)
            data = tl.load(ptr + offset)
            tl.store(ptr + offset, data * 2)
```

## Slice Operations

- Python Slicing Ban `b[0]` `b[i:j]`
- Modular: `tl.get_element(tensor, (index,))`
- Slice: `tl.extract_slice(tensor, offsets, sizes, strides)`
- Insert: `tl.insert_slice(full, sub, offsets, sizes, strides)`
- Prohibits the use of tl.ange tensor to get_election
- Prevents the return of scalar to `[0]`, for example `tl.sum(...)[0]`

## Other restrictions

- tl.constexpr is only used in kernel parameters, host side is not available
- Output tensor uses torch.emtty/ empty_like (avoiding initial zeros/ones costs)
- scalar conversion only `scalar.to(type)` is forbidden `tl.float16(scalar)`
- BLONK_SIZE must be less than 65536
- `tl.dot` uses fp32 accumulator: `acc = tl.zeros((BM, BN), dtype=tl.float32)`, and `tl.dot(a, b, acc)`
