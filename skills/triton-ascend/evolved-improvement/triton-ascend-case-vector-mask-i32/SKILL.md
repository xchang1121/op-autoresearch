---
name: triton-ascend-case-vector-mask-i32
description: "More than the two sides themselves.`tl.int32`(offset / attn_argI'm sorry, I'm sorry.`arith.cmpi`The output is:`i1`; more than half of the results are in**`i1` tensorDo it.`&`/`|`** when, loweringIt'll be inserted near every logic.**`extui`/`trunci`** and `select`Alignment;**As soon as each section compares.`.to(tl.int32)`**Let the whole thing go.mask in **`i32` 0/1Let's go.`&`/`|`**,backendIt's easier to handle on a continuous basis.**`vand.i32`/`vor.i32`**."
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
---

## Task characteristics
- Multiple paragraphs**`tl.int32` participation**(e.g. `q_off`/`k_off`, `q_attn_arg`/`k_attn_arg`), received**bool and spelled with `&` / `|` as mask**and finally handed over to `tl.where`.

## Reason

Jean.arith.cmpi of `i1`And the result is...`tensor<…xi1>`Up and over.`andi`/`ori`,Ascend backendIt'll be in every logic and last.`arith.select`Additional insert between`arith.extui` / `arith.trunci`Wait, wait, wait.vectorThe width to be used before and after alignment`tl.int32`Showdown.`.to(tl.int32)`In the middle, no more.`i1`Top alignment width,vectorMaking commands easier to use`vand.i32` / `vor.i32`.

## References

```python
@triton.jit
def mask(...):
    triu = (q_off[:, None] <= k_off[None, :]).to(tl.int32)
    return (
        (triu & ((q_arg[:, None] == k_arg[None, :]).to(tl.int32)
                 | (k_arg[None, :] == 0).to(tl.int32)))
        | (q_off[:, None] == k_off[None, :]).to(tl.int32))
```

## Attention.
- The mask syntax is only**0/1**, data type of `tl.int32` is derived from data type on both sides of the comparative operator.
