---
name: triton-ascend-case-vector-elemwise-bench-atlas-a2
description: "Atlas A2Let's go.Triton vectorOne dollar./Two dollars.`tl.*`:fp32/fp16/bf16Three.dtypeDownwards.operatorEnd-to-endtime (ms);and give\"Semantic equivalent,accuracyAlignment\"Should have been replaced.triton APIAnd recommendive writing."
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2"
---

# Replacement of API Recommendations (Atlas A2)

> **accuracy warns**: The following replacements are for the single operator unit test: Semantic Equivalence, accuracy Alignment, but when operator merges or is used on the Internet, the following is used:
> error may be cumulatively amplified.
> **If accuracy cannot be aligned after replacing an API, please retain the original text and do not replace it.**

## fp32

| Original | Suggested Replace | Proceeds |
|---|---|---|
| `x * tl.sigmoid(x)` achieve silu | **`x / (1 + tl.exp(-x))`** | **13%** |
| `tl.sqrt(x)` | `tl.sqrt_rn(x)` / `x*tl.rsqrt(x)` / `1.0/tl.rsqrt(x)` | 5% |
| `2*tl.sigmoid(2x)-1` vs `tl.math.tanh(x)` | **`tl.tanh(x)`** | 4% |

## fp16

| Original | Suggested Replace | Proceeds |
|---|---|---|
| `tl.rsqrt(x)` | **`1.0 / tl.sqrt(x)`** | **24%** |
| `tl.sqrt(x)` | **`tl.sqrt_rn(x)`**or `x*tl.rsqrt(x)` / `1.0/tl.rsqrt(x)` | **27%** |
| `tl.exp2(x)` | **`tl.exp(x * LN2)`** | **24%** |
| `tl.log2(x)` | `tl.log(x) * LOG2E` | 23% |
| `tl.maximum(x, 0)` Achievedrelu | **`(x + tl.abs(x)) * 0.5`**or `tl.where(x>0, x, 0)` | **19%-21%** |
| `(exp(x)-exp(-x))/(exp(x)+exp(-x))` achieves the taunh | **`tl.tanh(x)`**or `1-2/(exp(2x)+1)` / `2*sigmoid(2x)-1` | **47%** |
| `tl.sigmoid(x)` | `exp(x)/(1+exp(x))` or `1/(1+tl.exp(-x))` or `0.5*(1+tanh(x/2))` | 11% |


## bf16

| Original | Suggested Replace | Proceeds |
|---|---|---|
| `tl.exp2(x)` | **`tl.exp(x * LN2)`** | **33%**  |
| `tl.sqrt(x)` | **`x * tl.rsqrt(x)`**or `1.0/tl.rsqrt(x)` / `tl.sqrt_rn(x)` | **27%-31%** |
| `tl.log2(x)` | `tl.log(x) * LOG2E` | 31%  |
| `tl.maximum(x, 0)` Achievedrelu | **`tl.where(x>0, x, 0)`**or `(x + tl.abs(x)) * 0.5` | **21%** |
| `1 - 2/(tl.exp(2*x)+1)` achieves the taunh | **`tl.tanh(x)`**or `(exp(x)-exp(-x))/(exp(x)+exp(-x))` / `2*sigmoid(2x)-1` | 8% |
| `tl.abs(x)` | **`tl.where(x>=0, x, -x)`**or `tl.maximum(x, -x)` | **12%** |
| `tl.sigmoid(x)` | `exp(x)/(1+exp(x))` or `0.5*(1+tanh(x/2))` or `1/(1+tl.exp(-x))` | 9% |
