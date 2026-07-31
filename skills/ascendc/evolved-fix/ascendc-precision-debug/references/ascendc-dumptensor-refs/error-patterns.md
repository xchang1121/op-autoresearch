# Error pattern in DumpTensor perspective

> The focus of this paper is "What anomalies → should be checked in which direction you see in the dump output." Father SKILL's Symptoms-Cause Quick Checklist is more comprehensive, and only the models related to the diagnostic strength of the DumpTensor plugin are listed here.

## DumpTensor exclusive diagnostic model

| Dump phenomenon                                       | The most probable root cause.                          | Next                                                         |
|-------------------------------------------------|---------------------------------------|----------------------------------------------------------------|
| Disc = 100 (input) is abnormal.                          | DataCopy Unfinished / Without DeQue    | Move DumpTensor after DeQue; or temporary PipeBarrier authentication        |
| Desc = 100 Correct, desc = 200 (median) abnormal             | Compute Phase Question                      | Disaggregated stake inside Compute, binary position which API/parameter                |
| dsc = 200 Correct, desc = 300 (output) all 0 / all old    | Copyout is not valid/ Queue error (VECIN)     | Check output queue type, EnQue/DeQue, DataCopyPad alignment               |
| Monochrome dump. Correct, multi-nucleus merger chaos.                    | desc does not contain blockIdx, multiple log staggered      | `desc = base + GetBlockIdx() * 1000`                          |
| Dump display input does not match CPU gold               | host side data preparation / DataCopy profile error  | Compare host buffer, check DataCopy arguments                          |
| Every N is an anomaly                             | / Alignment                     | Check DataCopy side, do you need DataCopyPad                    |
| Nump / Inf                             | Except zero, exp spill, not initialized              | Add dump up to the earliest desc in the emergence of NN                   |
| After changing the code, the dump didn't change at all.                          | Binary unupdated / Kernel Cache hit      | `rm -rf build/ $HOME/atc_data/kernel_cache/` Recoding             |

## General error mode quick check

Use this when the dump is not sufficient to locate and needs to return to the common idea:

| Pattern                    | Root Cause                    | Fix                                            |
|----------------------------|-------------------------------|------------------------------------------------|
| All values off by constant | Missing bias/scale            | Check Adds/Muls operations                     |
| Every Nth value wrong      | Stride/alignment issue        | Verify DataCopy stride, check vector alignment |
| NaN or Inf values          | Division by zero/overflow     | Check denominators, verify input ranges        |
| First/last values wrong    | Boundary/padding issue        | Check tile alignment, edge case handling       |
| Errors accumulate          | Uninitialized vars/queue sync | Check init, verify EnQue/DeQue                 |
| Random sporadic errors     | Race condition/queue depth    | Increase BUFFER_NUM, check sync                |
| Output all zeros           | Missing compute/wrong queue   | Verify Compute called, check queue             |
| Output matches input       | Computation not applied       | Verify operation executed                      |

A more complete accuracy trap can be found in the [common-traps.md] (../common-traps.md) of the parent SKILL.

## Diagnosis Workstream

1. From desc minimum value (input side), find**the first phase inconsistent with CPU gold
2. Match the root causes against the DumpTensor exclusive diagnostic model table above
3. Add an additional layer upstream at this stage. Dump, verify the assumption.
4. After repair, the cache rerun must be compiled to confirm the dump value change
