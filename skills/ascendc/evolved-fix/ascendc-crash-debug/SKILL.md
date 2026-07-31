---
name: ascendc-crash-debug
description: Ascend C operator card dead/crash/RAM error debugging path is skill. The scene used to process the program cannot run out or perform an abnormal breakdown is: (1) the program card is dead/ hanging up/overtime, Kernel is not responding, (2) the program collapse (Segmentation Fault, Abort), (3) the core hanging of the conflict/mortlock, (4) the plog log log location card dead/crash, (5) the occasional collapse/outside result (suspecting that the memory has crossed the border) requires active detection of the memory error. Triggering key words: the card is dead, hanging up, overtime, crash, hang, crash, deadlock, Security Fault, Abort, Kernelhang, built-in border crossing, plog, memory check, memory check, memory error, occasional crash, aic error, aerrrr, illegal reading.
---

# Ascend C operator card dead/crash debug

Systematic debugging of Ascend C operator problems that could not be run, including the death of the application card/ hanging up, Kernel timeout, and the collapse of the application.

## Rapid diagnosis

```
Programs cannot run out or collapse occasionally
    │
    ├─ Program crashes (%1)Segmentation Fault / Abort)?
    │   ├─ Enable coredump → ulimit -c unlimited
    │   ├─ Generate core Documentation → Run Program
    │   └─ GDB Analysis → gdb <exe> <core> → bt / bt full / info locals
    │
    ├─ The program card's dead./Timeout?
    │   ├─ View plog Log → The location of the card is dead.
    │   ├─ Inspection Buffer Match → AllocTensor/FreeTensor
    │   ├─ Check Sync → EnQue/DeQue Match
    │   └─ Kernel Debug → AscendC::PRINTF / DumpTensor / msDebug
    │
    └─ An occasional collapse./Suspecting that memory error caused an anomaly?
        └─ Memory error active detection → mssanitizer
            ├─ Read and write across bordersIllegal Read/Write)
            ├─ Multi-nucleption.Multi-core Overwrite)
            ├─ Non-matching visitsMisaligned Access)
            └─ Memory Leaks (%1)Memory Leak)
```

## Symptoms - causes quick check

| Symptom | Possible causes | Diagnosis |
|------|----------|----------|
| **Program card dead/overtime** | Buffer Unreleased / Dead Lock | Check AllocTensor/FreeTensor Match, EnQue/ DeQue Match |
| **Core timeout/ hang-up** | Buffer Conflict/ Deadlock | Check Alloc / Free pairs |
| **Segmentation Fault** | Empty pointer decomposition reference / memory crossing | GDB analysis coredump → bt to view call stack; if stacks are not clear → [mssanitizer memory detection] (references/memcheck/) |
| **Abort** | Claim Failed / Unusual End | GDB parsing coremp → examination claim conditions |
| **Dock overflow** | Over-deep / Large arrays | Check group size in Kernel |
| **aic error** | Memory visits cross-border / non-matched visits | Check DataCopy Length, 32B alignment; for example, code logic → [mssanitizer memory detection] (references/memcheck/) |
| **AIV Carded to death** | Trans-nuclear sync missing | Check CrossCorre WaitFlag for SetFlag |
| **drain stuck to death** | No AIV→AIC flag sent | Check flag sending links |
| **An occasional collapse cannot be repeated** | Memory Step / Competition Conditions | [mssanitizer memory detection] (references/memcheck/) (plus -g compile options location) |

## Enter the detailed process by scene

| scene | Annotations | Detailed steps |
|------|------|----------|
| Program crashes | Segmentation Fault / Abort | [Coredump debug] (references/crash_workflow.md) |
| Program card dead/overtime | Kernel no response, hang up. | [Kernel Hangs Debug] (references/crash_workflow.md) |
| Buffer Conflict/ Deadlock | Alloc/Free's not match, sync's missing. | [Buffer Question] (references/crash_workflow.md) |
| Memory error active detection | Cross-border reading and writing/multi-nucleus/non-reciprocal access/RAM leak/incidental collapse | [mssanitizer memory detection] (references/memcheck/) |

## Detailed resources

- **[crash_workflow.md] (references/crash_workflow.md)**: full debugging process + debugging tool swab
- **[parse_plog.py] (scripts/parse_plog.py)**: log resolve script (Auto-ID card death/crash/hardware anomaly signal)
- **[mssanitizer memory detection] (references/memcheck/)**: memory error detection complete workflow
  - [Automated workflow guide] (references/memcheck/automated_workflow.md) - 3-step completed compilation, installation, detection
  - [User Guide] (references/memcheck/README.md) - configuration file Template + common issue
  - [msSanitizer Tool Guide] (references/memcheck/mssanitizer_guide.md) — Class 6 Memory Exhaustion
- **[Automated detection script] (scripts/run_memcheck_pre.sh)**: One key to perform memory testing
- **[configuration file template] (scripts/memcheck_input.json.template)**: memcheck_input.json template
