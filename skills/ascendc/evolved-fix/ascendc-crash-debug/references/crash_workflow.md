# Calcination/ Collapse debugging workflow

## Quick Decision Tree

```
Programs cannot run out or collapse occasionally
    │
    ├─ Program crash?
    │   └─ Coredump Debug
    │       ├─ GDB Analysis coredump Documentation
    │       └─ If the stack is not clear → Process3:mssanitizer Actively detect memory error
    │
    ├─ The program card's dead./Timeout?
    │   └─ KernelSuspend debugging → Viewplog → The location of the card is dead.
    │
    └─ An occasional collapse./Suspecting that memory error caused an anomaly?
        └─ Process3:mssanitizer Actively detect memory error
            ├─ The stack isn't clear.
            ├─ An occasional collapse can't be repeated.
            └─ aic error The code logic is complicated.
```

## Process 1: Kernel hangs up debugging

### Step 1: View the plog log

```bash
# plog default path
ls $HOME/ascend/log/debug/plog/plog-pid_*.log

# Or open the log screen.
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# Parsing logs using parse_plog.py (scripts/ directory in crash-debug skill)
python3 scripts/parse_plog.py <plog_file>
```

### Step 2: Analyse plog content

**Core timeout**

```
Symptoms: appearance in logs "timeout" Or the procedure has been unresponsive for a long time.

Possible causes:
    ├─ Buffer Not released → Inspection AllocTensor/FreeTensor Match
    ├─ Deadlock. → Inspection EnQue/DeQue Match
    ├─ Infinite Cycle → Check Loop End Conditions
    └─ Block Operations → Check Sync Point
```

**Memory visits crossed borders**

```
Symptoms:aic error or the procedure has been unresponsive for a long time

Possible causes:
    ├─ DataCopy Length Error → Inspection size Parameters
    ├─ GM Address Error → Inspection offset Calculate
    ├─ UB Cross-border visits → Inspection buffer Size
    └─ Non-matching visits → Inspection 32 Byte Alignment
```

### Step 3: Kernel debugging method

**Method 1: Printf debug**

```cpp
// Print key variables in Kernel
AscendC::PRINTF("blockLength=%llu, tileNum=%llu\n", blockLength_, tileNum_);
```

**Method 2: DumpTensor debug**

```cpp
// Print tensor content
AscendC::LocalTensor<T> xLocal = inQueue.DeQue<T>();
DumpTensor(xLocal, 0, 128);  // Before printing128An element
```

**Method 3: single-step debugging**

```bash
# One-step debugging using msDebug
# Reference: https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug
```

## Process 2: Coredump debugging (programme crash)

**Applicable scenario**: program crash, Segmentation Fault, Abort

### Step 1: Enable coredump

```bash
ulimit -c unlimited  # Enable coredump
```

### Step 2: Generate and analyse coredump

```bash
# Run the program (create a core file on crash)
./your_executable

# Use GDB analysis
gdb <executable> <core_file>

# GDB Common Commands
bt              # Viewcall stack
bt full         # View Fullcall stack(includes local variables)
frame N         # Switch to No. N Thrust
info locals     # View local variables
p variable      # Print Variable Values
```

### Step 3: Positioning problems

Common causes of collapse:
- **Empty pointer decomposition reference**: check whether tensor is fullptr
- **Memory crossing**: checking DataCopy length, GM/UB access
- **Flush**: check re-entry depth or large arrays

## Buffer related card death/crash

The following issues may lead to the death or collapse of the card:

| Problem | Performance | Solutions |
|------|------|----------|
| Buffer not released | Core hung/timeout | Alloc after cycle must Free |
| Core timeout/ hang-up | Program not responding | Check Buffer Conflict/ Deadlock → Alloc / Free pair |
| VECIN for output | Output equals input | The output must be queued with VECOUT |
| Double Buffer | Threshold error | ×2 when calculating thresholds |

### Deadlock Conflict with Buffer

```
Common death lock pattern:
    ├─ Buffer Distribution versus release → The same in the cycle buffer Multiple Alloc not Free
    ├─ EnQue/DeQue It doesn't match. → Queue to wait or be full
    ├─ MNA Missing → CrossCoreWaitFlag No corresponding SetFlag
    └─ PipeBarrier Abuse → AllpipelineStop, follow-up operations rely on incomplete data
```

## Process 3: mssanitizer memory testing

**Applicable scene**: an occasional collapse cannot be repeated, aic error code logic is complicated, coredump stacks are unclear, memory errors are suspected of causing abnormality and active memory detection errors.

### Detectable Class 6 memory anomalies

| Unusual Type | Level | Meaning |
|----------|------|------|
| Illegal Read/Write | ERROR | Access unallocated memory areas |
| Multi-core Overwrite | WARNING | Multiple nuclear access to overlapping GM areas and at least one data entry. |
| Misaligned Access (non-matched access) | ERROR | DMA address does not meet 32B alignment requirements |
| Illegal Free (illegal release) | ERROR | Release unallocated or released memory addresses |
| Memoory Leak (RAM leak) | ERROR | Not released after application memory (`--leak-check=yes` required) |
| Unused Memory (distributed unused) | WARNING | The assigned memory was never accessed (to add `--check-unused-memory=yes`) |

### Fast start.

```bash
# 1. Copy configuration template to fill in operator information
cp scripts/memcheck_input.json.template ./memcheck_input.json

# 2. Copy and implement automated testing scripts
cp scripts/run_memcheck_pre.sh .
chmod +x run_memcheck_pre.sh
./run_memcheck_pre.sh

# Analysis of output results
grep "====== ERROR:" <code_base>/memcheck_output/memcheck/ascendc_memcheck_report_raw.txt
grep "====== WARNING:" <code_base>/memcheck_output/memcheck/ascendc_memcheck_report_raw.txt
```

### Detailed documents

- **[memcheck] (memcheck/)**- Full workflow (analysis + report generation)
- **[memcheck/README.md](memcheck/README.md)** —Preconditions+Configure Field Description+ common issue
- **[memcheck/automated_workflow.md](memcheck/automated_workflow.md)** —Detailed parameters for automated scripts
- **[memcheck/ mssanitizer_guide.md] (memcheck/mssanitizer_guide.md)**- msSanitizer Tool Original Document

## Debug Tool Scanning

| Tools/methods | Purpose | Use scene |
|----------|------|---------|
| 'plog log ' | View runtime log | Carcasses/crash analysis |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Log screen | Require real time view of logs |
| `parse_plog.py` | Log Resolution | Automatically extract error/timeout/crash information |
| `AscendC::PRINTF` | Kernel Inner Print | Kernel Logical Debugging |
| `DumpTensor` | Printtensor Contents | Data validation |
| `msDebug` | Step Debug | Complex issues (calculation, cross-border) |
| `ulimit -c unlimited` | Enable coredump | Pre-Crash Settings |
| `gdb <exe> <core>` | Analysis coremp | Prefer to program crashes |
| `mssanitizer --tool=memcheck` | Actively detect memory error | Undefined stacks, occasional collapses, multi-nuclei, aic error |
