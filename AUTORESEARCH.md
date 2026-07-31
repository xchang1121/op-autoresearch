# AutoResearch workflow

AutoResearch uses Claude Code or OpenCode to optimize operator kernels. The
user supplies a reference implementation and a seed kernel; the state machine
then establishes a baseline, plans an edit, evaluates it, and decides whether
to keep the result:

```text
BASELINE -> PLAN -> EDIT -> EVAL -> KEEP / DISCARD / FAIL
                                      -> REPLAN / DIAGNOSE / FINISH
```

Normal use requires only `/autoresearch`. Hooks and plugins manage phase
transitions and constraints. Do not edit `.ar_state/state.json` or
`.ar_state/plan.md` manually.

## 1. Quick start

The evaluation environment is normally Linux. Enter the repository and load
the Python, hardware SDK, and DSL environment:

```bash
cd <op-autoresearch-repo>
python -m pip install -e .
source ~/env.sh
```

Confirm that the target device and Python runtime are available. For example,
an Ascend setup may use:

```bash
npu-smi info
python -c "import torch; import torch_npu"
```

Start either interactive client:

```bash
claude
# or
opencode
```

Create a task:

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <device-id>
```

Frequently used options:

| Option | Purpose |
|---|---|
| `--max-rounds N` | Maximum number of optimization rounds. |
| `--eval-timeout SEC` | Wall-clock budget for one evaluation. |
| `--output-dir DIR` | Parent directory for task output; defaults to `ar_tasks/`. |
| `--worker-url HOST:PORT` | Use an already running remote worker. |
| `--no-code-checker` | Disable CodeChecker for this task. |

Resume a task:

```text
/autoresearch --resume
/autoresearch --resume <task_dir>
```

Monitor progress:

```bash
python scripts/dashboard.py <task_dir> --watch
```

Run one headless OpenCode task:

```bash
source ~/env.sh
python .opencode/run_loop.py \
  --ref workspace/<op>_ref.py \
  --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <device-id>
```

## 2. Input files

### 2.1 Reference implementation

The reference file must expose at least:

- `Model`
- `get_init_inputs()`
- `get_inputs()` or `get_input_groups()`

Recommended path:

```text
workspace/<op>_ref.py
```

Sidecar `.json`, `.pt`, and `.npz` inputs with the same base name are copied
with the task and transferred to remote workers. Keep paths relative to the
reference directory.

### 2.2 Seed kernel

A single-file DSL kernel exposes `ModelNew`:

```text
workspace/<op>_kernel.py
```

Directory-based DSLs pass a directory to `--kernel`, with a Python wrapper in
an adjacent `kernel.py`. The scaffold records the allowed edit set in
`task.yaml: editable_files`; the agent must modify only those files.

Example directory layout:

```text
workspace/<op>/
  reference.py
  kernel.py
  catlass_op/
    CMakeLists.txt
    kernel/
    include/
    src/
```

```text
/autoresearch --ref workspace/<op>/reference.py \
  --kernel workspace/<op>/catlass_op \
  --op-name <op> --devices <device-id>
```

Defaults for `backend`, `framework`, `dsl`, and the skill family come from
`config.yaml`. The worker detects the hardware architecture unless it is
provided explicitly.

## 3. Remote workers

A developer machine may run the agent while a Linux evaluation machine runs
the worker. The remote machine needs:

- a checkout of this repository;
- an environment script that works in a non-interactive SSH shell;
- the target backend, DSL, SDK, and compiler toolchain.

Configure an SSH alias, then register the host in `config.yaml`:

```yaml
remote_worker:
  hosts:
    eval-host:
      repo_path: /path/to/op-autoresearch
      env_script: /path/to/env.sh
      ssh_alias: eval-host
```

Start the worker and its local SSH tunnel:

```bash
source ~/env.sh
op-autoresearch worker --remote-host eval-host --start \
  --backend ascend --arch ascend910b3 --devices 0
```

Check status:

```bash
op-autoresearch worker --remote-host eval-host --status
```

Create a task through the tunnel:

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices 0 --worker-url 127.0.0.1:<port>
```

Stop the worker and tunnel:

```bash
op-autoresearch worker --remote-host eval-host --stop
```

After changing worker-side code, synchronize the checkout and restart the
worker. A running daemon does not reload source files automatically.

## 4. Batch execution

Use this layout:

```text
<batch_dir>/
  refs/<op>_ref.py
  kernels/<op>_kernel.py
```

Prepare and pre-screen the batch:

```bash
python scripts/batch/prepare.py <batch_dir>
python scripts/batch/verify.py <batch_dir>
```

Run full verification through `KernelVerifier`:

```bash
python scripts/batch/verify.py <batch_dir> --full --devices <device-id>

# Remote worker:
python scripts/batch/verify.py <batch_dir> --full \
  --worker-url 127.0.0.1:<port>
```

Run, monitor, and summarize:

```bash
python -u scripts/batch/run.py <batch_dir> --devices <device-id>
python scripts/batch/monitor.py <batch_dir>
python scripts/batch/summarize.py <batch_dir>
```

For remote execution, add `--worker-url 127.0.0.1:<port>` to `run.py`.
Useful filters include `--only`, `--limit`, and `--retry-errored`.

## 5. Pipeline and traces

During each EDIT round, the agent implements the active plan item and runs:

```bash
python scripts/engine/pipeline.py "<task_dir>"
```

The pipeline performs the quick check, evaluation, KEEP/DISCARD/FAIL
settlement, and phase transition. After it completes, follow the final
`[AR Phase: ...]` guidance instead of guessing the next phase.

Ascend tasks can collect timeline evidence:

```bash
python scripts/engine/pipeline.py "<task_dir>" --trace
```

Each trace is stored under:

```text
.ar_state/op_autoresearch_verify/<op>/Iteration<op>_Step<round>_verify/
```

Important artifacts include:

- `kernel_details.csv` and `op_statistic.csv` for per-kernel timing;
- `trace_view.json` for a complete Perfetto or `chrome://tracing` timeline.

PLAN, REPLAN, and DIAGNOSE guidance discovers these artifacts automatically.
CUDA tasks do not use the msprof trace path.

## 6. Task state and artifacts

| Path | Purpose |
|---|---|
| `task.yaml` | Task configuration and `editable_files`. |
| `.ar_state/state.json` | Current phase and progress; treat as read-only. |
| `.ar_state/plan.md` | Current plan, managed by `create_plan.py`. |
| `.ar_state/history.jsonl` | Round results and decisions. |
| `.ar_state/report.md` | Final report. |
| `.ar_state/op_autoresearch_verify/` | Verification, profiling, and optional trace artifacts. |

Recovery guide:

| Situation | Action |
|---|---|
| Session interrupted | `/autoresearch --resume <task_dir>` |
| Inspect current phase | `python scripts/dashboard.py <task_dir>` |
| Worker unavailable | Run `op-autoresearch worker --remote-host <alias> --status`. |
| Tunnel interrupted | Repeat the worker `--start` command. |
| Correctness failure | Inspect the pipeline summary and corresponding verify directory. |
| Repeated failures | Allow the state machine to enter DIAGNOSE; do not edit state manually. |

## 7. Local client configuration

Shared hooks, plugins, and permission settings are versioned with the
repository. Keep model endpoints and API keys in local-only files:

- Claude Code: `.claude/settings.local.json`
- OpenCode: `.opencode/opencode.json`

Never write API keys to shared settings, `config.yaml`, task files, or logs.
