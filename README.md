# Op AutoResearch

Op AutoResearch is a standalone workspace for agent-driven optimization of
operator kernels. It combines reproducible correctness checks, performance
measurement, constrained code editing, and an explicit optimization state
machine:

```text
BASELINE -> PLAN -> EDIT -> EVAL -> KEEP / DISCARD / FAIL
                                      -> REPLAN / DIAGNOSE / FINISH
```

The repository includes the workspace workflow, `KernelVerifier`, backend and
DSL adapters, local and remote workers, a device lease pool, static code
checks, runtime guards, skill documentation, Claude Code and OpenCode entry
points, and batch-processing tools. It does not require another source tree.

## Installation

```bash
python -m pip install -e .

# Install the HTTP worker service when needed.
python -m pip install -e ".[worker]"
```

Hardware-specific runtimes are installed separately for the selected backend.
Typical examples include PyTorch, a device extension, a DSL compiler, and the
device SDK.

## Quick start

Place a reference implementation and a seed kernel in `workspace/`. Start
Claude Code or OpenCode from the repository root, then run:

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <device-id>
```

Monitor a task with:

```bash
python scripts/dashboard.py <task_dir> --watch
```

See [AUTORESEARCH.md](AUTORESEARCH.md) for the full workflow and
[AGENTS.md](AGENTS.md) for agent operating constraints.

## Worker service

Manage a local worker:

```bash
op-autoresearch worker --start --backend ascend --arch ascend910b3 --devices 0
op-autoresearch worker --status
op-autoresearch worker --stop
```

For a remote worker, define a host under `remote_worker.hosts` in
`config.yaml`. Its `repo_path` must point to the root of this repository on the
remote machine.

```bash
op-autoresearch worker --remote-host eval-host --start --backend ascend --devices 0
op-autoresearch worker --remote-host eval-host --status
op-autoresearch worker --remote-host eval-host --stop
```

## Repository layout

```text
scripts/                  State machine, evaluation bridge, and batch tools
src/op_autoresearch/      Verifier, workers, device pool, and shared runtime
skills/                   DSL-specific optimization and debugging guidance
.claude/                  Claude Code command, hooks, and diagnostic agent
.opencode/                OpenCode command, plugin, and headless loop
ar_examples/              Minimal examples for supported DSLs
tests/                    Workflow and component tests
```

## Validation

```bash
python -m compileall -q scripts src
python -m pytest -q
```
