# Op AutoResearch agent guide

This repository is a standalone operator-kernel optimization workspace. An
agent improves a measurable objective through a controlled
plan-edit-evaluate-keep/discard loop. The state machine, hooks, commands,
verifier, workers, and skill library are all contained here.

## Quick start

```bash
python -m pip install -e .
# HTTP worker support: python -m pip install -e ".[worker]"
```

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices 0
/autoresearch --resume
```

For the complete operator workflow, read
`.claude/commands/autoresearch.md` and `AUTORESEARCH.md`.

## Skill library

The skill root is `skills/`, organized by DSL. `.claude/settings.json` exposes
it through `OP_AUTORESEARCH_AR_SKILLS_ROOT=skills`. During PLAN, read the one
to three most relevant `SKILL.md` files and cite their paths in the plan
rationale.

## Operational invariants

1. `.ar_state/plan.md` is the sole source of truth for the plan. Only the
   settlement transactions in `create_plan.py` and `pipeline.py` may write it.
2. Plan IDs increase globally as `p1`, `p2`, and so on. Never reuse or skip an
   ID.
3. Settle every plan item as KEEP, DISCARD, or FAIL. An item may be abandoned
   only at a REPLAN or DIAGNOSE boundary, and the counter must still advance.
4. Baseline creation, plan submission, and round settlement are eventful
   transactions. Each transaction atomically records results, target cases,
   replay commands, and the new state in `state.json`. Hooks observe committed
   state and emit guidance; they do not own state transitions.
5. Edit only files listed in `task.yaml.editable_files`.
6. After interruption, use `/autoresearch --resume`; do not edit state files by
   hand.
7. If `create_plan.py` rejects a plan, revise it according to stderr. Do not
   resubmit an unchanged plan.
8. When guidance returns a Plan Mirror Payload, reproduce it exactly instead
   of composing a new plan manually.
9. Run workspace scripts as direct, foreground, top-level commands. Do not
   hide them behind shell wrappers, background jobs, or command chains.
10. DIAGNOSE should first use `ar-diagnosis` to produce a diagnostic artifact
    containing `Root cause`, `Fix direction`, and `What to avoid`, mark it
    complete, and generate the next plan. Manual fallback is allowed only
    after five consecutive failures.
11. Stop only in FINISH. Exhausted budget or repeated failure transitions to
    DIAGNOSE rather than ending the workflow early.

## Component boundaries

- `scripts/`: workspace state machine, batch tools, and evaluation entry points.
- `src/op_autoresearch/op/verifier/`: `KernelVerifier`, profiling, and backend,
  DSL, and framework adapters.
- `src/op_autoresearch/core/worker/`: local and remote workers plus the manager.
- `src/op_autoresearch/worker/server.py`: HTTP worker service.
- `src/op_autoresearch/core/async_pool/`: device lease pool.
- `src/op_autoresearch/op/utils/code_checker/`: static checks and runtime guards.
- `skills/`: the skill documentation source of truth.

## Dependencies

- Python 3.10 or newer.
- Base dependencies from `pyproject.toml`.
- Optional HTTP worker dependencies from `.[worker]`.
- Backend-specific frameworks, device extensions, DSL compilers, and hardware
  SDKs.
- Claude Code or OpenCode.
