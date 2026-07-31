"""utils/ — stateless library modules imported by engine/, hooks/,
phase_machine/, workflow/, task_config/, and batch/.

No CLI entry points live here. The Triton regression check (no
@triton.jit / forbidden torch.* in forward / etc.) is delivered by
`op_autoresearch.op.utils.code_checker.CodeChecker` directly — quick_check
and batch/verify call it without a WA-side wrapper.

Nothing in this package mutates state. Splitting them out makes the
dependency direction obvious: utils sits at the bottom of the stack and
never imports from any sibling package.

---------------------------------------------------------------------------
Invariant — IMPORT STYLE INSIDE utils/

When a module in utils/ imports another utils/ module, use the
relative form: `from .settings import …`, NOT the absolute
`from settings import …`. The absolute form silently relies on
`scripts/utils/` being in sys.path — daemons (worker, batch driver)
add only `scripts/`, so the absolute form works in ad-hoc CLI runs
but blows up the first time a long-running process imports the
module.

If you're adding a new utils module that needs another utils module:
always use `from .<sibling> import X`. Audit your callsite for the
absolute form before pushing.
---------------------------------------------------------------------------
"""
