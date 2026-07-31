"""Pure phase selection for resume and settled-round events.

This module never writes state.  The transaction that owns an event also
commits its target phase in the same ``save_state`` call as the event's other
control fields.  Keeping selection pure makes the transition table testable;
keeping writes with the transaction removes hook/process crash windows.
"""
from __future__ import annotations

import os

from phase_machine import (  # noqa: E402
    BASELINE, DIAGNOSE, EDIT, FINISH, PLAN, REPLAN,
    get_active_item, has_pending_items, load_progress, plan_path,
)


def phase_after_round(task_dir: str) -> str:
    """Select the phase after a round body and plan settlement complete."""
    progress = load_progress(task_dir)
    if progress is None:
        raise RuntimeError(
            "cannot settle a round without committed baseline progress")

    if progress.eval_rounds >= progress.max_rounds:
        return FINISH

    from utils.settings import consecutive_fail_threshold
    if progress.consecutive_failures >= consecutive_fail_threshold():
        return DIAGNOSE
    if has_pending_items(task_dir):
        return EDIT
    return REPLAN


def phase_on_resume(task_dir: str) -> str:
    """Derive the executable phase for an interrupted task."""
    progress = load_progress(task_dir)
    if progress is None:
        return BASELINE
    if progress.eval_rounds >= progress.max_rounds:
        return FINISH
    if progress.seed_metric is None or progress.baseline_outcome != "ok":
        return PLAN
    if not os.path.exists(plan_path(task_dir)):
        return PLAN
    if get_active_item(task_dir) is not None or has_pending_items(task_dir):
        return EDIT
    return REPLAN
