"""Runtime-guard policy, loaded from the shared CodeChecker config.

Single source of truth is ``op_autoresearch/op/config/code_checker.yaml`` under
``ascendc_anti_cheat.runtime_guard`` — the same file the static CodeChecker
reads, so the static and runtime anti-cheat blacklists never drift. Missing or
malformed keys surface as KeyError / TypeError on first access (no fallback
defaults), matching ``op.utils.code_checker``.
"""

import importlib.resources

import yaml

with importlib.resources.files("op_autoresearch.op.config").joinpath(
    "code_checker.yaml"
).open("r", encoding="utf-8") as _f:
    _RG = yaml.safe_load(_f)["ascendc_anti_cheat"]["runtime_guard"]

# Enforcement mode default; OP_AUTORESEARCH_GUARD_MODE env var overrides at runtime.
DEFAULT_MODE: str = _RG["default_mode"]

# ATen core-compute leaves the runtime gate disables during the candidate forward.
COMPUTE_LEAVES = tuple(_RG["compute_leaves"])
