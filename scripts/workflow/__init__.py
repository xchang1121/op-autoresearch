"""Workflow transactions between engine CLIs and state_store.

Contains baseline/round transaction bodies, the plan store, and two pure
phase selectors. Engine CLIs enter through ``task_handle.Task``; this package
does not expose a second orchestration facade.
"""
