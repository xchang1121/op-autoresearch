"""Configuration data structures shared by the autoresearch framework."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Internal defaults that control agent-loop behavior.

    Task authors normally do not need to change these values. A task can
    override them in the ``agent.config`` block of ``task.yaml``::

        agent:
          config:
            max_consecutive_failures: 20
    """

    # -- Experiment control ------------------------------------------------
    max_consecutive_failures: int = 10
    max_no_edit_turns: int = 3
    max_turns_multiplier: int = 8

    # -- Prompt truncation -------------------------------------------------
    chars_per_token: int = 3
    editable_file_truncate: int = 8_000
    system_context_file_truncate: int = 15_000
    system_context_total_truncate: int = 40_000
    system_fundamentals_max_chars: int = 20_000
    plan_max_chars: int = 4_000
    finish_hint_threshold: int = 2

    # -- Log truncation ---------------------------------------------------
    log_arg_truncate: int = 500
    log_result_truncate: int = 1_000
    cumulative_diff_truncate: int = 10_000
    smoke_output_limit: int = 2_000

    # -- Tool parameters ---------------------------------------------------
    raw_output_tail: int = 2_048
    """Maximum tail length retained from an evaluation's raw output."""

    # -- LLM calls ---------------------------------------------------------
    llm_max_tokens: int = 8_192
    thinking_budget: int = 8_000
    """Anthropic extended thinking budget (tokens). 0 = disabled.
    When enabled, max_tokens is auto-raised to thinking_budget + llm_max_tokens."""
    call_timeout: float = 120.0
    retry_initial_backoff: float = 5.0
    retry_max_backoff_rate_limit: float = 120.0
    retry_max_backoff_other: float = 60.0
    llm_max_retries: int = 5
    llm_connection_check_timeout: float = 15.0

    # -- Context compaction -----------------------------------------------
    context_limit: int | None = 150_000
    """Model context-window size in tokens; 150K is safe for most models."""
    compression_threshold: float = 0.75
    """Trigger compaction when estimated tokens exceed this context ratio."""
    microcompact_min_chars: int = 200
    """Do not compact tool results shorter than this many characters."""
    microcompact_keep_recent: int = 1
    """Number of recent tool results preserved by microcompaction."""
    compact_min_messages: int = 4
    """Do not compact conversations with fewer messages than this value."""
    compact_max_retries: int = 3
    """Maximum retries for each compaction-model call."""
    compact_diagnosis_truncate: int = 2_000
    """Maximum last-diagnosis length included in the bootstrap prompt."""
    compact_post_check_ratio: float = 0.9
    """Require compacted context to remain below this context-limit ratio."""
    compact_max_failures: int = 3
    """Maximum consecutive prompt-too-long failures."""
    compact_emergency_keep_rounds: int = 1
    """Recent rounds retained during emergency compaction."""
    compact_keep_recent_rounds: int = 3
    """Recent rounds retained during normal compaction."""
    # --- Multi-step compact pipeline (operator summary + plan analysis) ---
    compact_op_summary_max_tokens: int = 500
    """Maximum tokens for the operator-summary model call."""
    compact_plan_analysis_max_tokens: int = 1_500
    """Maximum tokens for the plan-analysis model call."""
    compact_kernel_sanity_cap: int = 80_000
    """Normal auto_compact path: per-editable-file char cap. Dominated
    by the LLM context window; tight cap isn't the goal here, safety is."""
    compact_rebuild_kernel_cap: int = 20_000
    """PTL-recovery force_rebuild path: much tighter per-file cap. Ensures
    the rebuilt buffer is strictly smaller than the one that just tripped
    PTL. Size independent from compact_kernel_sanity_cap because the
    recovery path has to shrink; the normal path doesn't."""
    compact_rebuild_ranking_cap: int = 8_000
    """PTL-recovery force_rebuild path: char cap for ranking.md. Normal
    auto_compact keeps ranking.md uncapped (full performance landscape);
    force_rebuild trims it because the whole point of force_rebuild is
    to shed bytes."""
    compact_plan_raw_fallback_chars: int = 6_000
    """Raw plan fallback length when model-based plan analysis fails."""
    replanning_max_idle_turns: int = 2
    """Maximum idle turns allowed during the replanning phase."""

    # -- Feedback and ranking truncation ---------------------------------
    eval_feedback_tail: int = 1_000
    """Maximum raw-output tail included in evaluation feedback."""
    log_raw_output_truncate: int = 4_096
    """Maximum raw-output length retained in JSONL logs."""
    history_summary_last_n: int = 10
    """Number of recent rounds included in the history summary."""
    ranking_description_truncate: int = 100
    """Maximum ranking-description length."""
    ranking_error_truncate: int = 120
    """Maximum ranking-error length."""
    compact_ranking_max_entries: int = 5
    """Maximum entries retained in the compacted ranking file."""

    # -- Skill injection ---------------------------------------------------
    skill_block_max_chars: int = 8_000
    """Total skill-content budget for the initial agent prompt."""
    skill_block_top_k: int = 5
    skill_keyword_max_per_item: int = 5
    """Maximum keywords retained for each ``update_plan`` item."""
    skill_narrow_timeout: float = 30.0
    """Hard timeout in seconds for model-generated skill keywords."""

    # -- Plan item rationale (forced reflection) ----------------------------
    plan_item_rationale_min_chars: int = 30
    """Minimum length for the rationale field on each update_plan item.
    Plans containing items with shorter rationale are rejected whole."""
    plan_item_rationale_max_chars: int = 400
    """Maximum length; longer rationale is truncated with ellipsis."""

    # -- Plan breadth (forced diversification) ------------------------------
    min_items_per_plan: int = 3
    """Minimum number of items a fresh `update_plan` call must contain.
    Forces the agent to articulate several distinct directions up front,
    so settled_history builds pattern signal inside a single plan version
    (instead of each plan being a 1-item reactive tweak of the previous
    outcome). Only applies to fresh submissions; replace_active_item
    (the must_replan surgical path) ignores this floor."""

    # -- Skill content injection (on skill-backed item activation) -----------
    skill_inject_max_chars: int = 6_000
    """Max chars of SKILL.md content injected into the conversation when
    a skill-backed plan item activates. This is a direct content injection."""

    # -- Diagnose ----------------------------------------------------------
    diagnose_suggest_threshold: int = 3
    """Failure count at which diagnosis is suggested."""
    subagent_code_truncate: int = 8_000
    """Maximum code length passed to the diagnosis subagent."""
    subagent_result_truncate: int = 10_000
    """Maximum tool-result length passed to the diagnosis subagent."""
    subagent_max_iterations: int = 15
    """Maximum diagnosis iterations."""

    # -- Runtime files -----------------------------------------------------
    session_dir: str = "agent_session"
    heartbeat_file: str = "RUNNING"


@dataclass
class TaskConfig:
    """Complete configuration for an optimization task.

    ``task.yaml`` declares the task independently of framework code. The
    framework consumes editable-file rules, evaluation settings, metrics,
    and acceptance criteria.
    """

    # Basic information
    name: str
    description: str

    # Adapter-based evaluation. The framework generates an evaluation script
    # when dsl, framework, backend, and arch are all specified.
    dsl: Optional[str] = None
    framework: Optional[str] = None
    backend: Optional[str] = None
    arch: Optional[str] = None
    dsl_config: dict = field(default_factory=dict)

    # Custom evaluation script (overrides adapter generation).
    eval_script: Optional[str] = None
    editable_files: list[str] = field(default_factory=list)

    # Evaluation parameters.
    eval_timeout: int = 600
    primary_metric: str = "score"
    lower_is_better: bool = True
    improvement_threshold: float = 0.0

    # Hard constraints: {metric_name: (operator, threshold)}
    constraints: dict = field(default_factory=dict)

    # Preflight smoke test
    smoke_test_script: Optional[str] = None
    smoke_test_timeout: int = 10
    import_timeout: int = 15

    # Edit guardrails.
    max_patch_size: int = 15_000
    # forbidden_patterns is built by config_loader.build_forbidden_patterns
    # from edit_guardrails.yaml (global + dsl + hardware + framework scopes)
    # merged with any task.yaml override. The default here is empty because
    # the YAML file is the single source of truth for defaults.
    forbidden_patterns: dict = field(default_factory=lambda: {
        "content": [],
        "diff": [],
        "diff_any": [],
    })

    # Agent context files.
    program_file: Optional[str] = None
    ref_file: Optional[str] = None
    context_files: list[str] = field(default_factory=list)

    # Git control
    git_push: bool = False
    git_branch: Optional[str] = None

    # Experiment control.
    max_rounds: int = 30

    # Agent behavior; most tasks can use the framework defaults.
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Task-specific metadata.
    metadata: dict = field(default_factory=dict)


@dataclass
class CommitResult:
    """Structured commit result without ``Optional[str]`` ambiguity."""

    hash: Optional[str] = None
    nothing_to_commit: bool = False
    error: Optional[str] = None

    @property
    def committed(self) -> bool:
        return self.hash is not None

    @property
    def ok(self) -> bool:
        """True if committed or nothing to commit (not an error)."""
        return self.committed or self.nothing_to_commit


@dataclass
class EvalResult:
    """Generic result for one evaluation run."""

    correctness: bool
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    raw_output: str = ""

    def get_metric(self, key: str, default=None):
        return self.metrics.get(key, default)


@dataclass
class RoundRecord:
    """A complete record of the experiment."""

    round_num: int
    description: str
    result: EvalResult
    accepted: bool
    commit_hash: Optional[str] = None
    duration_sec: float = 0.0
    constraint_violations: list[str] = field(default_factory=list)
