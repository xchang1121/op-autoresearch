import logging
from pathlib import Path
from typing import Optional, Dict, Iterable

from op_autoresearch.op.utils.arch_normalize import CUDA_ARCH_PATTERN, CPU_ARCH_PATTERN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single source of truth — SKU enumeration + per-family DSL whitelists.
#
# A SKU is a concrete arch string KernelVerifier accepts (e.g. ``ascend910b3``
# / ``a100`` / ``rtx4060``). Within a backend, SKUs cluster into FAMILIES that
# share a DSL capability list:
#   - ascend has two families: ``910`` (910B + 910_93xx + 950 — full DSL stack)
#     and ``310`` (310p3 — smaller stack, no triton / no tilelang).
#   - cuda / cpu are single-family (all listed archs share the same DSLs).
#
# Validation strategy differs by backend:
#   - ascend uses **explicit SKU tuples** — Ascend SKUs are discrete real
#     products (no ``ascend910b99``); enumeration gives crisp error msgs.
#   - cuda / cpu use **family regex** — model names are parametric
#     (``rtx<N>``, ``[ahvltb]<N>``, ``x86_64`` / ``aarch64`` / ...);
#     enumerating every SKU OP_AUTORESEARCH might see is busywork. A new RTX 5090 /
#     B300 / H200 is automatically accepted when ``hw_detect`` extracts
#     it from ``nvidia-smi``.
#
# Adding a new ascend SKU: one line in the matching ``_*_SKUS`` tuple.
# Adding a whole new cuda generation: zero code change (regex covers it).
# Adding a new DSL across an entire family: one entry in ``_DSL_TABLE``.
# ---------------------------------------------------------------------------

# Ascend 910-class families (full DSL stack) — explicit SKUs, real products
_ASCEND_910B_SKUS = (
    "ascend910b1", "ascend910b2", "ascend910b2c", "ascend910b3", "ascend910b4",
)
_ASCEND_910_93_SKUS = (
    "ascend910_9362", "ascend910_9372", "ascend910_9381",
    "ascend910_9382", "ascend910_9391", "ascend910_9392",
)
_ASCEND_950_SKUS = (
    "ascend950dt_95a",
    "ascend950pr_950z", "ascend950pr_9572", "ascend950pr_9574", "ascend950pr_9575",
    "ascend950pr_9576", "ascend950pr_9577", "ascend950pr_9578", "ascend950pr_9579",
    "ascend950pr_957b", "ascend950pr_957d", "ascend950pr_9581", "ascend950pr_9582",
    "ascend950pr_9584", "ascend950pr_9587", "ascend950pr_9588", "ascend950pr_9589",
    "ascend950pr_958a", "ascend950pr_958b", "ascend950pr_9591", "ascend950pr_9592",
    "ascend950pr_9599",
)
_ASCEND_910_FAMILY = _ASCEND_910B_SKUS + _ASCEND_910_93_SKUS + _ASCEND_950_SKUS

# Ascend 310 family (reduced stack — no triton / no tilelang / no pypto)
_ASCEND_310_SKUS = ("ascend310p3",)

# CUDA/CPU family patterns live in ``arch_normalize.py`` so detection and
# validation agree without maintaining a separate SKU list here.


# Family → DSL whitelist, keyed by (framework, backend, family-tag).
# family-tag is internal (``910`` / ``310`` / ``any``) — callers use the
# arch string directly and we resolve the family via ``_family_of``.
# Derived from adapters.factory.DSL_REGISTRY (the single source of truth);
# adding a new DSL is one entry in the registry, not two.
def _build_dsl_table():
    from op_autoresearch.op.verifier.adapters.factory import DSL_REGISTRY
    table: dict = {}
    for name, entry in DSL_REGISTRY.items():
        for fbf in entry.support:
            dsls = table.setdefault(fbf, [])
            if name not in dsls:
                dsls.append(name)
        for alias in entry.aliases:
            for fbf in entry.support:
                dsls = table.setdefault(fbf, [])
                if alias not in dsls:
                    dsls.append(alias)
    return {fbf: tuple(dsls) for fbf, dsls in table.items()}


_DSL_TABLE = _build_dsl_table()

# Canonical DSL set (single source — referenced by normalize_dsl / check_dsl
# instead of duplicating the literal list).
_ALL_DSLS = frozenset(
    dsl for dsls in _DSL_TABLE.values() for dsl in dsls
)


def _family_of(backend: str, arch: str) -> Optional[str]:
    """Return the family tag for (backend, arch), or None if the arch
    isn't recognized under that backend. Ascend uses explicit SKU
    membership; cuda / cpu use family regex."""
    if backend == "ascend":
        if arch in _ASCEND_310_SKUS:
            return "310"
        if arch in _ASCEND_910_FAMILY:
            return "910"
        return None
    if backend == "cuda":
        return "any" if CUDA_ARCH_PATTERN.match(arch) else None
    if backend == "cpu":
        return "any" if CPU_ARCH_PATTERN.match(arch) else None
    return None


def arch_hint(backend: str) -> str:
    """User-facing hint string describing what arch values ``backend``
    accepts. Used by error messages in this module + downstream CLI
    validators. Ascend is enumerated (discrete real SKUs); cuda / cpu
    describe the family-regex pattern since they accept anything in the
    family."""
    if backend == "ascend":
        return "/".join(_ASCEND_310_SKUS + _ASCEND_910_FAMILY)
    if backend == "cuda":
        return ("rtx<N> / gtx<N> / [ahvltb]<N> family "
                "(e.g. a100, h100, h200, l40s, b200, rtx4060, rtx5090)")
    if backend == "cpu":
        return "x86_64 / aarch64 / riscv64 / ppc64le"
    return ""


def check_backend_arch(backend: str, arch: str):
    """
    Verify backend match with architecture
    Args:
        Back: Calculate backend name (ascend/cuda/cpu)
        Arch: Hardware architecture name
    """
    if backend not in ("ascend", "cuda", "cpu"):
        raise ValueError("backend must be ascend, cuda or cpu")
    if _family_of(backend, arch) is None:
        raise ValueError(
            f"{backend} backend does not recognize arch={arch} "
            f"(accepted: {arch_hint(backend)})"
        )


def is_supported_arch(backend: str, arch: str) -> bool:
    """Whether ``arch`` belongs to the canonical backend family table."""
    return backend in ("ascend", "cuda", "cpu") \
        and _family_of(backend, arch) is not None


def normalize_dsl(dsl: str, backend: str = None) -> str:
    """
    Normalize the DSL type by converting the generic Triton based on Backend to Triton_cuda or Triton_ascend

    Args:
        dsl: Realization type
        Backend: Hardware backend name (ascend/cuda/cpu) for automatic conversion to triton

    Returns:
        A standardized DSL type

    Raises:
        ValueError: If dsl is \"triton\" but Backend does not provide or is invalid
    """
    dsl = dsl.lower()

    # If it's already a normative type, go straight back.
    if dsl in _ALL_DSLS:
        return dsl

    # If it's a generic triton, it needs to be converted according to Backend.
    if dsl == "triton":
        if backend is None:
            raise ValueError(
                "dsl='triton' is no longer supported. Please use 'triton_cuda' (for CUDA backend) "
                "or 'triton_ascend' (for Ascend backend) explicitly. "
                "Alternatively, provide backend parameter for automatic conversion."
            )
        backend = backend.lower()
        if backend == "cuda":
            return "triton_cuda"
        elif backend == "ascend":
            return "triton_ascend"
        else:
            raise ValueError(
                f"dsl='triton' cannot be used with backend='{backend}'. "
                "Please use 'triton_cuda' (for CUDA) or 'triton_ascend' (for Ascend) explicitly."
            )

    # Other cases returned directly
    return dsl


def check_dsl(dsl: str):
    """
    Validate Implementation Type
    Args:
        dsl: Realization type (triton_cuda/triton_ascend/triton-russia/swft/torch/pypto, etc.)
    """
    if dsl not in _ALL_DSLS:
        raise ValueError(
            f"dsl must be one of {sorted(_ALL_DSLS)}. "
            "Note: 'triton' is no longer supported. Use 'triton_cuda' or 'triton_ascend' instead."
        )


def check_task_type(task_type: str):
    """Validate a task type.

    Args:
        task_type: task type (prescription_only/profile)
    """
    if task_type not in ["precision_only", "profile"]:
        raise ValueError("task_type must be precision_only or profile")


def supported_dsls(framework: str, backend: str, arch: str) -> Optional[tuple]:
    """Return the DSL whitelist for ``(framework, backend, arch)``, or
    None if the combination is not supported. Single canonical lookup —
    every other validator in this module routes through this."""
    family = _family_of(backend, arch)
    if family is None:
        return None
    return _DSL_TABLE.get((framework, backend, family))


# Backward-compat: VALID_CONFIGS is the derived (framework, backend) →
# {arch: dsl_list} view. Only ascend gets per-arch keys (those SKUs are
# enumerated); cuda / cpu inner dicts are empty by design — their
# canonical accept-set is the regex family, not a list of dict keys.
# Use ``supported_dsls(framework, backend, arch)`` for membership checks;
# iterate ``VALID_CONFIGS[fw][be]`` only when you need the ascend SKU
# enumeration.
def _build_valid_configs() -> Dict[str, Dict[str, Dict[str, list]]]:
    table: Dict[str, Dict[str, Dict[str, list]]] = {}
    enumerated_skus = {
        ("ascend", "910"): _ASCEND_910_FAMILY,
        ("ascend", "310"): _ASCEND_310_SKUS,
    }
    for (fw, be, fam), dsls in _DSL_TABLE.items():
        be_table = table.setdefault(fw, {}).setdefault(be, {})
        for sku in enumerated_skus.get((be, fam), ()):
            be_table[sku] = list(dsls)
    return table


VALID_CONFIGS = _build_valid_configs()


def check_task_config(framework: str, backend: str, arch: str, dsl: str):
    """
    Harmonized authentication of dependency between configuration parameters
    Args:
        framework: framework type
        Backend: Hardware backend name
        Arch: Hardware architecture name
        dsl: Realizable type (raditon to triton_cuda or triton_ascend)
    """
    normalized_dsl = normalize_dsl(dsl, backend)

    if framework not in VALID_CONFIGS:
        raise ValueError(f"Unsupported framework: {framework}")
    if backend not in VALID_CONFIGS[framework]:
        raise ValueError(f"Framework {framework} does not support backend: {backend}")

    dsls = supported_dsls(framework, backend, arch)
    if dsls is None:
        # Distinguish "unknown arch under this backend" from "arch known but
        # this framework doesn't support that family" — the second case can
        # happen e.g. for mindspore + ascend910b3 (family 910 has no row
        # under mindspore at the moment? actually it does. example only).
        if _family_of(backend, arch) is None:
            raise ValueError(f"Backend {backend} does not support arch: {arch}")
        raise ValueError(
            f"Framework {framework} does not support arch {arch} on {backend}"
        )

    if normalized_dsl not in dsls:
        raise ValueError(f"Arch {arch} does not support dsl: {normalized_dsl}")

    return normalized_dsl


def collect_and_save_all_examples(
    arch: str,
    dsl: str,
    project_root_path: Path,
    source_dirs: Dict[str, Path],
) -> Optional[Path]:
    """
    Summarize Allexamplesand save to a unified directorydatabase/all_examples/{arch}/{dsl}

    This function consolidates the sample files from different source directories into the directory, and Python files and other files are saved separately.
    - Python files saved to: data/all_examples/{arch}/ {dsl}/code/
    - Save other files to: data/all_examples/{arch}/{dsl}/docs/

    Args:
        Arch: Hardware architecture name
        dsl: DSL type
        Project_root_path: root path of the item
        Source_dirs: Source Dictionary in { \"prefix\": Path(\"source_dir\")}
                    The file will be copied and renamed \"{prefix}_{original filename}\"
                    For example: \"user\": Path (\"user_examples/\"), \"local\": Path ( \"local_examples/\")

    Returns:
        Path: Unify the root path to save the directory, and return None if it fails
    """
    if not arch or not dsl:
        logger.warning("Arch or dsl is empty and cannot summarize the example code")
        return None

    # Create a single save directory
    base_dir = project_root_path / "database" / "all_examples" / arch / dsl
    code_dir = base_dir / "code"
    doc_dir = base_dir / "docs"

    code_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    saved_code_count = 0
    saved_doc_count = 0

    # Walk through all source directories
    for prefix, source_dir in source_dirs.items():
        if not source_dir or not isinstance(source_dir, Path):
            logger.warning(f"Skip invalid source directory: {prefix} -> {source_dir}")
            continue

        if not source_dir.exists():
            logger.warning(f"Source directory does not exist, Skip: {source_dir}")
            continue

        try:
            # If a directory, copy all the files in it
            if source_dir.is_dir():
                for file_path in source_dir.glob("*"):
                    if not file_path.is_file():
                        continue

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        if content:
                            # Select the destination directory by file extension
                            if file_path.suffix == ".py":
                                target_dir = code_dir
                                saved_code_count += 1
                            else:
                                target_dir = doc_dir
                                saved_doc_count += 1

                            # Destination filename: prefix_original filename
                            save_path = target_dir / f"{prefix}_{file_path.name}"
                            with open(save_path, "w", encoding="utf-8") as f:
                                f.write(content)
                    except Exception as e:
                        logger.warning(f"Copy File {file_path} Failed: {e}")

            # If a single file, copy it directly.
            elif source_dir.is_file():
                try:
                    with open(source_dir, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        # Select the destination directory by file extension
                        if source_dir.suffix == ".py":
                            target_dir = code_dir
                            saved_code_count += 1
                        else:
                            target_dir = doc_dir
                            saved_doc_count += 1

                        save_path = target_dir / f"{prefix}_{source_dir.name}"
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(content)
                except Exception as e:
                    logger.warning(f"Copy File {source_dir} Failed: {e}")

        except Exception as e:
            logger.warning(f"Process Source Directory {prefix}:{source_dir} An error occurred: {e}")

    logger.info(f"Summary completed, stored together {saved_code_count} individualPythonFile To: {code_dir}")
    logger.info(f"Summary completed, stored together {saved_doc_count} Other file to: {doc_dir}")
    return code_dir
