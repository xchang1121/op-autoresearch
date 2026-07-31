"""Lightweight regression tests for utils.failure_extractor.

Run from repo root:

    python tests/utils/run_failure_extractor_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.failure_extractor import extract_failure_signals, format_for_stdout  # noqa: E402


def _signal(diag, kind: str) -> dict:
    for item in diag.signals:
        if item.get("kind") == kind:
            return item
    raise AssertionError(f"missing signal kind {kind}; got {diag.signals!r}")


def test_runtime_tail_match_wins() -> None:
    log = "\n".join(
        [
            *("CMake Warning: harmless warning" for _ in range(30)),
            "ACL stream synchronize failed, error code:507011",
            *("warning: repeated compiler noise" for _ in range(30)),
            (
                "RuntimeError: ACL error 507035: vector core exception; "
                "CCU instruction address check error"
            ),
        ]
    )

    diag = extract_failure_signals(log)

    assert diag.primary == "ascendc_core_exception"
    assert _signal(diag, "acl_error_code")["error_code"] == 507035
    assert "507011" not in _signal(diag, "acl_error_code")["excerpt"]
    assert diag.python_error and "507035" in diag.python_error


def test_acl_stream_sync_uses_specific_signal() -> None:
    log = "ACL stream synchronize failed, error code:507011"

    diag = extract_failure_signals(log)

    assert diag.primary == "acl_stream_error"
    assert _signal(diag, "acl_stream_error")["error_code"] == 507011


def test_ascendc_compile_error_uses_last_error_line() -> None:
    log = "\n".join(
        [
            "old_kernel.asc:7:9: error: stale earlier compile error",
            *("warning: template instantiation noise" for _ in range(20)),
            (
                "/tmp/build/op_kernel/grid_sampler_3d_kernel.asc:123:45: "
                "error: no matching function for call to DataCopy"
            ),
        ]
    )

    diag = extract_failure_signals(log)
    sig = _signal(diag, "ascendc_compile_error")

    assert diag.primary == "ascendc_compile_error"
    assert sig["line"] == 123
    assert sig["column"] == 45
    assert sig["file"].endswith("grid_sampler_3d_kernel.asc")
    assert "stale earlier" not in sig["excerpt"]


def test_precision_location_accepts_grouped_decimal() -> None:
    log = (
        "AssertionError: output mismatch\n"
        "Location [3]: ref=1.0 impl=1.2 abs_diff=2.000000e-01 "
        "strict_tol=1.000000e-03\n"
    )

    diag = extract_failure_signals(log)
    sig = _signal(diag, "precision_fail_location")

    assert sig["abs_diff"] == 0.2
    assert sig["strict_tol"] == 0.001


def test_precision_location_accepts_relaxed_tol() -> None:
    log = (
        "Location [630, 0]: ref=3.283232e+00 impl=9.884769e+00 "
        "abs_diff=6.601537e+00 relaxed_tol=4.105543e-03\n"
    )

    diag = extract_failure_signals(log)
    sig = _signal(diag, "precision_fail_location")

    assert sig["tolerance_kind"] == "relaxed"
    assert sig["relaxed_tol"] == 0.004105543
    assert sig["strict_tol"] is None


def test_precision_assertion_beats_wrapper_error_code() -> None:
    log = "\n".join(
        [
            "[ERROR] 2026-06-17-23:33:54 ERR99999 UNKNOWN application exception",
            "Traceback (most recent call last):",
            "  raise AssertionError(",
            (
                "AssertionError: [case 1] compare: validation failed, "
                "18388 elements exceed the relaxed threshold (hard_fail)"
            ),
            "rtol=1.220000e-04 atol=1.000000e-05",
            "mere=4.703989e-02 mare=3.735385e+02",
        ]
    )

    diag = extract_failure_signals(log)
    sig = _signal(diag, "precision_fail")

    assert diag.primary == "precision_fail"
    assert sig["outlier_count"] == 18388
    assert not any(
        item.get("kind") == "acl_error_code" and item.get("error_code") == 18388
        for item in diag.signals
    )


def test_vector_timeout_beats_precision_assertion() -> None:
    log = "\n".join(
        [
            (
                "AssertionError: [case 0] compare: validation failed, "
                "144265 elements exceed the relaxed threshold (hard_fail)"
            ),
            "rtol=9.770000e-04 atol=1.000000e-03",
            "mere=1.141542e-01 mare=6.303151e+00",
            "[W617 20:16:30] Warning: NPU warning, error code is 507034[Error]:",
            "[Error]: Vector core execution timed out.",
            (
                "EE9999: rtDeviceSynchronizeWithTimeout execution failed, "
                "reason=vector core timeout"
            ),
            "wait for compute device to finish failed, runtime result = 507034.",
        ]
    )

    diag = extract_failure_signals(log)

    assert diag.primary == "vector_core_timeout"
    assert _signal(diag, "vector_core_timeout")["error_code"] == 507034
    assert _signal(diag, "precision_fail")["outlier_count"] == 144265
    assert not any(
        item.get("kind") == "acl_error_code" and item.get("error_code") == 144265
        for item in diag.signals
    )


def test_vector_timeout_beats_generic_core_exception() -> None:
    log = (
        "RuntimeError: ACL error 507034: vector core exception. "
        "Vector core execution timed out."
    )

    diag = extract_failure_signals(log)

    assert diag.primary == "vector_core_timeout"
    assert _signal(diag, "vector_core_timeout")["error_code"] == 507034
    assert _signal(diag, "ascendc_core_exception")["message"] == "vector core exception"


def test_tail_excerpt_filters_warning_flood() -> None:
    log = "\n".join(
        [
            *("CMake Warning: screen-filling warning" for _ in range(80)),
            "RuntimeError: final useful failure line",
        ]
    )

    diag = extract_failure_signals(log)

    assert diag.tail_excerpt
    assert "final useful failure line" in diag.tail_excerpt
    assert "screen-filling warning" not in diag.tail_excerpt
    rendered = format_for_stdout(diag.to_dict())
    assert "RuntimeError: final useful failure line" in rendered


def main() -> None:
    test_runtime_tail_match_wins()
    test_acl_stream_sync_uses_specific_signal()
    test_ascendc_compile_error_uses_last_error_line()
    test_precision_location_accepts_grouped_decimal()
    test_precision_location_accepts_relaxed_tol()
    test_precision_assertion_beats_wrapper_error_code()
    test_vector_timeout_beats_precision_assertion()
    test_vector_timeout_beats_generic_core_exception()
    test_tail_excerpt_filters_warning_flood()
    print("failure_extractor tests passed")


if __name__ == "__main__":
    main()
