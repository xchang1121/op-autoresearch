#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
accuracy debug - error Analysis Tool

Analyse the error between the operator output and the desired value and provide a detailed statistical report on error.
"""

import numpy as np
import sys


def analyze_error(pred_file, truth_file, rtol=1e-5, atol=1e-6):
    """
    Analyse error between predicted and real values

    Args:
        Pred_file: Forecasting outcome document path (.npy)
        Truth_file: Real value file path (.npy)
        rtol: relative error tolerance difference
        Atol: Absolute error tolerance difference

    Returns:
        Bool: Validation (pass rate > = 99%)
    """
    try:
        pred = np.load(pred_file)
        truth = np.load(truth_file)
    except Exception as e:
        print(f"Error: Could not load file - {e}")
        return False

    # Check shape
    if pred.shape != truth.shape:
        print(f"Error: shapeDo not match - pred={pred.shape}, truth={truth.shape}")
        return False

    # Calculate error
    abs_error = np.abs(pred - truth)
    rel_error = abs_error / (np.abs(truth) + atol)

    print("=" * 60)
    print("error analysis report")
    print("=" * 60)
    print(f"Projection documents: {pred_file}")
    print(f"Real value files: {truth_file}")
    print(f"Datashape: {pred.shape}")
    print()

    # Absolute error statistics
    print("\"Absolutely error Statistics\"")
    print(f"  Maximum value: {abs_error.max():.6e}")
    print(f"  Average: {abs_error.mean():.6e}")
    print(f"  Medium: {np.median(abs_error):.6e}")
    print(f"  Standard deviation: {abs_error.std():.6e}")
    print()

    # error Statistics
    print("\"error Statistics\"")
    print(f"  Maximum value: {rel_error.max():.6e}")
    print(f"  Average: {rel_error.mean():.6e}")
    print(f"  Medium: {np.median(rel_error):.6e}")
    print(f"  95Division: {np.percentile(rel_error, 95):.6e}")
    print(f"  99Division: {np.percentile(rel_error, 99):.6e}")
    print()

    # Pass rate
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_count = pass_mask.sum()
    total_count = pass_mask.size
    pass_rate = pass_count / total_count * 100

    print(f"Passage")
    print(f"  Pass.: {pass_count}/{total_count}")
    print(f"  Pass rate: {pass_rate:.2f}%")
    print(f"  Portability: rtol={rtol:.0e}, atol={atol:.0e}")
    print()

    # error distribution
    print("[error Distribution]")
    for threshold in [1e-3, 1e-4, 1e-5, 1e-6]:
        count = (abs_error > threshold).sum()
        rate = count / abs_error.size * 100
        print(f"  error > {threshold:.0e}: {count:6d} ({rate:5.2f}%)")
    print()

    # The worst sample.
    worst_idx = abs_error.argmax()
    worst_pos = np.unravel_index(worst_idx, pred.shape)
    print(f"The worst sample ever.")
    print(f"  Location: {worst_pos}")
    print(f"  Projected: {pred[worst_pos]:.6f}")
    print(f"  True value: {truth[worst_pos]:.6f}")
    print(f"  Absolutely.error: {abs_error[worst_pos]:.6e}")
    print(f"  Relativeerror: {rel_error[worst_pos]:.6e}")
    print()

    # Findings
    if pass_rate >= 99.0:
        print("✓ Authentication: PASS")
        return True
    else:
        print("✗ Authentication: FAIL")

        # Print failed samples (first 10)
        fail_indices = np.where(~pass_mask)
        fail_count = min(10, len(fail_indices[0]))
        if fail_count > 0:
            print()
            print("[failed sample (first 10)]")
            for i in range(fail_count):
                idx = tuple(dim[i] for dim in fail_indices)
                print(f"  @{idx}:")
                print(f"    Projections={pred[idx]:.6f}, Expectations={truth[idx]:.6f}, "
                      f"abs_err={abs_error[idx]:.2e}, rel_err={rel_error[idx]:.2e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 error_analysis.py <output.npy> <expected.npy> [rtol] [atol]")
        print()
        print("Example:")
        print("  python3 error_analysis.py output.npy expected.npy")
        print("  python3 error_analysis.py output.npy expected.npy 1e-3 1e-4  # FP16")
        print("  python3 error_analysis.py output.npy expected.npy 1e-5 1e-6  # FP32")
        sys.exit(1)

    pred_file = sys.argv[1]
    truth_file = sys.argv[2]
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6

    success = analyze_error(pred_file, truth_file, rtol, atol)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
