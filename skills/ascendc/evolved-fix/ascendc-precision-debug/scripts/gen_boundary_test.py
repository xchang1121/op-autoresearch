#!/usr/bin/env python3
"""
accuracy debug - Border Value Test Data Generation Tool

A variety of boundary value test data are generated to validate operator ' s accuracy and Rufus.
"""

import numpy as np
import sys
import argparse


def generate_boundary_cases(shape, dtype, output_dir="."):
    """
    Generate boundary value test data

    Args:
        Shape: Data shape (M, N, K) or (M, N)
        dtype: data type (`fp16', 'fp32', 'int8')
        output_dir: Output directory
    """
    np_type = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
    }[dtype]

    # Boundary value definition
    if dtype == 'fp16':
        boundary_values = {
            "zero": 0.0,
            "tiny": 1e-4,  # FP16 Minimum normal number
            "small": 1e-3,
            "normal": 1.0,
            "large": 100.0,
            "saturation": 65504.0,  # FP16 Max
            "negative": -1.0,
            "neg_saturation": -65504.0,
        }
    elif dtype == 'fp32':
        boundary_values = {
            "zero": 0.0,
            "tiny": 1e-10,
            "small": 1e-6,
            "normal": 1.0,
            "large": 1e6,
            "huge": 1e10,
            "negative": -1.0,
        }
    else:  # int8
        boundary_values = {
            "zero": 0,
            "min": -128,
            "max": 127,
            "normal": 42,
        }

    # Generate test data for each boundary value
    for name, value in boundary_values.items():
        data = np.full(shape, value, dtype=np_type)
        filename = f"{output_dir}/boundary_{name}_{dtype}.npy"
        np.save(filename, data)
        print(f"Generate: {filename} (value={value})")


def generate_random_aligned(shape, dtype, output_dir=".", seed=42):
    """
    Generate 32-byte random test data

    Args:
        Shape: Original shape
        dtype: data type
        output_dir: Output directory
        Seed: Random Feeds
    """
    np_type = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
    }[dtype]

    np.random.seed(seed)

    # Check and align
    element_size = np.dtype(np_type).itemsize
    aligned_size = 32 // element_size

    adjusted_shape = list(shape)
    adjusted_shape[-1] = ((shape[-1] + aligned_size - 1) // aligned_size) * aligned_size

    # Generate Random Data
    data = np.random.rand(*adjusted_shape).astype(np_type)

    filename = f"{output_dir}/random_aligned_{'_'.join(map(str, shape))}_{dtype}.npy"
    np.save(filename, data)

    print(f"Generate: {filename}")
    print(f"  Originalshape: {shape}")
    print(f"  Adjustmentsshape: {tuple(adjusted_shape)} (32Byte Alignment)")
    print(f"  Data range: [{data.min():.6f}, {data.max():.6f}]")


def generate_unaligned(shape, dtype, output_dir=".", seed=42):
    """
    Generate non-matched random test data

    Args:
        Shape: Original shape
        dtype: data type
        output_dir: Output directory
        Seed: Random Feeds
    """
    np_type = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
    }[dtype]

    np.random.seed(seed)

    # Make sure you're not aligned.
    unaligned_shape = list(shape)
    unaligned_shape[-1] = shape[-1] + 1  # Add one to destroy alignment.

    data = np.random.rand(*unaligned_shape).astype(np_type)

    filename = f"{output_dir}/random_unaligned_{'_'.join(map(str, shape))}_{dtype}.npy"
    np.save(filename, data)

    print(f"Generate: {filename}")
    print(f"  shape: {tuple(unaligned_shape)} (Inconsistent)")


def main():
    parser = argparse.ArgumentParser(description="Generate accuracy debug test data")
    parser.add_argument("--shape", nargs="+", type=int, required=True,
                        help="Data shape, e.g.: 8 16 16")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "int8"], default="fp32",
                        help="data type")
    parser.add_argument("--output", default=".",
                        help="Output Directory")
    parser.add_argument("--type", choices=["boundary", "aligned", "unaligned", "all"],
                        default="all", help="data type generated")

    args = parser.parse_args()

    shape = tuple(args.shape)

    if args.type in ["boundary", "all"]:
        print("\n [Generating Border Value Data]")
        generate_boundary_cases(shape, args.dtype, args.output)

    if args.type in ["aligned", "all"]:
        print("\n [Generating Alignment Random Data]")
        generate_random_aligned(shape, args.dtype, args.output)

    if args.type in ["unaligned", "all"]:
        print("\n [Generating non-match random data]")
        generate_unaligned(shape, args.dtype, args.output)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Example of usage:")
        print("  python3 gen_boundary_test.py --shape 8 16 16 --dtype fp16")
        print("  python3 gen_boundary_test.py --shape 8 16 --dtype fp32 --type boundary")
        sys.exit(1)

    main()
