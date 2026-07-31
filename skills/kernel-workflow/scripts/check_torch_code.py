#!/usr/bin/env python3
"""
Torch Task code format to validate scripts

Validation code for Kernel Bench (4 essential components):
1. class Model(nn.Module)
2. def forward(self, ...)
3. def get_inputs()
4. def get_init_inputs()

Usage:
    # Read replacement code files from command line parameters
    python check_torch_code.py path/to/code.py

    # Enter read replacement code from standard (recommended for LLM call)
    echo "import torch..." | python check_torch_code.py --stdin

    # Only static check (no code execution)
    python check_torch_code.py --stdin --static-only

    # Output JSON format
    python check_torch_code.py --stdin --json

Output format:
    [VALIID] Code corresponds to KernelBench format
    [INVALID] Code does not match format + reason
"""

import ast
import sys
import argparse
import json


def check_static(code: str) -> tuple[bool, list[str], list[str]]:
    """
    Static check if the code contains the required components

    Returns:
        (is_valid, missing_components, found_components)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"], []

    has = {
        "Model": False,
        "forward": False,
        "get_inputs": False,
        "get_init_inputs": False
    }

    for node in ast.walk(tree):
        # Check class Model (nn. Modeule)
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for base in node.bases:
                base_name = getattr(base, 'attr', getattr(base, 'id', ''))
                if base_name == "Module":
                    has["Model"] = True
                    # Check forward method
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "forward":
                            has["forward"] = True

        # Check the top layer functions get_inputs / get_init_inputs
        if isinstance(node, ast.FunctionDef) and node.name in ("get_inputs", "get_init_inputs"):
            has[node.name] = True

    found = [k for k, v in has.items() if v]
    missing = [k for k, v in has.items() if not v]

    return len(missing) == 0, missing, found


def check_runtime(code: str) -> tuple[bool, str]:
    """
    runtime check if the code is correctly executed

    Check process:
    1. exec(code)
    2. get_init_inputs()
    3. Model(*init_inputs)
    4. get_inputs()
    5. model.forward(*inputs)

    Returns:
        (is_valid, error_message)
    """
    namespace = {}

    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"exec error: {type(e).__name__}: {e}"

    # Get_init_inputs
    if "get_init_inputs" not in namespace:
        return False, "get_init_inputs not defined"
    try:
        init_inputs = namespace["get_init_inputs"]()
    except Exception as e:
        return False, f"get_init_inputs() error: {type(e).__name__}: {e}"

    # Check Model
    if "Model" not in namespace:
        return False, "Model not defined"
    try:
        model = namespace["Model"](*init_inputs)
    except Exception as e:
        return False, f"Model(*get_init_inputs()) error: {type(e).__name__}: {e}"

    # Check inputs
    if "get_inputs" not in namespace:
        return False, "get_inputs not defined"
    try:
        inputs = namespace["get_inputs"]()
    except Exception as e:
        return False, f"get_inputs() error: {type(e).__name__}: {e}"

    # Check forward
    try:
        model(*inputs)
    except Exception as e:
        return False, f"model.forward(*get_inputs()) error: {type(e).__name__}: {e}"

    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="Verify whether Torch Code matches KernelBench format"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to Python file to verify"
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read replacement code from standard input"
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Only static checks, no code execution."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result in JSON format"
    )

    args = parser.parse_args()

    # Read Replace Code
    if args.stdin:
        code = sys.stdin.read()
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                code = f.read()
        except FileNotFoundError:
            if args.json:
                print(json.dumps({
                    "valid": False,
                    "error": f"File not found: {args.file}"
                }))
            else:
                print(f"[ERROR] File does not exist: {args.file}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Static check
    static_valid, missing, found = check_static(code)

    result = {
        "valid": False,
        "static_check": {
            "passed": static_valid,
            "found_components": found,
            "missing_components": missing
        },
        "runtime_check": None,
        "suggestion": ""
    }

    if not static_valid:
        if missing and missing[0].startswith("SyntaxError"):
            result["error"] = missing[0]
            result["suggestion"] = "Run call_op_task_builder to regenerate the code."
        else:
            result["error"] = f"Missing component: {', '.join(missing)}"
            result["suggestion"] = "Run call_op_task_builder to regenerate the code."

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[INVALID] Code does not match KernelBench Format")
            print(f"Missing component: {', '.join(missing)}")
            print(f"Recommendations: {result['suggestion']}")
        sys.exit(1)

    # runtime Check
    if not args.static_only:
        runtime_valid, runtime_error = check_runtime(code)
        result["runtime_check"] = {
            "passed": runtime_valid,
            "error": runtime_error if not runtime_valid else None
        }

        if not runtime_valid:
            result["error"] = runtime_error
            result["suggestion"] = "Use call_op_task_builder to repair the code."

            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"[INVALID] CoderuntimeCheck failed")
                print(f"error message: {runtime_error}")
                print(f"Recommendations: {result['suggestion']}")
            sys.exit(1)

    # Check passed.
    result["valid"] = True
    check_type = "Static" if args.static_only else "Static +runtime"

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[VALID] Code Matches KernelBench Format{check_type}Checked through)")
        print(f"Include component: {', '.join(found)}")
        print(f"Directly available for generation kernel")

    sys.exit(0)


if __name__ == "__main__":
    main()

