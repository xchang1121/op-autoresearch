#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates the time span for the Chrome Trace file.

Range definition: The time difference between the earliest ts to the latest (ts + dur) for all ph = = \"X\" events.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculating the time span of ph=X in trackEvents"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Trace JSON file path"
    )
    return parser.parse_args()


def calc_span(trace_events: list[dict]) -> float:
    x_events = [e for e in trace_events if e.get("ph") == "X"]
    if not x_events:
        raise ValueError("Event not found ph=X")

    min_ts = min(float(e.get("ts", 0) or 0) for e in x_events)
    max_end = max(
        float(e.get("ts", 0) or 0) + float(e.get("dur", 0) or 0)
        for e in x_events
    )
    return max_end - min_ts


def main() -> None:
    args = parse_args()
    input_path = args.input

    if not input_path.exists():
        raise FileNotFoundError(f"File does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trace_events = data.get("traceEvents", [])
    span = calc_span(trace_events)
    print(span)


if __name__ == "__main__":
    main()
