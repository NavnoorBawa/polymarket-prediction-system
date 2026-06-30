#!/usr/bin/env python3
"""
Generate a comprehensive production report from the live prediction pipeline.
"""

import argparse
import sys

from main import run_single_analysis

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass


def main():
    parser = argparse.ArgumentParser(description="Generate full Polymarket prediction report")
    parser.add_argument(
        "--markets",
        type=int,
        default=20,
        help="Number of markets to include in the live report run",
    )
    parser.add_argument(
        "--report-file",
        default="data\\final_report.json",
        help="Path for the generated JSON report",
    )
    args = parser.parse_args()

    run_single_analysis(
        num_markets=args.markets,
        detailed_report=True,
        report_file=args.report_file,
    )


if __name__ == "__main__":
    main()
