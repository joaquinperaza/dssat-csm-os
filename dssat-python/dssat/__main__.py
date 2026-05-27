"""Command-line entry point: ``python -m dssat`` or ``dssat``."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    """Run a DSSAT simulation from the command line.

    Usage::

        dssat run experiment.json [--output output.json]
    """
    parser = argparse.ArgumentParser(
        prog="dssat",
        description="DSSAT Cropping System Model — Python edition",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a simulation from a JSON experiment file")
    run_p.add_argument("experiment", help="Path to experiment JSON file")
    run_p.add_argument(
        "--output", "-o", default=None,
        help="Write summary JSON to this file (default: print to stdout)",
    )
    run_p.add_argument(
        "--daily", "-d", default=None,
        help="Write daily time-series CSV to this file",
    )

    args = parser.parse_args()

    if args.command == "run":
        from dssat import Simulation
        sim = Simulation.from_json(args.experiment)
        results = sim.run()

        summary = results["summary"]
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Summary written to {args.output}")
        else:
            print(json.dumps(summary, indent=2))

        if args.daily:
            results["daily"].to_csv(args.daily, index=False)
            print(f"Daily output written to {args.daily}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
