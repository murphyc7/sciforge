#!/usr/bin/env python3
"""Benchmarks trained PINNs against classical reference solutions.

Trains each requested solver profile across several seeds and reports the relative L2
error against the matching reference solution. Requires no database connection.
"""

import argparse
import logging
from pathlib import Path

from sciforge.matkit.solvers.benchmark import (
    SUPPORTED_ENGINES,
    BenchmarkResult,
    format_results_table,
    run_benchmark,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SciForge PINN Validation Benchmark")
    parser.add_argument(
        "--engine",
        type=str,
        choices=[*SUPPORTED_ENGINES, "all"],
        default="all",
        help="Physics mode to benchmark",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Optimisation steps")
    parser.add_argument(
        "--seeds", type=int, default=3, help="Number of seeds to run (0..N-1)"
    )
    parser.add_argument(
        "--effective-mass", type=float, default=0.067, help="Effective mass (m_e)"
    )
    parser.add_argument(
        "--permittivity", type=float, default=12.9, help="Relative permittivity"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to write the table"
    )
    args = parser.parse_args()

    engines = SUPPORTED_ENGINES if args.engine == "all" else (args.engine,)
    results: list[BenchmarkResult] = []
    for engine in engines:
        for seed in range(args.seeds):
            logger.info(
                f"Benchmarking {engine.upper()} | seed={seed} | epochs={args.epochs}"
            )
            results.append(
                run_benchmark(
                    engine_mode=engine,
                    epochs=args.epochs,
                    seed=seed,
                    effective_mass=args.effective_mass,
                    permittivity=args.permittivity,
                )
            )

    table = format_results_table(results)
    print(f"\n{table}\n")

    if args.output is not None:
        args.output.write_text(table + "\n")
        logger.info(f"Results table written to {args.output}")


if __name__ == "__main__":
    main()
