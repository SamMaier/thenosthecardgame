"""Command-line entry point for batch simulations."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from thenos.simulation import (
    simulate_four_megamind,
    simulate_games,
    simulate_greedy_vs_random,
    simulate_megamind_vs_planner,
    simulate_planner_vs_greedy,
    write_report_csv,
)


def _code_revision() -> str:
    """Return a best-effort revision label without making output depend on Git."""
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except OSError:
        return "unknown"
    if not revision:
        return "unknown"
    return f"{revision}+dirty" if dirty else revision


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate games of The Nos")
    parser.add_argument("games", type=int, nargs="?", default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="worker processes to use (default: 16)",
    )
    parser.add_argument(
        "--greedy-vs-random",
        action="store_true",
        help="seat-balance one Greedy AI against three Random AIs",
    )
    parser.add_argument(
        "--planner-vs-greedy",
        action="store_true",
        help="seat-balance one Planner AI against three Greedy AIs",
    )
    parser.add_argument(
        "--megamind-vs-planner",
        action="store_true",
        help="seat-balance one Megamind AI against three Planner AIs",
    )
    parser.add_argument(
        "--four-megamind",
        action="store_true",
        help="run the standard seat-rotated four-Megamind card-data batch",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="atomically write the complete card report to this CSV file",
    )
    args = parser.parse_args()

    matchup_modes = sum(
        (
            args.greedy_vs_random,
            args.planner_vs_greedy,
            args.megamind_vs_planner,
            args.four_megamind,
        )
    )
    if matchup_modes > 1:
        parser.error("choose only one matchup mode")
    if args.four_megamind and args.output is None:
        parser.error(
            "--four-megamind requires --output so card stats are preserved"
        )

    revision = _code_revision()
    started = time.perf_counter()
    if args.four_megamind:
        report = simulate_four_megamind(
            args.games, args.seed, workers=args.workers
        )
        run_mode = "four-megamind"
        competitors = "Megamind,Megamind,Megamind,Megamind"
        rotate_seats = True
    elif args.megamind_vs_planner:
        report = simulate_megamind_vs_planner(
            args.games, args.seed, workers=args.workers
        )
        run_mode = "megamind-vs-planner"
        competitors = "Megamind,Planner,Planner,Planner"
        rotate_seats = True
    elif args.planner_vs_greedy:
        report = simulate_planner_vs_greedy(
            args.games, args.seed, workers=args.workers
        )
        run_mode = "planner-vs-greedy"
        competitors = "Planner,Greedy,Greedy,Greedy"
        rotate_seats = True
    elif args.greedy_vs_random:
        report = simulate_greedy_vs_random(
            args.games, args.seed, workers=args.workers
        )
        run_mode = "greedy-vs-random"
        competitors = "Greedy,Random,Random,Random"
        rotate_seats = True
    else:
        report = simulate_games(args.games, args.seed, workers=args.workers)
        run_mode = "four-random"
        competitors = "Random,Random,Random,Random"
        rotate_seats = False
    elapsed_seconds = time.perf_counter() - started

    if args.output is not None:
        output = write_report_csv(
            report,
            args.output,
            metadata={
                "run_games": report.games,
                "seed": "" if args.seed is None else args.seed,
                "run_mode": run_mode,
                "competitors": competitors,
                "rotate_seats": rotate_seats,
                "workers": args.workers,
                "code_revision": revision,
                "elapsed_seconds": f"{elapsed_seconds:.3f}",
            },
        )
        print(f"Output file: {output.resolve()}")
    print(f"Games: {report.games}")
    if matchup_modes:
        print("AI          Games  Avg score  Win rate")
        for name, stats in report.ais.items():
            print(
                f"{name:<12} {stats.games:>5} "
                f"{stats.average_score:>10.2f} {stats.win_rate:>9.1%}"
            )
        print()
    rows = report.rows()
    card_width = max(len(str(row["card"])) for row in rows)
    print(
        f"{'Card':<{card_width}}  Free pick rate   Win rate   Fun added"
    )
    for row in rows:
        print(
            f"{row['card']:<{card_width}} {row['free_pick_rate']:>14.1%} "
            f"{row['win_rate']:>10.1%} {row['fun_added']:>+11.2f}"
        )


if __name__ == "__main__":
    main()
