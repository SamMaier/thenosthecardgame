"""Command-line entry point for batch simulations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
import re
import subprocess
import sys
import time
from pathlib import Path

from thenos.simulation import (
    character_report_path,
    condition_report_path,
    simulate_argonbrain_vs_planner,
    simulate_beryllium_vs_planner,
    simulate_carbon_vs_planner,
    simulate_four_argonbrain,
    simulate_four_beryllium,
    simulate_four_carbon,
    simulate_four_galaxybrain,
    simulate_four_hydrogen,
    simulate_four_osmium,
    simulate_galaxybrain_vs_planner,
    simulate_games,
    simulate_greedy_vs_random,
    simulate_hydrogen_vs_planner,
    simulate_osmium_vs_planner,
    simulate_planner_vs_greedy,
    simulate_two_hydrogen_two_carbon,
    simulate_two_osmium_two_hydrogen,
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


DEFAULT_RESULTS_DIR = Path("results")
_FOUR_OSMIUM_RUN_RE = re.compile(
    r"^four-osmium-(\d+)-(-?\d+)(?:\.(?:daily_conditions|characters))?\.csv$"
)
_FOUR_OSMIUM_LEGACY_SEED_RE = re.compile(
    r"^four-osmium-(?:.*-)?seed-(-?\d+)(?:\.(?:daily_conditions|characters))?\.csv$"
)


def _next_default_osmium_run(
    results_dir: Path | None = None,
    today: date | None = None,
) -> tuple[int, Path]:
    directory = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(
            f"output directory is not a directory: {directory}"
        )
    current_date = date.today() if today is None else today
    base_seed = int(current_date.strftime("%Y%m%d"))

    max_run_number = 0
    used_seeds: set[int] = set()
    if directory.is_dir():
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            match = _FOUR_OSMIUM_RUN_RE.match(entry.name)
            if match is not None:
                max_run_number = max(max_run_number, int(match.group(1)))
                used_seeds.add(int(match.group(2)))
                continue
            legacy_match = _FOUR_OSMIUM_LEGACY_SEED_RE.match(entry.name)
            if legacy_match is not None:
                used_seeds.add(int(legacy_match.group(1)))

    run_number = max_run_number + 1
    seed = base_seed
    while (
        seed in used_seeds
        or (directory / f"four-osmium-{run_number}-{seed}.csv").exists()
    ):
        seed += 1
    return seed, directory / f"four-osmium-{run_number}-{seed}.csv"


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    no_args = len(raw_args) == 0

    parser = argparse.ArgumentParser(description="Simulate games of The Nos")
    parser.add_argument("games", type=int, nargs="?", default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="worker processes to use (default: 32)",
    )
    parser.add_argument(
        "--four-random",
        "--4-random",
        dest="four_random",
        action="store_true",
        help="run an unrotated four-Random plumbing batch",
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
        "--carbon-vs-planner",
        "--carbonbrain-vs-planner",
        dest="carbon_vs_planner",
        action="store_true",
        help="seat-balance one Carbon AI against three Planner AIs",
    )
    parser.add_argument(
        "--four-carbon",
        "--four-carbonbrain",
        dest="four_carbon",
        action="store_true",
        help="run a seat-rotated four-Carbon card-data batch",
    )
    parser.add_argument(
        "--hydrogen-vs-planner",
        "--hydrogenbrain-vs-planner",
        dest="hydrogen_vs_planner",
        action="store_true",
        help="seat-balance one Hydrogen AI against three Planner AIs",
    )
    parser.add_argument(
        "--four-hydrogen",
        "--four-hydrogenbrain",
        "--4-hydrogen",
        dest="four_hydrogen",
        action="store_true",
        help="run a seat-rotated four-Hydrogen card-data batch",
    )
    parser.add_argument(
        "--two-hydrogen-two-carbon",
        "--two-hydrogenbrain-two-carbonbrain",
        "--2-hydrogen-2-carbon",
        dest="two_hydrogen_two_carbon",
        action="store_true",
        help="run a seat-rotated two-Hydrogen, two-Carbon card-data batch",
    )
    parser.add_argument(
        "--osmium-vs-planner",
        "--osmiumbrain-vs-planner",
        dest="osmium_vs_planner",
        action="store_true",
        help="seat-balance one Osmium AI against three Planner AIs",
    )
    parser.add_argument(
        "--four-osmium",
        "--four-osmiumbrain",
        "--4-osmium",
        dest="four_osmium",
        action="store_true",
        help="run the standard seat-rotated four-Osmium card-data batch (default)",
    )
    parser.add_argument(
        "--two-osmium-two-hydrogen",
        "--two-osmiumbrain-two-hydrogenbrain",
        "--2-osmium-2-hydrogen",
        dest="two_osmium_two_hydrogen",
        action="store_true",
        help="run a seat-rotated two-Osmium, two-Hydrogen card-data batch",
    )
    parser.add_argument(
        "--beryllium-vs-planner",
        "--berylliumbrain-vs-planner",
        dest="beryllium_vs_planner",
        action="store_true",
        help="seat-balance one Beryllium AI against three Planner AIs",
    )
    parser.add_argument(
        "--four-beryllium",
        "--four-berylliumbrain",
        dest="four_beryllium",
        action="store_true",
        help="run a seat-rotated four-Beryllium card-data batch",
    )
    parser.add_argument(
        "--argonbrain-vs-planner",
        action="store_true",
        help="seat-balance one Argonbrain AI against three Planner AIs",
    )
    parser.add_argument(
        "--four-argonbrain",
        action="store_true",
        help="run a seat-rotated four-Argonbrain card-data batch",
    )
    parser.add_argument(
        "--galaxybrain-vs-planner",
        action="store_true",
        help="seat-balance one Galaxybrain AI against three Planner AIs",
    )
    parser.add_argument(
        "--four-galaxybrain",
        action="store_true",
        help="run a seat-rotated four-Galaxybrain card-data batch",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="atomically write the complete card report to this CSV file",
    )
    parser.add_argument("--daily-conditions", action="store_true", help="enable the optional Daily Condition deck")
    parser.add_argument(
        "--characters",
        action="store_true",
        help="randomly assign each player a persistent character bonus",
    )
    args = parser.parse_args(raw_args)

    matchup_modes = sum(
        (
            args.four_random,
            args.greedy_vs_random,
            args.planner_vs_greedy,
            args.carbon_vs_planner,
            args.four_carbon,
            args.hydrogen_vs_planner,
            args.four_hydrogen,
            args.two_hydrogen_two_carbon,
            args.osmium_vs_planner,
            args.four_osmium,
            args.two_osmium_two_hydrogen,
            args.beryllium_vs_planner,
            args.four_beryllium,
            args.argonbrain_vs_planner,
            args.four_argonbrain,
            args.galaxybrain_vs_planner,
            args.four_galaxybrain,
        )
    )
    if matchup_modes > 1:
        parser.error("choose only one matchup mode")
    if matchup_modes == 0:
        args.four_osmium = True
    if no_args:
        args.seed, args.output = _next_default_osmium_run()
    if args.four_carbon and args.output is None:
        parser.error(
            "--four-carbon requires --output so card stats are preserved"
        )
    if args.four_hydrogen and args.output is None:
        parser.error(
            "--four-hydrogen requires --output so card stats are preserved"
        )
    if args.two_hydrogen_two_carbon and args.output is None:
        parser.error(
            "--two-hydrogen-two-carbon requires --output so card stats are preserved"
        )
    if args.four_osmium and args.output is None:
        parser.error(
            "--four-osmium requires --output so card stats are preserved"
        )
    if args.two_osmium_two_hydrogen and args.output is None:
        parser.error(
            "--two-osmium-two-hydrogen requires --output so card stats are preserved"
        )
    if args.four_beryllium and args.output is None:
        parser.error(
            "--four-beryllium requires --output so card stats are preserved"
        )
    if args.four_argonbrain and args.output is None:
        parser.error(
            "--four-argonbrain requires --output so card stats are preserved"
        )
    if args.four_galaxybrain and args.output is None:
        parser.error(
            "--four-galaxybrain requires --output so card stats are preserved"
        )

    revision = _code_revision()
    started = time.perf_counter()
    if args.four_carbon:
        report = simulate_four_carbon(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "four-carbon"
        competitors = "Carbon,Carbon,Carbon,Carbon"
        rotate_seats = True
    elif args.carbon_vs_planner:
        report = simulate_carbon_vs_planner(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "carbon-vs-planner"
        competitors = "Carbon,Planner,Planner,Planner"
        rotate_seats = True
    elif args.four_hydrogen:
        report = simulate_four_hydrogen(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "four-hydrogen"
        competitors = "Hydrogen,Hydrogen,Hydrogen,Hydrogen"
        rotate_seats = True
    elif args.two_hydrogen_two_carbon:
        report = simulate_two_hydrogen_two_carbon(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "two-hydrogen-two-carbon"
        competitors = "Hydrogen,Carbon,Hydrogen,Carbon"
        rotate_seats = True
    elif args.hydrogen_vs_planner:
        report = simulate_hydrogen_vs_planner(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "hydrogen-vs-planner"
        competitors = "Hydrogen,Planner,Planner,Planner"
        rotate_seats = True
    elif args.four_osmium:
        report = simulate_four_osmium(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "four-osmium"
        competitors = "Osmium,Osmium,Osmium,Osmium"
        rotate_seats = True
    elif args.two_osmium_two_hydrogen:
        report = simulate_two_osmium_two_hydrogen(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "two-osmium-two-hydrogen"
        competitors = "Osmium,Hydrogen,Osmium,Hydrogen"
        rotate_seats = True
    elif args.osmium_vs_planner:
        report = simulate_osmium_vs_planner(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "osmium-vs-planner"
        competitors = "Osmium,Planner,Planner,Planner"
        rotate_seats = True
    elif args.four_beryllium:
        report = simulate_four_beryllium(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "four-beryllium"
        competitors = "Beryllium,Beryllium,Beryllium,Beryllium"
        rotate_seats = True
    elif args.beryllium_vs_planner:
        report = simulate_beryllium_vs_planner(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "beryllium-vs-planner"
        competitors = "Beryllium,Planner,Planner,Planner"
        rotate_seats = True
    elif args.four_argonbrain:
        report = simulate_four_argonbrain(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "four-argonbrain"
        competitors = "Argonbrain,Argonbrain,Argonbrain,Argonbrain"
        rotate_seats = True
    elif args.argonbrain_vs_planner:
        report = simulate_argonbrain_vs_planner(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "argonbrain-vs-planner"
        competitors = "Argonbrain,Planner,Planner,Planner"
        rotate_seats = True
    elif args.four_galaxybrain:
        report = simulate_four_galaxybrain(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "four-galaxybrain"
        competitors = "Galaxybrain,Galaxybrain,Galaxybrain,Galaxybrain"
        rotate_seats = True
    elif args.galaxybrain_vs_planner:
        report = simulate_galaxybrain_vs_planner(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "galaxybrain-vs-planner"
        competitors = "Galaxybrain,Planner,Planner,Planner"
        rotate_seats = True
    elif args.planner_vs_greedy:
        report = simulate_planner_vs_greedy(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "planner-vs-greedy"
        competitors = "Planner,Greedy,Greedy,Greedy"
        rotate_seats = True
    elif args.greedy_vs_random:
        report = simulate_greedy_vs_random(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
        run_mode = "greedy-vs-random"
        competitors = "Greedy,Random,Random,Random"
        rotate_seats = True
    else:
        report = simulate_games(
            args.games,
            args.seed,
            workers=args.workers,
            daily_conditions=args.daily_conditions,
            characters=args.characters,
        )
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
                "daily_conditions": args.daily_conditions,
                "characters": args.characters,
                "workers": args.workers,
                "code_revision": revision,
                "elapsed_seconds": f"{elapsed_seconds:.3f}",
            },
        )
        print(f"Output file: {output.resolve()}")
        if report.daily_conditions:
            print(f"Daily condition output: {condition_report_path(output).resolve()}")
        if report.characters:
            print(f"Character output: {character_report_path(output).resolve()}")
    print(f"Games: {report.games}")
    print(f"Daily conditions: {args.daily_conditions}")
    print(f"Characters: {args.characters}")
    if run_mode != "four-random":
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
        "   Play rate   Win credit when acquired   First seen win rate"
    )
    for row in rows:
        print(
            f"{row['card']:<{card_width}} {row['free_pick_rate']:>14.1%} "
            f"{row['win_rate']:>10.1%} {row['fun_added']:>+11.2f} "
            f"{row['play_rate']:>11.1%} {row['win_credit_when_acquired']:>26.2f} "
            f"{row['first_seen_win_rate']:>21.1%}"
        )
    if report.daily_conditions:
        print("\nDaily conditions: net Fun per player per day (including immediate effects)")
        print("Difference compares with the overall daily average; observational, not causal.")
        print(f"{'Condition':<22} {'Days':>8} {'Avg Fun':>10} {'Difference':>12}")
        for row in report.condition_rows():
            average = row['average_daily_fun']
            difference = row['fun_difference']
            average_text = "n/a" if average is None else f"{average:.2f}"
            difference_text = "n/a" if difference is None else f"{difference:+.2f}"
            print(f"{row['condition']:<22} {row['days']:>8} {average_text:>10} {difference_text:>12}")
    if report.characters:
        print("\nCharacters: final results per assigned player-game")
        print("Difference compares with the overall Fun average for this run.")
        print(f"{'Character':<12} {'Games':>8} {'Avg Fun':>10} {'Win rate':>10} {'Difference':>12}")
        for row in report.character_rows():
            average = row["average_fun"]
            win_rate = row["win_rate"]
            difference = row["fun_difference"]
            average_text = "n/a" if average is None else f"{average:.2f}"
            win_rate_text = "n/a" if win_rate is None else f"{win_rate:.1%}"
            difference_text = "n/a" if difference is None else f"{difference:+.2f}"
            print(
                f"{row['character']:<12} {row['player_games']:>8} "
                f"{average_text:>10} {win_rate_text:>10} {difference_text:>12}"
            )


if __name__ == "__main__":
    main()
