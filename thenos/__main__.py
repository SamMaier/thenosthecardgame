"""Command-line entry point for batch simulations."""

from __future__ import annotations

import argparse

from thenos.simulation import (
    simulate_genius_vs_planner,
    simulate_games,
    simulate_greedy_vs_random,
    simulate_planner_vs_greedy,
)


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
        "--genius-vs-planner",
        action="store_true",
        help="seat-balance one Genius AI against three Planner AIs",
    )
    args = parser.parse_args()

    matchup_modes = sum(
        (args.greedy_vs_random, args.planner_vs_greedy, args.genius_vs_planner)
    )
    if matchup_modes > 1:
        parser.error("choose only one matchup mode")

    if args.genius_vs_planner:
        report = simulate_genius_vs_planner(
            args.games, args.seed, workers=args.workers
        )
    elif args.planner_vs_greedy:
        report = simulate_planner_vs_greedy(
            args.games, args.seed, workers=args.workers
        )
    elif args.greedy_vs_random:
        report = simulate_greedy_vs_random(
            args.games, args.seed, workers=args.workers
        )
    else:
        report = simulate_games(args.games, args.seed, workers=args.workers)
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
