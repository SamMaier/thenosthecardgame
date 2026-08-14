"""Command-line entry point for batch simulations."""

from __future__ import annotations

import argparse

from thenos.simulation import simulate_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate games of The Nos")
    parser.add_argument("games", type=int, nargs="?", default=1000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    report = simulate_games(args.games, args.seed)
    print(f"Games: {report.games}")
    print(
        "Card                 Offers   Picks  Pick%  Owned   Plays  Play%  "
        "Pick Win%  Owned Win%"
    )
    for row in report.rows():
        print(
            f"{row['card']:<20} {row['offers']:>6} {row['picks']:>7} "
            f"{row['pick_rate']:>6.1%} {row['acquisitions']:>6} "
            f"{row['plays']:>7} {row['play_rate']:>6.1%} "
            f"{row['win_rate_when_picked']:>9.1%} "
            f"{row['win_rate_when_acquired']:>10.1%}"
        )


if __name__ == "__main__":
    main()
